"""Project cleanup scanner for the digital-human demo.

Default mode is a dry-run. Nothing is deleted unless --archive or --apply is
explicitly passed. --archive moves safe cleanup candidates into
_cleanup_archive/<timestamp>/; --apply only deletes existing archive contents.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "_cleanup_archive"
REPORT_MD = PROJECT_ROOT / "cleanup_report.md"
REPORT_JSON = PROJECT_ROOT / "cleanup_report.json"

PROTECTED_FILES = {
    "app.py",
    "pipeline.py",
    ".env",
    ".env.example",
    "README.md",
    "README_BOSS.md",
    "requirements.txt",
    "start_windows.bat",
    "start_windows.ps1",
}
PROTECTED_DIRS = {
    "services",
    "prompts",
    "scripts",
    "demo_assets",
    "config",
    "assets",
    "static",
    "templates",
}
PROTECTED_EXACT_REL = {
    "scripts/cleanup_project.py",
}

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TEMP_DIR_NAMES = {"temp", "tmp", "cache", ".cache", "downloads", "generated", "render_temp", "preview_temp"}
LOG_FILE_NAMES = {"error.log", "debug.log"}
MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wav", ".mp3", ".m4a", ".png", ".jpg", ".jpeg"}
ARCHIVE_EXTS = {".zip", ".7z", ".tar", ".gz"}
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024
PROJECT_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-fA-F]{6}$")


@dataclass
class CleanupItem:
    path: str
    rel_path: str
    kind: str
    category: str
    size_bytes: int
    modified_time: str
    action: str
    reason: str
    risk: str = "low"


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan/archive cleanup candidates.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Only scan and report. Default.")
    group.add_argument("--archive", action="store_true", help="Move safe cleanup candidates into _cleanup_archive.")
    group.add_argument("--apply", action="store_true", help="Permanently delete existing _cleanup_archive contents.")
    args = parser.parse_args()

    mode = "dry-run"
    if args.archive:
        mode = "archive"
    elif args.apply:
        mode = "apply"

    if mode == "apply":
        return apply_archive()

    report = build_report()
    write_reports(report)

    if mode == "archive":
        archive_result = archive_safe_items(report["safe_cleanup_items"])
        report["archive_result"] = archive_result
        write_reports(report)
        print_summary(report, mode="archive")
    else:
        print_summary(report, mode="dry-run")
    return 0


def build_report() -> dict:
    now = datetime.now()
    safe: list[CleanupItem] = []
    confirm: list[CleanupItem] = []
    keep: list[CleanupItem] = []

    outputs_info = scan_outputs(safe, confirm, keep)
    scan_python_caches(safe)
    scan_logs(safe, confirm)
    scan_temp_dirs(safe, confirm)
    large_files = scan_large_files(safe, confirm, keep)
    zip_info = scan_zip_files(safe, confirm, keep)

    safe = dedupe_items(safe)
    confirm = dedupe_items(confirm)
    keep = dedupe_items(keep)

    safe_bytes = sum(item.size_bytes for item in safe)
    return {
        "scan_time": now.isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "safe_cleanup_items": [asdict(item) for item in safe],
        "needs_confirmation_items": [asdict(item) for item in confirm],
        "keep_items": [asdict(item) for item in keep],
        "large_files": large_files,
        "outputs": outputs_info,
        "zip_files": zip_info,
        "estimated_reclaimable_bytes": safe_bytes,
        "estimated_reclaimable_human": human_size(safe_bytes),
        "risk_notes": [
            "dry-run 不会移动或删除任何文件。",
            "archive 模式只移动 safe_cleanup_items 到 _cleanup_archive，不永久删除。",
            "清理 seedance_cache 后，未来再次生成数字人可能重新调用 Seedance API。",
            "带 final_videos 的历史项目已放入 needs_confirmation，避免误删有价值 demo 结果。",
            "源码、prompt、services、核心 scripts、.env 和可运行配置均受保护。",
        ],
    }


def scan_outputs(safe: list[CleanupItem], confirm: list[CleanupItem], keep: list[CleanupItem]) -> dict:
    outputs = PROJECT_ROOT / "outputs"
    info = {
        "exists": outputs.exists(),
        "kept_recent_project_dirs": [],
        "safe_old_project_dirs": [],
        "confirm_project_dirs_with_final_videos": [],
        "cache_or_probe_dirs": [],
        "other_output_items": [],
    }
    if not outputs.exists():
        return info

    child_dirs = [p for p in outputs.iterdir() if p.is_dir()]
    project_dirs = [p for p in child_dirs if PROJECT_ID_RE.match(p.name)]
    project_dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    keep_projects = set(project_dirs[:2])
    for p in project_dirs[:2]:
        item = make_item(p, "directory", "outputs_recent_project", "keep", "最近 2 个输出项目，建议保留。", "low")
        keep.append(item)
        info["kept_recent_project_dirs"].append(item.rel_path)

    for p in project_dirs[2:]:
        has_final_videos = any((p / "final_videos").glob("*.mp4")) if (p / "final_videos").exists() else False
        if has_final_videos:
            item = make_item(p, "directory", "outputs_old_project_with_final_videos", "confirm", "历史项目含 final_videos，可能是有价值 demo，需确认后再归档。", "medium")
            confirm.append(item)
            info["confirm_project_dirs_with_final_videos"].append(item.rel_path)
        else:
            item = make_item(p, "directory", "outputs_old_project", "archive", "老历史输出项目，非最近 2 个。", "low")
            safe.append(item)
            info["safe_old_project_dirs"].append(item.rel_path)

    cache_names = {
        "seedance_cache",
        "seedance_t2v_cache",
        "seedance_model_probe",
        "seedance_i2v_model_probe",
        "seedance_capability_test",
        "seedance_lipsync_url_test",
    }
    cache_prefixes = {
        "seedance_chain_smoke_",
        "seedance_fast_fix_test_",
        "seedance_t2v_main_",
        "seedance_t2v_virtual_once",
    }
    for p in child_dirs:
        if p in keep_projects or PROJECT_ID_RE.match(p.name):
            continue
        if p.name in cache_names or any(p.name.startswith(prefix) for prefix in cache_prefixes):
            item = make_item(p, "directory", "outputs_cache_or_probe", "archive", "Seedance/探测/冒烟测试缓存，可归档；清理后可能需要重新调用 API 生成。", "medium")
            safe.append(item)
            info["cache_or_probe_dirs"].append(item.rel_path)

    for p in outputs.iterdir():
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            item = make_item(p, "file", "outputs_temp_media", "archive", "outputs 根目录临时媒体文件。", "low")
            safe.append(item)
            info["other_output_items"].append(item.rel_path)
        elif p.is_dir() and p.name == "uploads":
            item = make_item(p, "directory", "outputs_uploads_temp", "archive", "历史上传临时目录。", "low")
            safe.append(item)
            info["other_output_items"].append(item.rel_path)
    return info


def scan_python_caches(safe: list[CleanupItem]) -> None:
    for p in walk_project():
        if p.is_dir() and p.name in CACHE_DIR_NAMES:
            safe.append(make_item(p, "directory", "python_cache", "archive", "Python/工具缓存目录。", "low"))
        elif p.is_file() and p.suffix in {".pyc", ".pyo"}:
            safe.append(make_item(p, "file", "python_cache", "archive", "Python 编译缓存文件。", "low"))


def scan_logs(safe: list[CleanupItem], confirm: list[CleanupItem]) -> None:
    for p in walk_project():
        if p.is_dir() and p.name == "logs":
            confirm.append(make_item(p, "directory", "logs", "confirm", "日志目录，需确认是否包含近期关键报错。", "medium"))
        elif p.is_file() and (p.suffix.lower() == ".log" or p.name in LOG_FILE_NAMES):
            critical = log_may_contain_recent_error(p)
            target = confirm if critical else safe
            action = "confirm" if critical else "archive"
            reason = "日志可能包含近期关键报错，先保留待确认。" if critical else "普通日志文件。"
            target.append(make_item(p, "file", "logs", action, reason, "medium" if critical else "low"))


def scan_temp_dirs(safe: list[CleanupItem], confirm: list[CleanupItem]) -> None:
    for p in walk_project():
        if p.is_dir() and p.name in TEMP_DIR_NAMES:
            if p.parts and p.name in {"cache", ".cache"} and p.parent.name in PROTECTED_DIRS:
                confirm.append(make_item(p, "directory", "temp_dir", "confirm", "位于受保护目录下的 cache，需确认是否为静态资源。", "medium"))
            else:
                safe.append(make_item(p, "directory", "temp_dir", "archive", "临时/缓存目录。", "low"))


def scan_large_files(safe: list[CleanupItem], confirm: list[CleanupItem], keep: list[CleanupItem]) -> list[dict]:
    large = []
    safe_paths = {Path(item.path).resolve() for item in safe}
    confirm_paths = {Path(item.path).resolve() for item in confirm}
    keep_paths = {Path(item.path).resolve() for item in keep}
    for p in walk_project(include_files=True, include_dirs=False):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size <= LARGE_FILE_THRESHOLD:
            continue
        rel = rel_path(p)
        suggested = "confirm"
        reason = "大文件，需要人工确认。"
        if any(is_under(p, s) for s in safe_paths):
            suggested = "archive"
            reason = "位于可清理输出/缓存目录内。"
        elif any(is_under(p, c) for c in confirm_paths):
            suggested = "confirm"
            reason = "位于需确认的历史输出目录内。"
        elif any(is_under(p, k) for k in keep_paths):
            suggested = "keep"
            reason = "位于建议保留的最近输出项目内。"
        large.append({
            "path": str(p),
            "rel_path": rel,
            "size_bytes": size,
            "size_human": human_size(size),
            "modified_time": modified_time(p),
            "suggested_action": suggested,
            "reason": reason,
        })
    return sorted(large, key=lambda item: item["size_bytes"], reverse=True)


def scan_zip_files(safe: list[CleanupItem], confirm: list[CleanupItem], keep: list[CleanupItem]) -> dict:
    zip_files = []
    candidates = list(PROJECT_ROOT.glob("*.zip"))
    parent = PROJECT_ROOT.parent
    candidates.extend(parent.glob("*_boss_demo_*.zip"))
    unique = sorted({p.resolve() for p in candidates}, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    boss_zips = [p for p in unique if "_boss_demo_" in p.name]
    latest_boss = boss_zips[0] if boss_zips else None
    for p in unique:
        if not p.exists():
            continue
        action = "confirm"
        reason = "zip 包，需确认是否仍需保留。"
        if latest_boss and p == latest_boss:
            action = "keep"
            reason = "最新老板交付 zip，建议保留。"
            keep.append(make_item(p, "file", "zip", "keep", reason, "low"))
        elif "_boss_demo_" in p.name:
            action = "confirm"
            reason = "旧老板交付 zip，可清理但建议先确认。"
            confirm.append(make_item(p, "file", "zip", "confirm", reason, "medium"))
        elif p.is_relative_to(PROJECT_ROOT):
            confirm.append(make_item(p, "file", "zip", "confirm", reason, "medium"))
        zip_files.append({
            "path": str(p),
            "rel_path": rel_path(p) if is_under(p, PROJECT_ROOT) else str(p),
            "size_bytes": safe_stat_size(p),
            "size_human": human_size(safe_stat_size(p)),
            "modified_time": modified_time(p),
            "suggested_action": action,
            "reason": reason,
        })
    return {"latest_boss_demo_zip": str(latest_boss) if latest_boss else None, "items": zip_files}


def archive_safe_items(items: list[dict]) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ARCHIVE_ROOT / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    selected = collapse_nested_items([Path(item["path"]) for item in items])
    moved = []
    errors = []
    total = 0
    for src in selected:
        try:
            enforce_protection(src)
            if not src.exists():
                continue
            size = path_size(src)
            rel = rel_path(src)
            dest = archive_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved.append({"from": str(src), "to": str(dest), "size_bytes": size})
            total += size
        except Exception as exc:
            errors.append({"path": str(src), "error": str(exc)})
    return {
        "archive_dir": str(archive_dir),
        "moved_count": len(moved),
        "moved_size_bytes": total,
        "moved_size_human": human_size(total),
        "moved": moved,
        "errors": errors,
    }


def apply_archive() -> int:
    if not ARCHIVE_ROOT.exists():
        print("No _cleanup_archive directory exists. Nothing to apply.")
        return 0
    total = path_size(ARCHIVE_ROOT)
    children = list(ARCHIVE_ROOT.iterdir())
    for child in children:
        enforce_archive_path(child)
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(json.dumps({
        "applied": True,
        "deleted_archive_children": len(children),
        "deleted_size_bytes": total,
        "deleted_size_human": human_size(total),
    }, ensure_ascii=False, indent=2))
    return 0


def write_reports(report: dict) -> None:
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    lines = [
        "# 项目清理报告",
        "",
        f"- 扫描时间：{report['scan_time']}",
        f"- 项目根目录：`{report['project_root']}`",
        f"- 预计可释放空间：**{report['estimated_reclaimable_human']}**",
        f"- 可安全归档项：**{len(report['safe_cleanup_items'])}**",
        f"- 需要确认项：**{len(report['needs_confirmation_items'])}**",
        f"- 建议保留项：**{len(report['keep_items'])}**",
        "",
        "## 可安全清理项",
    ]
    lines.extend(render_item_table(report["safe_cleanup_items"]))
    lines.extend(["", "## 需要确认的项"])
    lines.extend(render_item_table(report["needs_confirmation_items"]))
    lines.extend(["", "## 不建议删除 / 建议保留"])
    lines.extend(render_item_table(report["keep_items"]))
    lines.extend(["", "## 大文件列表"])
    if report["large_files"]:
        lines.append("| 路径 | 大小 | 修改时间 | 建议 | 理由 |")
        lines.append("|---|---:|---|---|---|")
        for item in report["large_files"]:
            lines.append(
                f"| `{item['rel_path']}` | {item['size_human']} | {item['modified_time']} | "
                f"{item['suggested_action']} | {item['reason']} |"
            )
    else:
        lines.append("未发现超过 100MB 的文件。")
    lines.extend(["", "## outputs 历史项目"])
    outputs = report["outputs"]
    lines.append(f"- 保留最近项目：{', '.join(outputs.get('kept_recent_project_dirs') or ['无'])}")
    lines.append(f"- 可归档老项目：{', '.join(outputs.get('safe_old_project_dirs') or ['无'])}")
    lines.append(f"- 含 final_videos 需确认项目：{', '.join(outputs.get('confirm_project_dirs_with_final_videos') or ['无'])}")
    lines.append(f"- Seedance/探测缓存：{', '.join(outputs.get('cache_or_probe_dirs') or ['无'])}")
    lines.extend(["", "## 老 zip / 老板交付包"])
    latest_zip = report["zip_files"].get("latest_boss_demo_zip")
    lines.append(f"- 建议保留最新老板交付 zip：`{latest_zip or '未发现'}`")
    for item in report["zip_files"].get("items") or []:
        lines.append(f"- `{item['path']}` - {item['size_human']} - {item['suggested_action']} - {item['reason']}")
    lines.extend(["", "## 清理后风险说明"])
    for note in report["risk_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## 下一步建议"])
    lines.append("当前只建议先检查本报告；如确认无误，再运行：")
    lines.append("")
    lines.append("```powershell")
    lines.append("python scripts/cleanup_project.py --archive")
    lines.append("```")
    return "\n".join(lines) + "\n"


def render_item_table(items: list[dict]) -> list[str]:
    if not items:
        return ["无。"]
    lines = ["| 路径 | 类型 | 分类 | 大小 | 动作 | 理由 |", "|---|---|---|---:|---|---|"]
    for item in items:
        lines.append(
            f"| `{item['rel_path']}` | {item['kind']} | {item['category']} | "
            f"{human_size(item['size_bytes'])} | {item['action']} | {item['reason']} |"
        )
    return lines


def print_summary(report: dict, mode: str) -> None:
    payload = {
        "mode": mode,
        "cleanup_report_md": str(REPORT_MD),
        "cleanup_report_json": str(REPORT_JSON),
        "safe_cleanup_count": len(report["safe_cleanup_items"]),
        "needs_confirmation_count": len(report["needs_confirmation_items"]),
        "estimated_reclaimable_bytes": report["estimated_reclaimable_bytes"],
        "estimated_reclaimable_human": report["estimated_reclaimable_human"],
        "kept_recent_project_dirs": report["outputs"].get("kept_recent_project_dirs"),
        "safe_old_project_dirs": report["outputs"].get("safe_old_project_dirs"),
        "confirm_project_dirs_with_final_videos": report["outputs"].get("confirm_project_dirs_with_final_videos"),
        "large_file_count": len(report["large_files"]),
        "latest_boss_demo_zip": report["zip_files"].get("latest_boss_demo_zip"),
        "archive_result": report.get("archive_result"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def walk_project(include_files: bool = True, include_dirs: bool = True):
    skip_dirs = {".git", "_cleanup_archive"}
    for p in PROJECT_ROOT.rglob("*"):
        if any(part in skip_dirs for part in p.relative_to(PROJECT_ROOT).parts):
            continue
        if p.is_dir() and include_dirs:
            yield p
        elif p.is_file() and include_files:
            yield p


def make_item(path: Path, kind: str, category: str, action: str, reason: str, risk: str) -> CleanupItem:
    enforce_protection(path, action=action)
    return CleanupItem(
        path=str(path.resolve()),
        rel_path=rel_path(path),
        kind=kind,
        category=category,
        size_bytes=path_size(path),
        modified_time=modified_time(path),
        action=action,
        reason=reason,
        risk=risk,
    )


def enforce_protection(path: Path, action: str = "archive") -> None:
    resolved = path.resolve()
    outside_project = not is_under(resolved, PROJECT_ROOT)
    if outside_project:
        if "_boss_demo_" in path.name and action in {"keep", "confirm"}:
            return
        raise RuntimeError(f"Refusing to handle path outside project root: {path}")
    rel = normal_rel(path)
    name = path.name
    if rel in PROTECTED_EXACT_REL or name in PROTECTED_FILES:
        raise RuntimeError(f"Protected path cannot be {action}d: {rel}")
    if rel in PROTECTED_DIRS:
        raise RuntimeError(f"Protected directory cannot be {action}d: {rel}")
    first = rel.split("/", 1)[0]
    if first in PROTECTED_DIRS:
        allowed_cache = (
            path.name in CACHE_DIR_NAMES
            or path.suffix in {".pyc", ".pyo"}
            or "__pycache__" in path.parts
        )
        if not allowed_cache:
            raise RuntimeError(f"Protected source/static path cannot be {action}d: {rel}")


def enforce_archive_path(path: Path) -> None:
    resolved = path.resolve()
    if not is_under(resolved, ARCHIVE_ROOT.resolve()):
        raise RuntimeError(f"Refusing to apply outside archive: {path}")


def collapse_nested_items(paths: list[Path]) -> list[Path]:
    resolved = sorted({p.resolve() for p in paths}, key=lambda p: len(p.parts))
    selected = []
    for p in resolved:
        if any(is_under(p, parent) and p != parent for parent in selected):
            continue
        selected.append(p)
    return selected


def dedupe_items(items: list[CleanupItem]) -> list[CleanupItem]:
    seen = set()
    out = []
    for item in sorted(items, key=lambda i: (i.category, i.rel_path)):
        key = str(Path(item.path).resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def log_may_contain_recent_error(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        age_seconds = datetime.now().timestamp() - path.stat().st_mtime
        text = path.read_text(encoding="utf-8", errors="ignore")[-12000:]
        has_error = any(token in text for token in ("Traceback", "ERROR", "Exception", "失败", "error"))
        return age_seconds < 7 * 24 * 3600 and has_error
    except Exception:
        return True


def path_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(safe_stat_size(p) for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0
    return 0


def safe_stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def rel_path(path: Path) -> str:
    try:
        return normal_rel(path)
    except Exception:
        return str(path)


def normal_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def human_size(size: int) -> str:
    value = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


if __name__ == "__main__":
    raise SystemExit(main())
