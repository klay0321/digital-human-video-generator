import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import knowledge_planner as kp


def _load_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _project_dir_from_arg(project_id_or_path):
    if not project_id_or_path:
        candidates = [
            p for p in Path("outputs").iterdir()
            if p.is_dir() and (p / "knowledge_modules.json").exists()
        ]
        if not candidates:
            raise FileNotFoundError("No outputs/<project_id>/knowledge_modules.json found")
        return max(candidates, key=lambda p: (p / "knowledge_modules.json").stat().st_mtime)
    p = Path(project_id_or_path)
    if p.exists():
        return p
    return Path("outputs") / project_id_or_path


def _load_source_segments(project_dir):
    cleaned = _load_json(project_dir / "cleaned_segments.json", [])
    if cleaned:
        return cleaned
    transcript = _load_json(project_dir / "transcript.json", {})
    return transcript.get("segments") or []


def _build_repaired_render_plan(plan, existing_render_plan=None):
    paths = plan.get("assembly_paths") or []
    first_path = next((p for p in paths if len(p.get("ordered_kp_ids") or []) >= 2), paths[0] if paths else {})
    selected_ids = first_path.get("ordered_kp_ids") or [
        item.get("kp_id") for item in (plan.get("knowledge_points") or []) if item.get("kp_id")
    ]
    render_plan = kp.build_selected_render_plan(
        selected_module_ids=selected_ids,
        selected_fragment_ids=[],
        knowledge_modules=plan,
        fragment_order="user_order",
        voice_style=(existing_render_plan or {}).get("voice_style") or "讲解风格",
        title=first_path.get("path_title") or (existing_render_plan or {}).get("title") or "自动修复后的知识点组合",
    )
    hook = next((h for h in plan.get("big_hooks") or [] if selected_ids and selected_ids[0] in (h.get("recommended_kp_ids") or [])), None)
    render_plan["selected_hook_ids"] = [hook.get("hook_id")] if hook else []
    render_plan["selected_assembly_path_id"] = first_path.get("path_id")
    return render_plan


def repair_project(project_dir, apply=False):
    project_dir = Path(project_dir)
    km_path = project_dir / "knowledge_modules.json"
    rp_path = project_dir / "selected_render_plan.json"
    clips_path = project_dir / "clips.json"
    if not km_path.exists():
        raise FileNotFoundError(km_path)

    original_plan = _load_json(km_path, {})
    original_render_plan = _load_json(rp_path, {})
    source_segments = _load_source_segments(project_dir)

    before_kp_count = len(original_plan.get("knowledge_points") or []) if isinstance(original_plan, dict) else 0
    before_invalid = _count_invalid_fragments(original_plan)
    repaired_plan = kp.repair_semantic_plan_for_render(original_plan, source_segments=source_segments)
    repaired_render_plan = _build_repaired_render_plan(repaired_plan, original_render_plan)
    repaired_clips_data = None
    if clips_path.exists():
        clips_data = _load_json(clips_path, {})
        repaired_clips_data = {
            **clips_data,
            "clips": kp.build_timeline_clips_from_modules(repaired_plan),
            "knowledge_modules": repaired_plan,
        }

    km_repaired_path = project_dir / "knowledge_modules.repaired.json"
    rp_repaired_path = project_dir / "selected_render_plan.repaired.json"
    km_repaired_path.write_text(json.dumps(repaired_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    rp_repaired_path.write_text(json.dumps(repaired_render_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    clips_repaired_path = None
    if repaired_clips_data is not None:
        clips_repaired_path = project_dir / "clips.repaired.json"
        clips_repaired_path.write_text(json.dumps(repaired_clips_data, ensure_ascii=False, indent=2), encoding="utf-8")

    backup_paths = {}
    if apply:
        backup_paths["knowledge_modules"] = _backup_once(km_path, project_dir / "knowledge_modules.before_repair.json")
        if rp_path.exists():
            backup_paths["selected_render_plan"] = _backup_once(rp_path, project_dir / "selected_render_plan.before_repair.json")
        if clips_path.exists():
            backup_paths["clips"] = _backup_once(clips_path, project_dir / "clips.before_repair.json")
        shutil.copyfile(km_repaired_path, km_path)
        shutil.copyfile(rp_repaired_path, rp_path)
        if clips_repaired_path:
            shutil.copyfile(clips_repaired_path, clips_path)

    after_kp_count = len(repaired_plan.get("knowledge_points") or [])
    after_invalid = _count_invalid_fragments(repaired_plan)
    path_lengths = [len(p.get("ordered_kp_ids") or []) for p in repaired_plan.get("assembly_paths") or []]
    return {
        "project_dir": str(project_dir),
        "apply": bool(apply),
        "knowledge_modules_repaired": str(km_repaired_path),
        "selected_render_plan_repaired": str(rp_repaired_path),
        "clips_repaired": str(clips_repaired_path) if clips_repaired_path else None,
        "backup_paths": {k: str(v) for k, v in backup_paths.items()},
        "before_knowledge_point_count": before_kp_count,
        "after_knowledge_point_count": after_kp_count,
        "fragments_repaired": max(0, before_invalid - after_invalid),
        "before_invalid_fragment_count": before_invalid,
        "after_invalid_fragment_count": after_invalid,
        "assembly_path_avg_length": round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else 0,
        "can_enter_video_generation": kp.validate_render_quality(repaired_plan, repaired_render_plan)[0],
    }


def _backup_once(source, backup):
    source = Path(source)
    backup = Path(backup)
    if not backup.exists():
        shutil.copyfile(source, backup)
    return backup


def _count_invalid_fragments(plan):
    count = 0
    if not isinstance(plan, dict):
        return count
    for item in plan.get("knowledge_points") or []:
        for frag in item.get("fragments") or []:
            try:
                start = float(frag.get("start", 0) or 0)
                end = float(frag.get("end", 0) or 0)
            except Exception:
                count += 1
                continue
            if end - start < 1.5:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", nargs="?", help="outputs/<project_id> or project_id. Defaults to latest output with knowledge_modules.json.")
    parser.add_argument("--apply", action="store_true", help="Overwrite knowledge_modules.json and selected_render_plan.json after writing .repaired files.")
    args = parser.parse_args()
    project_dir = _project_dir_from_arg(args.project_id)
    result = repair_project(project_dir, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
