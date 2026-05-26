import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.knowledge_planner import calculate_kp_duration


BAD_HOOK_TITLES = {"mcp", "agent", "操作", "介绍", "项目介绍", "功能说明", "工具讲解", "第一段内容"}
BAD_KP_TITLES = {"项目介绍", "功能说明", "工具讲解", "操作", "介绍", "第一段", "第二段", "第三段", "内容总结"}
VALID_KP_TYPES = {
    "problem", "concept", "principle", "cause", "operation", "implementation",
    "workflow", "tool_usage", "case", "business_value", "comparison", "pitfall",
    "decision", "summary", "transition",
}
REQUIRED_PATH_TYPES = {"boss_report", "tutorial", "operation_demo"}
VALID_DISCARD_TYPES = {"repetition", "filler", "off_topic", "unclear", "low_value"}
FILLER_PATTERNS = ("嗯", "就是", "然后这个", "那个", "啊", "额", "呃")
HOOK_ANGLE_PATTERNS = ("为什么", "如何", "怎么", "从", "到", "真正", "关键", "不是", "而是", "降低", "提升", "避免", "搞懂")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _norm_text(text):
    return re.sub(r"\s+", "", str(text or "")).lower()


def _similarity(a, b):
    a = _norm_text(a)
    b = _norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _score_average(values):
    values = [v for v in values if v is not None]
    if not values:
        return 0
    return int(round(sum(values) / len(values)))


def _clip_score(value):
    return max(0, min(100, int(round(value))))


def _is_generic_title(title, bad_titles):
    normalized = _norm_text(title)
    if not normalized:
        return True
    if normalized in {_norm_text(x) for x in bad_titles}:
        return True
    return bool(re.fullmatch(r"第[一二三四五六七八九十0-9]+段.*", str(title or "").strip()))


def _hook_score(hook):
    title = str(hook.get("hook_title") or "").strip()
    score = 100
    issues = []
    if _is_generic_title(title, BAD_HOOK_TITLES):
        score -= 45
        issues.append("hook_title too generic")
    if len(title) < 8:
        score -= 20
        issues.append("hook_title too short")
    if not any(p in title for p in HOOK_ANGLE_PATTERNS):
        score -= 20
        issues.append("missing short-video angle")
    if not hook.get("opening_hook"):
        score -= 10
        issues.append("missing opening_hook")
    if not hook.get("recommended_kp_ids"):
        score -= 15
        issues.append("missing recommended_kp_ids")
    return _clip_score(score), issues


def _kp_specificity_score(kp):
    title = str(kp.get("kp_title") or "").strip()
    score = 100
    issues = []
    if _is_generic_title(title, BAD_KP_TITLES):
        score -= 35
        issues.append("title too generic")
    if len(title) < 8:
        score -= 15
        issues.append("title too short")
    if not kp.get("kp_summary"):
        score -= 15
        issues.append("missing kp_summary")
    if not kp.get("selection_reason"):
        score -= 10
        issues.append("missing selection_reason")
    if not kp.get("fragments"):
        score -= 20
        issues.append("missing fragments")
    return _clip_score(score), issues


def _fragment_issues(kp):
    issues = []
    durations = []
    for frag in kp.get("fragments") or []:
        start = frag.get("start")
        end = frag.get("end")
        try:
            start = float(start)
            end = float(end)
        except Exception:
            issues.append(f"{frag.get('fragment_id')}: invalid timestamp")
            continue
        duration = end - start
        durations.append(duration)
        if not start < end:
            issues.append(f"{frag.get('fragment_id')}: start >= end")
        elif duration < 1.0:
            issues.append(f"{frag.get('fragment_id')}: duration too short ({duration:.1f}s)")
    return issues, durations


def _voice_script_score(kp, unit_by_id, atom_by_id):
    script = str(kp.get("voice_script") or "").strip()
    score = 100
    issues = []
    direct_copy = False
    if not script:
        return 0, ["missing voice_script"], False
    for filler in FILLER_PATTERNS:
        if filler in script:
            score -= 12
            issues.append(f"contains filler: {filler}")
    evidence_parts = []
    for unit_id in kp.get("source_unit_ids") or []:
        unit = unit_by_id.get(unit_id) or {}
        evidence_parts.append(unit.get("raw_text") or unit.get("clean_text") or "")
    for atom_id in kp.get("source_atom_ids") or []:
        atom = atom_by_id.get(atom_id) or {}
        evidence_parts.append(atom.get("evidence_text") or atom.get("atom_text") or "")
    for frag in kp.get("fragments") or []:
        evidence_parts.append(frag.get("raw_text") or frag.get("clean_text") or frag.get("cleaned_text") or "")
    evidence = " ".join(p for p in evidence_parts if p)
    sim = _similarity(script, evidence)
    if sim > 0.92 and len(_norm_text(script)) > 20:
        score -= 35
        direct_copy = True
        issues.append(f"voice_script may directly copy source text (similarity={sim:.2f})")
    if len(script) < 12:
        score -= 20
        issues.append("voice_script too short")
    return _clip_score(score), issues, direct_copy


def _subtitle_score(kp, unit_by_id):
    lines = kp.get("subtitle_lines") or []
    script = _norm_text(kp.get("voice_script"))
    score = 100
    issues = []
    if not lines:
        return 0, ["missing subtitle_lines"]
    for idx, line in enumerate(lines, 1):
        clean = _norm_text(line)
        if not clean:
            score -= 10
            issues.append(f"line {idx} empty")
        if len(clean) > 22:
            score -= 10
            issues.append(f"line {idx} too long ({len(clean)} chars)")
        if clean and clean not in script:
            score -= 8
            issues.append(f"line {idx} not derived from voice_script")
    raw_text = " ".join((unit_by_id.get(uid) or {}).get("raw_text", "") for uid in (kp.get("source_unit_ids") or []))
    if raw_text and _similarity("".join(lines), raw_text) > 0.92:
        score -= 25
        issues.append("subtitle_lines may directly copy raw_text")
    return _clip_score(score), issues


def _subtitles_derived_ratio(kps):
    total = 0
    derived = 0
    for kp in kps or []:
        script = _norm_text(kp.get("voice_script"))
        for line in kp.get("subtitle_lines") or []:
            clean = _norm_text(line)
            if not clean:
                continue
            total += 1
            if clean in script:
                derived += 1
    return round(derived / total, 3) if total else 0.0


def audit(project_dir):
    project_dir = Path(project_dir)
    km_path = project_dir / "knowledge_modules.json"
    rp_path = project_dir / "selected_render_plan.json"
    video_topics_path = project_dir / "video_topics.json"
    render_jobs_path = project_dir / "render_jobs.json"
    selected_render_jobs_path = project_dir / "selected_render_jobs.json"
    cleaning_report_path = project_dir / "cleaning_report.json"
    report_json_path = project_dir / "report.json"
    if not km_path.exists():
        raise FileNotFoundError(km_path)
    plan = _load_json(km_path)
    render_plan = _load_json(rp_path) if rp_path.exists() else {}
    video_topics_doc = _load_json(video_topics_path) if video_topics_path.exists() else {}
    render_jobs_doc = _load_json(render_jobs_path) if render_jobs_path.exists() else {}
    selected_render_jobs_doc = _load_json(selected_render_jobs_path) if selected_render_jobs_path.exists() else {}
    cleaning_report = _load_json(cleaning_report_path) if cleaning_report_path.exists() else {}
    project_report = _load_json(report_json_path) if report_json_path.exists() else {}

    if not isinstance(plan, dict):
        plan = {"knowledge_points": [], "big_hooks": [], "assembly_paths": [], "discarded_content": []}

    hooks = plan.get("big_hooks") or []
    kps = plan.get("knowledge_points") or []
    atoms = plan.get("knowledge_atoms") or []
    units = plan.get("semantic_units") or []
    paths = plan.get("assembly_paths") or []
    discarded = plan.get("discarded_content") or []
    used_chunked = bool(plan.get("used_chunked_planning"))
    chunk_count = int(plan.get("chunk_count") or 0)
    failed_chunk_count = int(plan.get("failed_chunk_count") or 0)
    cached_chunk_count = int(plan.get("cached_chunk_count") or 0)

    unit_by_id = {u.get("unit_id"): u for u in units}
    atom_by_id = {a.get("atom_id"): a for a in atoms}
    kp_by_id = {kp.get("kp_id"): kp for kp in kps}
    plan_level_issues = []
    if len(kps) < 3:
        plan_level_issues.append("too few knowledge_points for flexible selection")

    hook_reports = []
    for hook in hooks:
        score, issues = _hook_score(hook)
        hook_reports.append({"hook_id": hook.get("hook_id"), "hook_title": hook.get("hook_title"), "score": score, "issues": issues})

    kp_reports = []
    direct_copy_kps = []
    fragment_timestamp_issues = []
    type_counts = {}
    for kp in kps:
        kid = kp.get("kp_id")
        type_counts[kp.get("kp_type")] = type_counts.get(kp.get("kp_type"), 0) + 1
        specificity_score, specificity_issues = _kp_specificity_score(kp)
        frag_issues, durations = _fragment_issues(kp)
        kp_duration = calculate_kp_duration(kp)
        duration_method = kp.get("duration_calculation_method")
        fragment_timestamp_issues.extend(f"{kid}: {issue}" for issue in frag_issues)
        voice_score, voice_issues, direct_copy = _voice_script_score(kp, unit_by_id, atom_by_id)
        subtitle_score, subtitle_issues = _subtitle_score(kp, unit_by_id)
        atom_issues = []
        if not kp.get("source_atom_ids"):
            atom_issues.append("missing source_atom_ids")
        for atom_id in kp.get("source_atom_ids") or []:
            atom = atom_by_id.get(atom_id)
            if not atom:
                atom_issues.append(f"unknown atom {atom_id}")
            elif not atom.get("evidence_text"):
                atom_issues.append(f"atom {atom_id} missing evidence_text")
        if direct_copy:
            direct_copy_kps.append(kid)
        kp_reports.append({
            "kp_id": kid,
            "kp_title": kp.get("kp_title"),
            "kp_type": kp.get("kp_type"),
            "specificity_score": specificity_score,
            "voice_script_score": voice_score,
            "subtitle_score": subtitle_score,
            "duration": round(kp_duration, 2),
            "kp_duration_method": duration_method,
            "continuous_span_duration": kp.get("continuous_span_duration"),
            "fragment_sum_duration": kp.get("fragment_sum_duration"),
            "issues": specificity_issues + frag_issues + voice_issues + subtitle_issues + atom_issues,
        })

    hook_quality_score = _score_average([h["score"] for h in hook_reports])
    granularity_base = _score_average([kp["specificity_score"] for kp in kp_reports])
    type_diversity_score = 100 if len([t for t in type_counts if t in VALID_KP_TYPES]) >= 3 else (55 if len(type_counts) > 1 else 25)
    if len(type_counts) == 1 and "concept" in type_counts:
        type_diversity_score = 0
    kp_count_score = 100 if len(kps) >= 3 else (70 if len(kps) == 2 else (35 if len(kps) == 1 else 0))
    knowledge_point_granularity_score = _score_average([granularity_base, type_diversity_score, kp_count_score])

    evidence_scores = []
    for kp in kp_reports:
        atom_issue_count = len([x for x in kp["issues"] if "atom" in x or "source_atom" in x or "timestamp" in x or "duration" in x or "start >=" in x])
        evidence_scores.append(_clip_score(100 - atom_issue_count * 20))
    semantic_evidence_score = _score_average(evidence_scores)

    path_reports = []
    path_types = {p.get("recommended_for") for p in paths}
    path_sets = []
    for path in paths:
        ids = path.get("ordered_kp_ids") or []
        missing_ids = [kid for kid in ids if kid not in kp_by_id]
        path_sets.append(tuple(ids))
        issues = []
        if missing_ids:
            issues.append(f"unknown ordered_kp_ids: {missing_ids}")
        if not ids:
            issues.append("empty ordered_kp_ids")
        if len(set(ids)) < 2:
            issues.append("ordered_kp_ids too short for a real assembly path")
        if not path.get("narrative_structure"):
            issues.append("missing narrative_structure")
        path_reports.append({"path_id": path.get("path_id"), "path_title": path.get("path_title"), "recommended_for": path.get("recommended_for"), "issues": issues, "ordered_kp_ids": ids})
    assembly_score = 100
    missing_required_paths = sorted(REQUIRED_PATH_TYPES - path_types)
    if missing_required_paths:
        assembly_score -= 30
    if len(set(path_sets)) <= 1 and len(path_sets) > 1:
        assembly_score -= 25
    if len(kps) < 3:
        assembly_score -= 20
    assembly_score -= sum(len(p["issues"]) for p in path_reports) * 10
    assembly_value_score = _clip_score(assembly_score)

    discarded_types = {d.get("discard_type") for d in discarded}
    discarded_issues = []
    if not discarded:
        discarded_issues.append("no discarded_content")
    elif not discarded_types & VALID_DISCARD_TYPES:
        discarded_issues.append("discarded_content has no recognized discard_type")

    render_units = render_plan.get("render_units") or []
    render_issues = []
    if not render_units:
        render_issues.append("selected_render_plan missing render_units")
    for unit in render_units:
        if not unit.get("kp_id"):
            render_issues.append(f"{unit.get('unit_id')}: missing kp_id")
        if not unit.get("voice_script"):
            render_issues.append(f"{unit.get('unit_id')}: missing voice_script")
        if not unit.get("subtitle_lines"):
            render_issues.append(f"{unit.get('unit_id')}: missing subtitle_lines")
        if unit.get("tts_text_source") != "voice_script":
            render_issues.append(f"{unit.get('unit_id')}: tts_text_source is not voice_script")
        if unit.get("subtitle_text_source") != "subtitle_lines":
            render_issues.append(f"{unit.get('unit_id')}: subtitle_text_source is not subtitle_lines")

    tts_script_quality_score = _score_average([kp["voice_script_score"] for kp in kp_reports])
    subtitle_readability_score = _score_average([kp["subtitle_score"] for kp in kp_reports])

    suggestions = []
    for kp in kp_reports:
        if "title too generic" in kp["issues"]:
            suggestions.append(f"建议重写 {kp['kp_id']} 的标题，使其表达明确知识价值。")
        if kp["duration"] > 90:
            suggestions.append(f"建议把 {kp['kp_id']} 拆成更细的知识点。")
        if kp["duration"] < 5:
            suggestions.append(f"建议把 {kp['kp_id']} 合并到相邻知识点，或补充上下文。")
        if kp["voice_script_score"] < 75:
            suggestions.append(f"建议重写 {kp['kp_id']} 的 voice_script，去掉口头禅并降低与原文的重复。")
    for hook in hook_reports:
        if hook["score"] < 75:
            suggestions.append(f"建议重写 {hook['hook_id']} 的 hook_title，改成“为什么/如何/从...到...”这类短视频选题。")
    for path in path_reports:
        if path["issues"]:
            suggestions.append(f"建议调整 {path['path_id']} 的 ordered_kp_ids 或叙事结构。")

    blocking_fragment_issues = [
        issue for issue in fragment_timestamp_issues
        if "start >=" in issue or "invalid timestamp" in issue or "duration too short" in issue
    ]
    invalid_fragment_count = len(blocking_fragment_issues)
    path_lengths = [len(p.get("ordered_kp_ids") or []) for p in paths]
    assembly_path_avg_length = round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else 0
    subtitles_derived_ratio = _subtitles_derived_ratio(kps)
    long_kp_count = len([kp for kp in kp_reports if float(kp.get("duration") or 0) > 90])
    long_kp_after_merge_count = long_kp_count
    long_repair = plan.get("long_kp_repair") if isinstance(plan.get("long_kp_repair"), dict) else {}
    split_repair_applied = bool(
        long_repair.get("split_repair_applied")
        or (long_repair.get("debug") or {}).get("split_kps")
    )
    split_repair_needed = long_kp_count > 0
    merge_decisions = plan.get("merge_decisions") or []
    merged_cross_chunk_kp_count = 0
    for item in kps:
        source_chunks = set(item.get("source_chunk_ids") or [])
        if len(source_chunks) > 1:
            merged_cross_chunk_kp_count += 1
    if not merged_cross_chunk_kp_count:
        for decision in merge_decisions:
            merged_from = decision.get("merged_from") or []
            chunks = {str(x).split("_kp_")[0] for x in merged_from if "_kp_" in str(x)}
            if len(chunks) > 1:
                merged_cross_chunk_kp_count += 1
    title_counts = {}
    for item in kps:
        key = _norm_text(item.get("kp_title"))[:24]
        if key:
            title_counts[key] = title_counts.get(key, 0) + 1
    duplicate_kp_count = len(plan.get("discarded_duplicates") or []) + sum(max(0, c - 1) for c in title_counts.values())
    if used_chunked:
        global_merge_quality_score = 100
        if plan.get("global_merge_status") != "success":
            global_merge_quality_score -= 15
        if chunk_count:
            global_merge_quality_score -= int(round((failed_chunk_count / max(1, chunk_count)) * 30))
        global_merge_quality_score -= min(20, duplicate_kp_count * 2)
        global_merge_quality_score -= min(30, long_kp_after_merge_count * 10)
        global_merge_quality_score = _clip_score(global_merge_quality_score)
    else:
        global_merge_quality_score = 100
    overall_score = _score_average([
        hook_quality_score,
        knowledge_point_granularity_score,
        semantic_evidence_score,
        assembly_value_score,
        tts_script_quality_score,
        subtitle_readability_score,
        global_merge_quality_score,
    ])
    if blocking_fragment_issues:
        suggestions.append("建议先修复 fragment 时间戳，确保每个片段 start < end，否则不应进入视频生成。")
        overall_score = min(overall_score, 65)
    if render_issues:
        overall_score = min(overall_score, 69)
    if len(kps) < 3:
        overall_score = min(overall_score, 70)
    can_enter_video_generation = (
        overall_score >= 70
        and not render_issues
        and not blocking_fragment_issues
        and not plan_level_issues
        and semantic_evidence_score >= 60
        and subtitles_derived_ratio >= 1.0
        and assembly_path_avg_length >= 2
    )
    best_hooks = sorted(hook_reports, key=lambda x: x["score"], reverse=True)[:3]
    worst_kps = sorted(kp_reports, key=lambda x: (x["specificity_score"] + x["voice_script_score"] + x["subtitle_score"]) / 3)[:3]
    too_generic_kps = [kp for kp in kp_reports if "title too generic" in kp["issues"]]
    voice_rewrite_kps = [kp for kp in kp_reports if kp["voice_script_score"] < 75]
    weak_paths = [p for p in path_reports if p["issues"]]

    # New product-shape metrics (round-3): cap, video_topics, render_jobs,
    # cleaning_report and plan-reusability.
    video_topics = (
        plan.get("video_topics")
        or video_topics_doc.get("video_topics")
        or []
    )
    video_topics_count = len(video_topics)
    kp_cap_summary = plan.get("kp_cap_summary") or {}
    user_visible_ids = plan.get("user_visible_kp_ids") or kp_cap_summary.get("user_visible_kp_ids") or []
    user_visible_option_count = (
        kp_cap_summary.get("user_visible_count")
        or len(user_visible_ids)
        or video_topics_count
    )
    valid_kp_durations = []
    too_small_kp_count = 0
    too_small_visible_kp_ids = []
    for kp_item in kps:
        dur = float(kp_item.get("duration") or calculate_kp_duration(kp_item) or 0)
        if dur > 0:
            valid_kp_durations.append(dur)
        if dur < 15 and kp_item.get("visibility") != "advanced":
            too_small_kp_count += 1
            kid = kp_item.get("kp_id")
            if kid:
                too_small_visible_kp_ids.append(kid)
    avg_kp_duration = round(sum(valid_kp_durations) / len(valid_kp_durations), 2) if valid_kp_durations else 0.0

    render_jobs_list = render_jobs_doc.get("render_jobs") or []
    render_job_count = len(render_jobs_list)
    render_jobs_kp_count_ok = True
    render_jobs_with_bad_count = []
    for job in render_jobs_list:
        cnt = len(job.get("selected_kp_ids") or [])
        if cnt < 3 or cnt > 6:
            render_jobs_kp_count_ok = False
            render_jobs_with_bad_count.append({"job_id": job.get("job_id"), "kp_count": cnt})

    plan_reusable = bool(
        render_jobs_doc.get("plan_reusable")
        or project_report.get("plan_reusable")
        or (render_job_count > 0 and km_path.exists() and (project_dir / "clips.json").exists())
    )
    planning_input_text = (
        cleaning_report.get("planning_input_text")
        or project_report.get("planning_input_text")
        or "cleaned_segments"
    )
    raw_text_used_for_planning = bool(
        cleaning_report.get("raw_text_used_for_planning")
        or project_report.get("raw_text_used_for_planning", False)
    )
    selected_render_jobs_exists = selected_render_jobs_path.exists() and bool(selected_render_jobs_doc.get("render_jobs"))

    # Sanity: 普通用户可见选项必须 ≤10（10/18/25 三档 cap，但暴露给用户选择的应是 video_topics）。
    fragments_visible_to_user = False  # planner 不再暴露 fragments，但 audit 仍校验
    if user_visible_option_count > 25:
        fragments_visible_to_user = True

    new_metrics = {
        "user_visible_option_count": int(user_visible_option_count),
        "video_topics_count": int(video_topics_count),
        "render_job_count": int(render_job_count),
        "avg_kp_duration": avg_kp_duration,
        "too_small_kp_count": int(too_small_kp_count),
        "too_small_visible_kp_ids": too_small_visible_kp_ids,
        "render_jobs_kp_count_ok": render_jobs_kp_count_ok,
        "render_jobs_with_bad_count": render_jobs_with_bad_count,
        "plan_reusable": bool(plan_reusable),
        "planning_input_text": planning_input_text,
        "raw_text_used_for_planning": raw_text_used_for_planning,
        "selected_render_jobs_exists": bool(selected_render_jobs_exists),
        "fragments_visible_to_user": bool(fragments_visible_to_user),
        "cleaning_report_present": bool(cleaning_report),
        "kp_cap_summary": kp_cap_summary,
    }

    # Suggestions tied to the new metrics
    if video_topics_count == 0:
        suggestions.append("缺少 video_topics，请重新生成内容计划。")
    elif video_topics_count > 10:
        suggestions.append(f"video_topics 数量 {video_topics_count} > 10，建议合并相近方案。")
    if user_visible_option_count > 25:
        suggestions.append(f"用户可见选项 {user_visible_option_count} > 25，超出 cap，请检查 kp_cap。")
    if too_small_kp_count > 0:
        suggestions.append(f"{too_small_kp_count} 个用户可见 kp 时长 < 15s，请合并或标记 advanced。")
    if not render_jobs_kp_count_ok:
        suggestions.append("某些 render_job 的 selected_kp_ids 数量不在 3–6 区间。")
    if not plan_reusable:
        suggestions.append("plan_reusable=false，可能缺少 render_jobs.json 或内容计划文件不完整。")
    if planning_input_text != "cleaned_segments":
        suggestions.append(f"planning_input_text={planning_input_text}，应为 cleaned_segments。")
    if raw_text_used_for_planning:
        suggestions.append("raw_text_used_for_planning=true，意味着规划阶段读到了原文，请检查清洗链路。")

    report = {
        "project_dir": str(project_dir),
        **new_metrics,
        "scores": {
            "overall_score": overall_score,
            "hook_quality_score": hook_quality_score,
            "knowledge_point_granularity_score": knowledge_point_granularity_score,
            "semantic_evidence_score": semantic_evidence_score,
            "assembly_value_score": assembly_value_score,
            "tts_script_quality_score": tts_script_quality_score,
            "subtitle_readability_score": subtitle_readability_score,
            "global_merge_quality_score": global_merge_quality_score,
        },
        "best_big_hooks": best_hooks,
        "worst_knowledge_points": worst_kps,
        "too_generic_knowledge_points": too_generic_kps,
        "voice_scripts_need_rewrite": voice_rewrite_kps,
        "weak_assembly_paths": weak_paths,
        "direct_copy_voice_script_kp_ids": direct_copy_kps,
        "plan_level_issues": plan_level_issues,
        "type_counts": type_counts,
        "knowledge_point_count": len(kps),
        "invalid_fragment_count": invalid_fragment_count,
        "assembly_path_avg_length": assembly_path_avg_length,
        "subtitles_derived_from_voice_script_ratio": subtitles_derived_ratio,
        "kp_duration_method": {
            kp["kp_id"]: kp.get("kp_duration_method") for kp in kp_reports
        },
        "long_kp_count": long_kp_count,
        "used_chunked_planning": used_chunked,
        "chunk_count": chunk_count,
        "failed_chunk_count": failed_chunk_count,
        "cached_chunk_count": cached_chunk_count,
        "merged_cross_chunk_kp_count": merged_cross_chunk_kp_count,
        "duplicate_kp_count": duplicate_kp_count,
        "long_kp_after_merge_count": long_kp_after_merge_count,
        "split_repair_needed": split_repair_needed,
        "split_repair_applied": split_repair_applied,
        "discarded_content_types": sorted(t for t in discarded_types if t),
        "discarded_content_issues": discarded_issues,
        "render_plan_issues": render_issues,
        "fragment_timestamp_issues": fragment_timestamp_issues,
        "blocking_fragment_issues": blocking_fragment_issues,
        "selected_render_plan_uses_voice_script": not any("tts_text_source" in issue for issue in render_issues),
        "selected_render_plan_uses_subtitle_lines": not any("subtitle_text_source" in issue for issue in render_issues),
        "can_enter_video_generation": can_enter_video_generation,
        "suggestions": suggestions,
    }
    return report


def write_reports(project_dir, report):
    project_dir = Path(project_dir)
    json_path = project_dir / "knowledge_plan_quality_report.json"
    md_path = project_dir / "knowledge_plan_quality_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    scores = report["scores"]
    lines = [
        "# Knowledge Plan Quality Report",
        "",
        f"- project_dir: `{report['project_dir']}`",
        f"- overall_score: **{scores['overall_score']}**",
        f"- can_enter_video_generation: **{report['can_enter_video_generation']}**",
        "",
        "## Scores",
    ]
    for key, value in scores.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Key Metrics",
        f"- knowledge_point_count: {report['knowledge_point_count']}",
        f"- user_visible_option_count: {report.get('user_visible_option_count', 0)}",
        f"- video_topics_count: {report.get('video_topics_count', 0)}",
        f"- render_job_count: {report.get('render_job_count', 0)}",
        f"- avg_kp_duration: {report.get('avg_kp_duration', 0)}",
        f"- too_small_kp_count: {report.get('too_small_kp_count', 0)}",
        f"- render_jobs_kp_count_ok: {report.get('render_jobs_kp_count_ok', True)}",
        f"- plan_reusable: {report.get('plan_reusable', False)}",
        f"- planning_input_text: {report.get('planning_input_text', '?')}",
        f"- raw_text_used_for_planning: {report.get('raw_text_used_for_planning', False)}",
        f"- selected_render_jobs_exists: {report.get('selected_render_jobs_exists', False)}",
        f"- cleaning_report_present: {report.get('cleaning_report_present', False)}",
        f"- invalid_fragment_count: {report['invalid_fragment_count']}",
        f"- assembly_path_avg_length: {report['assembly_path_avg_length']}",
        f"- subtitles_derived_from_voice_script_ratio: {report['subtitles_derived_from_voice_script_ratio']}",
        f"- long_kp_count: {report['long_kp_count']}",
        f"- used_chunked_planning: {report['used_chunked_planning']}",
        f"- chunk_count: {report['chunk_count']}",
        f"- failed_chunk_count: {report['failed_chunk_count']}",
        f"- cached_chunk_count: {report['cached_chunk_count']}",
        f"- merged_cross_chunk_kp_count: {report['merged_cross_chunk_kp_count']}",
        f"- duplicate_kp_count: {report['duplicate_kp_count']}",
        f"- long_kp_after_merge_count: {report['long_kp_after_merge_count']}",
        f"- split_repair_needed: {report['split_repair_needed']}",
        f"- split_repair_applied: {report['split_repair_applied']}",
    ]
    lines += ["", "## Plan Level Issues"]
    if report["plan_level_issues"]:
        lines.extend(f"- {issue}" for issue in report["plan_level_issues"])
    else:
        lines.append("- None")
    lines += ["", "## Best 3 Big Hooks"]
    for item in report["best_big_hooks"]:
        lines.append(f"- {item.get('hook_id')}: {item.get('hook_title')} ({item.get('score')})")
    lines += ["", "## Worst 3 Knowledge Points"]
    for item in report["worst_knowledge_points"]:
        lines.append(f"- {item.get('kp_id')}: {item.get('kp_title')} | type={item.get('kp_type')} | issues={'; '.join(item.get('issues') or [])}")
    lines += ["", "## Too Generic Knowledge Points"]
    for item in report["too_generic_knowledge_points"] or []:
        lines.append(f"- {item.get('kp_id')}: {item.get('kp_title')}")
    if not report["too_generic_knowledge_points"]:
        lines.append("- None")
    lines += ["", "## Voice Scripts Need Rewrite"]
    for item in report["voice_scripts_need_rewrite"] or []:
        lines.append(f"- {item.get('kp_id')}: score={item.get('voice_script_score')} issues={'; '.join(item.get('issues') or [])}")
    if not report["voice_scripts_need_rewrite"]:
        lines.append("- None")
    lines += ["", "## Weak Assembly Paths"]
    for item in report["weak_assembly_paths"] or []:
        lines.append(f"- {item.get('path_id')}: {item.get('path_title')} issues={'; '.join(item.get('issues') or [])}")
    if not report["weak_assembly_paths"]:
        lines.append("- None")
    lines += ["", "## Render Plan Checks"]
    if report["render_plan_issues"]:
        lines.extend(f"- {issue}" for issue in report["render_plan_issues"])
    else:
        lines.append("- selected_render_plan render_units are valid.")
    lines += ["", "## Fragment Timestamp Issues"]
    if report["fragment_timestamp_issues"]:
        lines.extend(f"- {issue}" for issue in report["fragment_timestamp_issues"])
    else:
        lines.append("- None")
    lines += ["", "## Suggestions"]
    if report["suggestions"]:
        lines.extend(f"- {s}" for s in report["suggestions"])
    else:
        lines.append("- No major fixes suggested.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", nargs="?", help="outputs/<project_id> or project_id. Defaults to latest output with knowledge_modules.json.")
    args = parser.parse_args()
    project_dir = _project_dir_from_arg(args.project_id)
    report = audit(project_dir)
    md_path, json_path = write_reports(project_dir, report)
    print(json.dumps({
        "project_dir": str(project_dir),
        "markdown_report": str(md_path),
        "json_report": str(json_path),
        **report["scores"],
        "best_big_hooks": report["best_big_hooks"],
        "worst_knowledge_points": report["worst_knowledge_points"],
        "direct_copy_voice_script_kp_ids": report["direct_copy_voice_script_kp_ids"],
        "knowledge_point_count": report["knowledge_point_count"],
        "invalid_fragment_count": report["invalid_fragment_count"],
        "assembly_path_avg_length": report["assembly_path_avg_length"],
        "subtitles_derived_from_voice_script_ratio": report["subtitles_derived_from_voice_script_ratio"],
        "long_kp_count": report["long_kp_count"],
        "used_chunked_planning": report["used_chunked_planning"],
        "chunk_count": report["chunk_count"],
        "failed_chunk_count": report["failed_chunk_count"],
        "cached_chunk_count": report["cached_chunk_count"],
        "merged_cross_chunk_kp_count": report["merged_cross_chunk_kp_count"],
        "duplicate_kp_count": report["duplicate_kp_count"],
        "long_kp_after_merge_count": report["long_kp_after_merge_count"],
        "split_repair_needed": report["split_repair_needed"],
        "split_repair_applied": report["split_repair_applied"],
        "selected_render_plan_uses_voice_script": report["selected_render_plan_uses_voice_script"],
        "selected_render_plan_uses_subtitle_lines": report["selected_render_plan_uses_subtitle_lines"],
        "can_enter_video_generation": report["can_enter_video_generation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
