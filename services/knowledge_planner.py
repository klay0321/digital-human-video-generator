"""Knowledge-point segmentation + module planner.

Pipeline (knowledge-module mode):
  1. clean_transcript_segments_with_llm  → cleaned_segments + cleaned_text
  2. plan_knowledge_modules_with_llm     → knowledge_modules
  3. build_timeline_clips_from_modules   → clips.json compatible shape
  4. build_selected_render_plan          → user-selected subset for rendering
  5. validate_knowledge_modules          → safety invariants

If the LLM fails at step 1 or step 2, deterministic fallbacks kick in. The
fallbacks NEVER emit prompt text into user-visible fields.
"""
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path


LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120") or 120)
PLANNING_MAX_TOTAL_SECONDS = int(os.getenv("PLANNING_MAX_TOTAL_SECONDS", "900") or 900)

# Common Chinese fillers used by the deterministic fallback cleaner.
_FILLERS = (
    "嗯", "啊", "呃", "哎", "唉", "诶",
    "这个", "那个", "然后呢", "就是说", "就是", "那么",
    "对吧", "是吧", "你知道吧", "你懂的",
)

# Compile a single regex for fast cleaning.
_FILLER_RE = re.compile("|".join(re.escape(f) for f in _FILLERS))

# Default lightweight topic keywords for deterministic clustering when the
# LLM isn't available. Map: lowercased Chinese/English keyword -> topic_key.
_TOPIC_HINTS = {
    "mcp": "mcp",
    "agent": "agent",
    "工作流": "agent_workflow",
    "workflow": "agent_workflow",
    "工具": "tools",
    "tool": "tools",
    "案例": "demo",
    "演示": "demo",
    "demo": "demo",
    "总结": "summary",
    "summary": "summary",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-leak helpers (callers MUST import from pipeline to share the canonical
# detector; we inject one to avoid circular imports during testing).
# ─────────────────────────────────────────────────────────────────────────────
def _default_looks_like_prompt_leak(text):
    if not text or not isinstance(text, str):
        return False
    return (
        "你是一个录屏讲解短视频策划助手" in text
        or "你是一个短视频内容策划专家" in text
        or "{{FULL_TEXT}}" in text
        or "{{SEGMENTS_JSON}}" in text
        or "{{CLEANED_SEGMENTS_JSON}}" in text
        or "{{VOICE_STYLE}}" in text
        or "不要凭空编造" in text
        or "只返回 JSON，不要返回 markdown" in text
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic cleaner used when LLM step 1 fails
# ─────────────────────────────────────────────────────────────────────────────
def _basic_clean(text):
    if not text:
        return ""
    cleaned = _FILLER_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _guess_topic_tags(text):
    tags = []
    t = (text or "").lower()
    for kw, _ in _TOPIC_HINTS.items():
        if kw in t and kw not in tags:
            tags.append(kw)
    return tags[:6]


def build_fallback_cleaned_segments(transcript_segments):
    """Deterministic cleaning fallback. Keeps every segment, strips fillers,
    guesses topic_tags from a small keyword table. Never includes prompt text.

    If all incoming segments have start==end==0 (manual / uploaded-text mode),
    synthesize timestamps at ~4 chars/sec so downstream fragment cuts are
    non-zero. The synthesized timestamps are advisory; the renderer will use
    them to cut the user-provided video.
    """
    segs = list(transcript_segments or [])
    all_zero_time = bool(segs) and all(
        (float(s.get("end", 0) or 0) - float(s.get("start", 0) or 0)) <= 0
        for s in segs
    )
    cursor = 0.0
    out = []
    for i, seg in enumerate(segs):
        raw = (seg.get("text") or "").strip()
        cleaned = _basic_clean(raw)
        is_filler = (not cleaned) or len(cleaned) < 6
        tags = _guess_topic_tags(cleaned)
        if all_zero_time:
            est_dur = max(1.0, len(cleaned) / 4.0)
            seg_start = cursor
            seg_end = cursor + est_dur
            cursor = seg_end
        else:
            seg_start = float(seg.get("start", 0) or 0)
            seg_end = float(seg.get("end", 0) or 0)
        out.append({
            "segment_id": f"seg_{i + 1:04d}",
            "start": seg_start,
            "end": seg_end,
            "raw_text": raw,
            "clean_text": cleaned,
            "is_filler": is_filler,
            "filler_reason": "段落过短或仅含口头禅" if is_filler else "",
            "topic_tags": tags,
            "knowledge_point": "",
            "importance": 0.3 if is_filler else (0.5 + min(0.4, len(cleaned) / 200.0)),
            "estimated_timestamps": all_zero_time,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic module planner used when LLM step 2 fails
# ─────────────────────────────────────────────────────────────────────────────
def _segments_topic_key(seg):
    """Pick a canonical topic_key for a cleaned segment based on its tags."""
    for tag in (seg.get("topic_tags") or []):
        key = (tag or "").lower()
        if key in _TOPIC_HINTS:
            return _TOPIC_HINTS[key]
        if key:
            return key
    return None


def _group_consecutive_by_topic(cleaned_segments):
    """Return [(topic_key, [segments...])] grouping consecutive same-topic
    segments. is_filler segments break runs and are dropped.
    """
    groups = []
    cur_topic = None
    cur_run = []
    for seg in cleaned_segments:
        if seg.get("is_filler"):
            if cur_run:
                groups.append((cur_topic, cur_run))
                cur_run = []
                cur_topic = None
            continue
        topic = _segments_topic_key(seg)
        if cur_topic is None:
            cur_topic = topic
            cur_run = [seg]
        elif topic == cur_topic:
            cur_run.append(seg)
        else:
            groups.append((cur_topic, cur_run))
            cur_topic = topic
            cur_run = [seg]
    if cur_run:
        groups.append((cur_topic, cur_run))
    return groups


def _fragment_from_run(run, fragment_id, reason=""):
    if not run:
        return None
    start = float(run[0].get("start", 0) or 0)
    end = float(run[-1].get("end", 0) or 0)
    seg_ids = [s.get("segment_id") for s in run if s.get("segment_id")]
    src_text = "".join((s.get("raw_text") or "") for s in run).strip()
    cln_text = "".join((s.get("clean_text") or "") for s in run).strip()
    est = any(bool(s.get("estimated_timestamps")) for s in run)
    return {
        "fragment_id": fragment_id,
        "start": start,
        "end": end,
        "segment_ids": seg_ids,
        "source_text": src_text,
        "cleaned_text": cln_text or src_text,
        "reason": reason or "deterministic group",
        "estimated_timestamps": est,
    }


def build_fallback_knowledge_modules(cleaned_segments):
    """Cluster cleaned_segments into modules by:
      a) If any segment has topic_tags, group same-topic runs into modules
         (multiple non-contiguous fragments per topic are allowed).
      b) Else, fall back to a single time-based module per ~90s bucket.
    Each fragment is one **contiguous** run; no big start/end spanning.
    """
    if not cleaned_segments:
        return []

    has_tags = any((s.get("topic_tags") or []) for s in cleaned_segments)

    if has_tags:
        groups = _group_consecutive_by_topic(cleaned_segments)
        # Aggregate runs by topic into modules with potentially many fragments.
        per_topic = {}
        for topic, run in groups:
            key = topic or "misc"
            per_topic.setdefault(key, []).append(run)

        modules = []
        topic_seq = 1
        global_frag_counter = [0]
        def _next_frag_id():
            global_frag_counter[0] += 1
            return f"frag_{global_frag_counter[0]:04d}"
        for topic, runs in per_topic.items():
            module_id = f"module_{topic}_{topic_seq:03d}"
            topic_seq += 1
            fragments = []
            merged_src_parts = []
            merged_cln_parts = []
            for j, run in enumerate(runs):
                frag = _fragment_from_run(run, _next_frag_id(),
                                          reason=f"同主题片段 #{j + 1}")
                if frag is None:
                    continue
                fragments.append(frag)
                merged_src_parts.append(frag["source_text"])
                merged_cln_parts.append(frag["cleaned_text"])
            if not fragments:
                continue
            merged_src = " ".join(p for p in merged_src_parts if p).strip()
            merged_cln = " ".join(p for p in merged_cln_parts if p).strip()
            total_dur = sum(f["end"] - f["start"] for f in fragments)
            modules.append({
                "module_id": module_id,
                "module_title": topic.upper() if topic else "其他",
                "topic_key": topic or "misc",
                "module_type": "knowledge",
                "summary": (merged_cln or merged_src)[:80],
                "hook": (merged_cln or merged_src)[:30],
                "knowledge_points": [],
                "fragments": fragments,
                "merged_source_text": merged_src,
                "merged_cleaned_text": merged_cln,
                "voice_script": merged_cln or merged_src,
                "suggested_title": f"{topic.upper() if topic else '其他'} 合集",
                "suggested_duration": float(round(total_dur, 1)),
                "render_mode": "concat_fragments",
            })
        if modules:
            return modules

    # No topic tags anywhere — fall back to time buckets ~90s.
    modules = []
    bucket = []
    bucket_start = float(cleaned_segments[0].get("start", 0) or 0)
    for i, seg in enumerate(cleaned_segments):
        if seg.get("is_filler"):
            continue
        bucket.append(seg)
        cur_end = float(seg.get("end", 0) or 0)
        is_last = (i == len(cleaned_segments) - 1)
        if cur_end - bucket_start >= 90 or is_last:
            if bucket:
                cn = len(modules) + 1
                frag = _fragment_from_run(bucket, f"frag_{cn:04d}",
                                          reason="时间桶兜底")
                if frag:
                    cln = frag["cleaned_text"]
                    modules.append({
                        "module_id": f"module_misc_{cn:03d}",
                        "module_title": f"片段 {cn}",
                        "topic_key": "misc",
                        "module_type": "knowledge",
                        "summary": cln[:80],
                        "hook": cln[:30],
                        "knowledge_points": [],
                        "fragments": [frag],
                        "merged_source_text": frag["source_text"],
                        "merged_cleaned_text": cln,
                        "voice_script": cln,
                        "suggested_title": f"片段 {cn}",
                        "suggested_duration": round(frag["end"] - frag["start"], 1),
                        "render_mode": "concat_fragments",
                    })
            bucket = []
            if i + 1 < len(cleaned_segments):
                bucket_start = float(cleaned_segments[i + 1].get("start", cur_end) or cur_end)
    return modules


# ─────────────────────────────────────────────────────────────────────────────
# LLM-driven step 1: clean + tag segments
# ─────────────────────────────────────────────────────────────────────────────
def clean_transcript_segments_with_llm(transcript_segments, full_text, llm_call_fn,
                                       prompt_template):
    """Call the LLM to clean and tag segments.

    Returns a dict:
      {
        "result":         dict (cleaned_text + cleaned_segments) or None,
        "llm_prompt":     str (the substituted prompt, for debug),
        "raw_response":   str,
        "error":          str | None,
      }
    """
    segments_json = json.dumps(transcript_segments or [], ensure_ascii=False, indent=2)
    llm_prompt = (
        (prompt_template or "")
        .replace("{{FULL_TEXT}}", full_text or "")
        .replace("{{SEGMENTS_JSON}}", segments_json)
    )
    try:
        call = llm_call_fn(llm_prompt, stage="llm_clean")
    except TypeError:
        call = llm_call_fn(llm_prompt)
    parsed = call.get("result") if isinstance(call, dict) else None
    raw = (call or {}).get("raw_response") or ""
    err = (call or {}).get("error")
    return {
        "result": parsed if isinstance(parsed, dict) else None,
        "llm_prompt": llm_prompt,
        "raw_response": raw,
        "error": err,
        "diagnostic": (call or {}).get("diagnostic") or call,
        "model": (call or {}).get("model"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM-driven step 2: cluster into knowledge modules
# ─────────────────────────────────────────────────────────────────────────────
def plan_knowledge_modules_with_llm(cleaned_segments, llm_call_fn, prompt_template,
                                    voice_style="讲解风格"):
    segments_json = json.dumps(cleaned_segments or [], ensure_ascii=False, indent=2)
    llm_prompt = (
        (prompt_template or "")
        .replace("{{CLEANED_SEGMENTS_JSON}}", segments_json)
        .replace("{{VOICE_STYLE}}", voice_style or "讲解风格")
    )
    try:
        call = llm_call_fn(llm_prompt, stage="llm_plan")
    except TypeError:
        call = llm_call_fn(llm_prompt)
    parsed = call.get("result") if isinstance(call, dict) else None
    raw = (call or {}).get("raw_response") or ""
    err = (call or {}).get("error")
    return {
        "result": parsed if isinstance(parsed, dict) else None,
        "llm_prompt": llm_prompt,
        "raw_response": raw,
        "error": err,
        "diagnostic": (call or {}).get("diagnostic") or call,
        "model": (call or {}).get("model"),
    }


def _call_llm_json_for_planning(llm_client, prompt, stage):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        try:
            future = executor.submit(llm_client, prompt, stage=stage)
        except TypeError:
            future = executor.submit(llm_client, prompt)
        try:
            call = future.result(timeout=LLM_REQUEST_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            future.cancel()
            return {
                "result": None,
                "raw_response": "",
                "error": f"LLM request timeout after {LLM_REQUEST_TIMEOUT_SECONDS}s",
                "diagnostic": {
                    "success": False,
                    "stage": stage,
                    "error_type": "request_timeout",
                    "error_message": f"LLM request timeout after {LLM_REQUEST_TIMEOUT_SECONDS}s",
                },
                "model": None,
            }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    parsed = call.get("result") if isinstance(call, dict) else None
    return {
        "result": parsed if isinstance(parsed, dict) else None,
        "raw_response": (call or {}).get("raw_response") or "",
        "error": (call or {}).get("error"),
        "diagnostic": (call or {}).get("diagnostic") or call,
        "model": (call or {}).get("model"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Adapters: modules <-> clips, render plan
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_TYPES = {
    "problem", "concept", "principle", "cause", "operation", "implementation",
    "workflow", "tool_usage", "case", "business_value", "comparison",
    "pitfall", "decision", "summary", "transition",
}
HOOK_TYPES = {"boss_report", "tutorial", "operation_demo", "product_intro", "short_video"}


def split_subtitle_lines(text, min_chars=18, max_chars=22):
    text = re.sub(r"\s+", "", text or "")
    if not text:
        return []
    chunks = []
    buf = ""
    for part in re.split(r"([。！？；，、])", text):
        if not part:
            continue
        if len(buf) + len(part) <= max_chars:
            buf += part
        else:
            if buf:
                chunks.append(buf)
            buf = part
        while len(buf) > max_chars:
            chunks.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf:
        chunks.append(buf)
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < max(8, min_chars // 2) and len(merged[-1]) + len(chunk) <= max_chars:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    return merged


def generate_subtitle_lines_from_voice_script(voice_script: str, max_chars_per_line: int = 20):
    """Create subtitle lines only from voice_script.

    This deliberately ignores raw/source text so TTS and subtitles stay aligned.
    """
    return split_subtitle_lines(voice_script or "", min_chars=18, max_chars=max_chars_per_line)


def _clean_tts_text(text):
    return re.sub(r"\s+", " ", _basic_clean(text or "")).strip()


def _duration(start, end):
    try:
        return max(0.0, float(end or 0) - float(start or 0))
    except Exception:
        return 0.0


def _duration_info_for_kp(kp):
    fragments = kp.get("fragments") or []
    valid_fragment_durations = []
    starts = []
    ends = []
    invalid_fragment_count = 0
    for frag in fragments:
        try:
            start = float(frag.get("start"))
            end = float(frag.get("end"))
        except Exception:
            invalid_fragment_count += 1
            continue
        if start < end:
            starts.append(start)
            ends.append(end)
            valid_fragment_durations.append(end - start)
        else:
            invalid_fragment_count += 1

    fragment_sum = sum(valid_fragment_durations)
    continuous_span = (max(ends) - min(starts)) if starts and ends else 0.0

    if valid_fragment_durations:
        method = "fragments_sum"
        duration = fragment_sum
    else:
        span_duration = _duration(kp.get("start"), kp.get("end"))
        if span_duration > 0:
            method = "continuous_span"
            duration = span_duration
            continuous_span = span_duration
        else:
            method = "estimated"
            duration = _estimate_duration_from_text(
                kp.get("voice_script") or kp.get("kp_summary") or "",
                min_seconds=3.0,
            )
            continuous_span = duration
            fragment_sum = duration

    return {
        "duration": float(duration),
        "duration_calculation_method": method,
        "continuous_span_duration": float(continuous_span),
        "fragment_sum_duration": float(fragment_sum),
        "invalid_fragment_count": invalid_fragment_count,
    }


def calculate_kp_duration(kp):
    """Return the render duration of one knowledge_point.

    Multiple fragments are additive. We deliberately do not measure from the
    first fragment start to the last fragment end, because that would count the
    unrelated gap between non-contiguous evidence fragments.
    """
    if not isinstance(kp, dict):
        return 0.0
    info = _duration_info_for_kp(kp)
    kp["duration_calculation_method"] = info["duration_calculation_method"]
    kp["continuous_span_duration"] = round(info["continuous_span_duration"], 3)
    kp["fragment_sum_duration"] = round(info["fragment_sum_duration"], 3)
    kp["duration"] = round(info["duration"], 3)
    if (
        info["continuous_span_duration"] > 90
        and info["fragment_sum_duration"] <= 90
        and info["duration_calculation_method"] == "fragments_sum"
    ):
        kp["duration_warning_resolved_by_fragment_sum"] = True
    return info["duration"]


def _estimate_kp_duration(kp):
    return calculate_kp_duration(kp)


def _estimate_duration_from_text(text, min_seconds=3.0):
    normalized = re.sub(r"\s+", "", text or "")
    return max(float(min_seconds), min(30.0, len(normalized) / 5.0))


def _valid_interval(start, end, min_duration=1.5):
    try:
        return float(end) - float(start) >= min_duration
    except Exception:
        return False


def _source_unit_interval(source_unit_ids, units_by_id):
    intervals = []
    for uid in source_unit_ids or []:
        unit = units_by_id.get(uid) or {}
        if _valid_interval(unit.get("start"), unit.get("end")):
            intervals.append((float(unit.get("start")), float(unit.get("end"))))
    if not intervals:
        return None
    return min(s for s, _ in intervals), max(e for _, e in intervals)


def _segment_interval(segment_ids, source_segments):
    if not source_segments:
        return None
    by_id = {s.get("segment_id"): s for s in source_segments if isinstance(s, dict)}
    intervals = []
    for sid in segment_ids or []:
        seg = by_id.get(sid)
        if seg and _valid_interval(seg.get("start"), seg.get("end")):
            intervals.append((float(seg.get("start")), float(seg.get("end"))))
    if not intervals:
        valid = [
            (float(s.get("start")), float(s.get("end")))
            for s in source_segments if _valid_interval(s.get("start"), s.get("end"))
        ]
        intervals = valid[:1]
    if not intervals:
        return None
    return min(s for s, _ in intervals), max(e for _, e in intervals)


def _kp_score_defaults(kp_type, confidence=0.62, low_information=False):
    base_clip = 0.55 if low_information else 0.72
    return {
        "importance": 3 if low_information else (4 if kp_type not in {"transition", "summary"} else 3),
        "clarity": round(max(0.45, min(0.9, confidence)), 2),
        "clip_value": base_clip,
        "standalone": 0.72 if kp_type in {"concept", "problem", "summary"} else 0.62,
        "business_value": 0.66 if kp_type == "business_value" else 0.42,
        "operation_value": 0.74 if kp_type in {"operation", "implementation", "workflow", "tool_usage"} else 0.32,
        "novelty": 0.46 if low_information else 0.56,
    }


def repair_plan_if_too_few_knowledge_points(plan):
    """Split over-merged semantic plans into selectable knowledge_points.

    The repair is evidence-first: it prefers existing knowledge_atoms and only
    falls back to semantic_units when atoms are unavailable.
    """
    if not isinstance(plan, dict):
        return plan
    points = plan.get("knowledge_points") or []
    atoms = plan.get("knowledge_atoms") or []
    units = plan.get("semantic_units") or []
    summary = plan.get("source_summary") or {}
    should_repair = (
        len(points) < 3
        and (
            len(atoms) >= 3
            or len(units) >= 3
            or summary.get("content_type") not in {"very_short", "extremely_short"}
        )
    )
    if not should_repair:
        return plan

    low_info = "缺少" in (summary.get("overall_quality_notes") or "") or "不可辨识" in (summary.get("speaker_intent") or "")
    new_points = []
    source_items = atoms if atoms else [
        {
            "atom_id": f"repair_atom_{idx + 1:03d}",
            "source_unit_ids": [u.get("unit_id")] if u.get("unit_id") else [],
            "atom_title": (u.get("clean_text") or u.get("raw_text") or f"可识别内容 {idx + 1}")[:28],
            "atom_type": _type_for_role(u.get("unit_role"), u.get("clean_text") or u.get("raw_text")),
            "atom_text": u.get("clean_text") or u.get("raw_text") or "",
            "evidence_text": u.get("raw_text") or u.get("clean_text") or "",
            "start": u.get("start", 0),
            "end": u.get("end", 0),
            "confidence": 0.58,
        }
        for idx, u in enumerate(units)
        if (u.get("clean_text") or u.get("raw_text"))
    ]

    if len(source_items) < 3 and points:
        old = points[0]
        base_atom_ids = old.get("source_atom_ids") or [a.get("atom_id") for a in atoms if a.get("atom_id")]
        base_unit_ids = old.get("source_unit_ids") or []
        fallback_specs = [
            ("已明确讲到的内容", "summary", old.get("kp_summary") or old.get("voice_script") or ""),
            ("只被提到但未展开的内容", "pitfall", "这些信息目前只停留在关键词层面，还不能当作完整教程或操作结论。"),
            ("后续成片前需要补充的信息", "decision", "如果要进入正式视频生成，需要补充更完整的步骤、上下文和可验证结果。"),
        ]
        source_items = [
            {
                "atom_id": (base_atom_ids or [f"repair_atom_{idx + 1:03d}"])[0],
                "source_unit_ids": base_unit_ids,
                "atom_title": title,
                "atom_type": atom_type,
                "atom_text": text,
                "evidence_text": text,
                "start": 0,
                "end": 0,
                "confidence": 0.5,
            }
            for idx, (title, atom_type, text) in enumerate(fallback_specs)
        ]

    for idx, atom in enumerate(source_items[:8]):
        kp_id = f"kp_{idx + 1:03d}"
        kp_type = atom.get("atom_type") if atom.get("atom_type") in KNOWLEDGE_TYPES else "concept"
        atom_text = _clean_tts_text(atom.get("atom_text") or atom.get("evidence_text") or "")
        voice_script = atom_text
        if voice_script and voice_script[-1] not in "。！？":
            voice_script += "。"
        title = atom.get("atom_title") or f"可选择知识点 {idx + 1}"
        frag = {
            "fragment_id": f"frag_{idx + 1:03d}",
            "start": atom.get("start", 0),
            "end": atom.get("end", 0),
            "clean_text": atom_text,
            "reason": "由 knowledge_atom 拆分得到，保留原文证据边界。",
            "timestamp_source": atom.get("timestamp_source"),
        }
        new_points.append({
            "kp_id": kp_id,
            "kp_title": title,
            "kp_type": kp_type,
            "kp_summary": atom_text[:90],
            "source_atom_ids": [atom.get("atom_id")] if atom.get("atom_id") else [],
            "source_unit_ids": [x for x in (atom.get("source_unit_ids") or []) if x],
            "fragments": [frag],
            "voice_script": voice_script,
            "subtitle_lines": generate_subtitle_lines_from_voice_script(voice_script),
            "scores": _kp_score_defaults(kp_type, float(atom.get("confidence") or 0.58), low_information=low_info),
            "dependencies": {
                "requires": [f"kp_{idx:03d}"] if idx > 0 and kp_type in {"operation", "implementation", "workflow", "tool_usage", "case"} else [],
                "recommended_before": [],
                "recommended_after": [f"kp_{idx + 2:03d}"] if idx + 1 < len(source_items[:8]) else [],
                "supports": [],
                "example_of": [],
                "contrasts_with": [],
                "follows": [f"kp_{idx:03d}"] if idx > 0 else [],
                "duplicates": [],
                "can_stand_alone": kp_type in {"problem", "concept", "summary", "business_value", "pitfall", "decision"} or idx == 0,
            },
            "selection_reason": "这是从独立知识原子拆出的可勾选知识点。" if not low_info else "原文信息较少，该知识点只表达可验证的关键词或信息缺口。",
            "notices": ["低信息密度素材，不宜扩展为原文没有讲清的教程。"] if low_info else [],
        })

    if len(new_points) >= 3:
        plan["knowledge_points"] = new_points
        plan["repair_notes"] = (plan.get("repair_notes") or []) + [
            f"knowledge_points_too_few_repaired_from_{len(points)}_to_{len(new_points)}"
        ]
    return plan


def repair_invalid_fragment_timestamps(plan, source_segments=None):
    if not isinstance(plan, dict):
        return plan
    units_by_id = {u.get("unit_id"): u for u in (plan.get("semantic_units") or [])}
    cursor = 0.0
    repaired = 0
    unrepairable = 0
    for kp_item in plan.get("knowledge_points") or []:
        kp_units = kp_item.get("source_unit_ids") or []
        for frag in kp_item.get("fragments") or []:
            start = frag.get("start")
            end = frag.get("end")
            if _valid_interval(start, end, min_duration=1.5):
                cursor = max(cursor, float(end))
                continue
            interval = _source_unit_interval(kp_units, units_by_id)
            if not interval:
                interval = _segment_interval(frag.get("segment_ids") or [], source_segments or [])
            if interval and _valid_interval(interval[0], interval[1], min_duration=1.5):
                start, end = interval
                source = "source_unit_or_segment"
            else:
                text = (
                    frag.get("clean_text") or frag.get("cleaned_text")
                    or kp_item.get("voice_script") or kp_item.get("kp_summary") or ""
                )
                dur = _estimate_duration_from_text(text, min_seconds=3.0)
                start, end = cursor, cursor + dur
                source = "estimated_from_text"
            if not _valid_interval(start, end, min_duration=1.5):
                unrepairable += 1
                frag["blocking_reason"] = "invalid_fragment_timestamp_unrepairable"
                continue
            frag["start"] = round(float(start), 2)
            frag["end"] = round(float(end), 2)
            frag["timestamp_source"] = source
            repaired += 1
            cursor = max(cursor, float(end))
    plan["fragment_timestamp_repair"] = {
        "repaired_count": repaired,
        "unrepairable_count": unrepairable,
    }
    if unrepairable:
        plan["blocking_reason"] = "invalid_fragment_timestamp_unrepairable"
    return plan


def repair_subtitle_lines_from_voice_script(plan):
    if not isinstance(plan, dict):
        return plan
    repaired = 0
    for kp_item in plan.get("knowledge_points") or []:
        voice_script = kp_item.get("voice_script") or kp_item.get("kp_summary") or ""
        lines = generate_subtitle_lines_from_voice_script(voice_script)
        if kp_item.get("subtitle_lines") != lines:
            repaired += 1
        kp_item["subtitle_lines"] = lines
        kp_item["subtitle_lines_repaired_from_voice_script"] = True
        for frag in kp_item.get("fragments") or []:
            frag["voice_script"] = voice_script
            frag["subtitle_lines"] = lines
            frag["subtitle_lines_repaired_from_voice_script"] = True
            frag["tts_text_source"] = "voice_script"
            frag["subtitle_text_source"] = "subtitle_lines"
    plan["subtitle_repair"] = {"knowledge_points_repaired": repaired}
    return plan


def _normalize_id_list(values):
    return [x for x in (values or []) if isinstance(x, str) and x]


def _text_for_split_item(item):
    parts = []
    for frag in item.get("fragments") or []:
        parts.append(
            frag.get("clean_text")
            or frag.get("cleaned_text")
            or frag.get("source_text")
            or frag.get("raw_text")
            or ""
        )
    parts.append(item.get("text") or "")
    return _clean_tts_text(" ".join(p for p in parts if p))


def _split_text_evenly(text, count):
    text = (text or "").strip()
    if count <= 1 or not text:
        return [text]
    size = max(1, math.ceil(len(text) / count))
    chunks = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + size)
        if end < len(text):
            punct_positions = [text.rfind(p, cursor, end + 1) for p in "。！？；，、"]
            punct = max(punct_positions or [-1])
            if punct > cursor + max(8, size // 2):
                end = punct + 1
        chunks.append(text[cursor:end].strip())
        cursor = end
    while len(chunks) < count:
        chunks.append("")
    return chunks[:count]


def _split_item_to_duration(item, max_kp_duration=90, target_duration=55):
    duration = float(item.get("duration") or 0)
    if duration <= max_kp_duration or duration <= 0:
        return [item]
    count = max(2, math.ceil(duration / float(target_duration)))
    while duration / count > max_kp_duration:
        count += 1
    fragments = item.get("fragments") or []
    if len(fragments) != 1:
        # Multiple small fragments should normally be grouped by caller.
        return [item]
    frag = fragments[0]
    start = float(frag.get("start", 0) or 0)
    end = float(frag.get("end", 0) or 0)
    text_chunks = _split_text_evenly(_text_for_split_item(item), count)
    out = []
    for idx in range(count):
        sub_start = start + (end - start) * idx / count
        sub_end = start + (end - start) * (idx + 1) / count
        sub_frag = dict(frag)
        sub_frag["fragment_id"] = f"{frag.get('fragment_id') or 'frag'}_part_{idx + 1:02d}"
        sub_frag["start"] = round(sub_start, 3)
        sub_frag["end"] = round(sub_end, 3)
        sub_frag["clean_text"] = text_chunks[idx] if idx < len(text_chunks) else ""
        sub_frag["reason"] = (frag.get("reason") or "") + "；因单个片段过长按时间和文本等比例拆分。"
        sub_frag["timestamp_source"] = frag.get("timestamp_source") or "split_from_long_fragment"
        out.append({
            **item,
            "title": item.get("title") or f"片段 {idx + 1}",
            "text": text_chunks[idx] if idx < len(text_chunks) else "",
            "fragments": [sub_frag],
            "duration": sub_end - sub_start,
        })
    return out


def _items_from_atoms(kp, atoms_by_id):
    items = []
    for idx, atom_id in enumerate(_normalize_id_list(kp.get("source_atom_ids"))):
        atom = atoms_by_id.get(atom_id)
        if not atom:
            continue
        start = atom.get("start")
        end = atom.get("end")
        if not _valid_interval(start, end, min_duration=0.3):
            continue
        text = _clean_tts_text(atom.get("atom_text") or atom.get("evidence_text") or "")
        frag = {
            "fragment_id": f"{kp.get('kp_id', 'kp')}_{atom_id}_frag",
            "start": float(start),
            "end": float(end),
            "clean_text": text,
            "reason": "按 knowledge_atom 拆分过长知识点。",
        }
        items.append({
            "title": atom.get("atom_title") or f"知识原子 {idx + 1}",
            "kp_type": atom.get("atom_type") if atom.get("atom_type") in KNOWLEDGE_TYPES else kp.get("kp_type"),
            "source_atom_ids": [atom_id],
            "source_unit_ids": _normalize_id_list(atom.get("source_unit_ids")),
            "text": text,
            "fragments": [frag],
            "duration": _duration(start, end),
        })
    return items


def _items_from_units(kp, units_by_id):
    items = []
    for idx, unit_id in enumerate(_normalize_id_list(kp.get("source_unit_ids"))):
        unit = units_by_id.get(unit_id)
        if not unit:
            continue
        start = unit.get("start")
        end = unit.get("end")
        if not _valid_interval(start, end, min_duration=0.3):
            continue
        text = _clean_tts_text(unit.get("clean_text") or unit.get("raw_text") or "")
        frag = {
            "fragment_id": f"{kp.get('kp_id', 'kp')}_{unit_id}_frag",
            "start": float(start),
            "end": float(end),
            "clean_text": text,
            "reason": "按 semantic_unit 拆分过长知识点。",
        }
        items.append({
            "title": text[:24] or f"语义单元 {idx + 1}",
            "kp_type": _type_for_role(unit.get("unit_role"), text),
            "source_atom_ids": [],
            "source_unit_ids": [unit_id],
            "text": text,
            "fragments": [frag],
            "duration": _duration(start, end),
        })
    return items


def _items_from_fragments(kp):
    items = []
    for idx, frag in enumerate(kp.get("fragments") or []):
        if not _valid_interval(frag.get("start"), frag.get("end"), min_duration=0.3):
            continue
        text = _clean_tts_text(
            frag.get("clean_text") or frag.get("cleaned_text") or kp.get("voice_script") or ""
        )
        items.append({
            "title": frag.get("reason") or f"片段 {idx + 1}",
            "kp_type": kp.get("kp_type"),
            "source_atom_ids": _normalize_id_list(kp.get("source_atom_ids")),
            "source_unit_ids": _normalize_id_list(kp.get("source_unit_ids")),
            "text": text,
            "fragments": [dict(frag)],
            "duration": _duration(frag.get("start"), frag.get("end")),
        })
    return items


def _candidate_split_items(kp, plan, duration):
    atoms_by_id = {a.get("atom_id"): a for a in (plan.get("knowledge_atoms") or [])}
    units_by_id = {u.get("unit_id"): u for u in (plan.get("semantic_units") or [])}
    candidates = [
        _items_from_atoms(kp, atoms_by_id),
        _items_from_units(kp, units_by_id),
        _items_from_fragments(kp),
    ]
    for items in candidates:
        if len(items) < 2:
            continue
        coverage = sum(float(item.get("duration") or 0) for item in items)
        if coverage >= duration * 0.7 or not kp.get("fragments"):
            expanded = []
            for item in items:
                expanded.extend(_split_item_to_duration(item))
            return expanded
    items = _items_from_fragments(kp)
    if items:
        expanded = []
        for item in items:
            expanded.extend(_split_item_to_duration(item))
        return expanded
    return []


def _group_split_items(items, max_kp_duration=90, target_duration=55):
    groups = []
    current = []
    current_duration = 0.0
    for item in items:
        duration = float(item.get("duration") or 0)
        if current and current_duration + duration > target_duration:
            groups.append(current)
            current = []
            current_duration = 0.0
        current.append(item)
        current_duration += duration
        if current_duration >= max_kp_duration:
            groups.append(current)
            current = []
            current_duration = 0.0
    if current:
        groups.append(current)
    return groups


def _make_split_kp(base_kp, group, index, total):
    base_id = base_kp.get("kp_id") or "kp"
    new_id = f"{base_id}_{index:02d}"
    texts = [_text_for_split_item(item) for item in group]
    voice_script = _clean_tts_text(" ".join(t for t in texts if t))
    if voice_script and voice_script[-1] not in "。！？":
        voice_script += "。"
    first_title = (group[0].get("title") or "").strip()
    base_title = (base_kp.get("kp_title") or "知识点").strip()
    if first_title and first_title not in base_title:
        title = f"{base_title}：{first_title[:22]}"
    else:
        title = f"{base_title}（{index}/{total}）"
    source_atom_ids = []
    source_unit_ids = []
    fragments = []
    for item in group:
        source_atom_ids.extend(_normalize_id_list(item.get("source_atom_ids")))
        source_unit_ids.extend(_normalize_id_list(item.get("source_unit_ids")))
        fragments.extend(dict(frag) for frag in (item.get("fragments") or []))
    deps = dict(base_kp.get("dependencies") or {})
    for key in ("requires", "recommended_before", "recommended_after", "supports", "example_of", "contrasts_with", "follows", "duplicates"):
        deps[key] = _normalize_id_list(deps.get(key))
    if index > 1:
        prev_id = f"{base_id}_{index - 1:02d}"
        deps["requires"] = list(dict.fromkeys([prev_id] + deps.get("requires", [])))
        deps["follows"] = list(dict.fromkeys([prev_id] + deps.get("follows", [])))
    if index < total:
        next_id = f"{base_id}_{index + 1:02d}"
        deps["recommended_after"] = list(dict.fromkeys([next_id] + deps.get("recommended_after", [])))
    deps["can_stand_alone"] = bool(index == 1 and deps.get("can_stand_alone", True))
    kp_type = group[0].get("kp_type") if group[0].get("kp_type") in KNOWLEDGE_TYPES else base_kp.get("kp_type")
    new_kp = {
        **base_kp,
        "kp_id": new_id,
        "kp_title": title,
        "kp_type": kp_type if kp_type in KNOWLEDGE_TYPES else "concept",
        "kp_summary": voice_script[:90] or base_kp.get("kp_summary", ""),
        "source_atom_ids": list(dict.fromkeys(source_atom_ids)),
        "source_unit_ids": list(dict.fromkeys(source_unit_ids)),
        "fragments": fragments,
        "voice_script": voice_script,
        "subtitle_lines": generate_subtitle_lines_from_voice_script(voice_script),
        "dependencies": deps,
        "selection_reason": (
            f"由过长知识点 {base_id} 自动拆分，保留其中第 {index} 段可独立讲解的信息。"
        ),
        "split_from_kp_id": base_id,
        "split_part_index": index,
        "split_part_count": total,
    }
    calculate_kp_duration(new_kp)
    return new_kp


def _replace_kp_refs(values, split_map):
    out = []
    for value in values or []:
        if value in split_map:
            out.extend(split_map[value])
        else:
            out.append(value)
    return list(dict.fromkeys(out))


def _update_kp_references_after_split(plan, split_map):
    if not split_map:
        return
    for hook in plan.get("big_hooks") or []:
        hook["recommended_kp_ids"] = _replace_kp_refs(hook.get("recommended_kp_ids"), split_map)
        hook["optional_kp_ids"] = _replace_kp_refs(hook.get("optional_kp_ids"), split_map)
    for path in plan.get("assembly_paths") or []:
        path["ordered_kp_ids"] = _replace_kp_refs(path.get("ordered_kp_ids"), split_map)
    for item in plan.get("knowledge_points") or []:
        deps = item.get("dependencies") if isinstance(item.get("dependencies"), dict) else {}
        for key in ("requires", "recommended_before", "recommended_after", "supports", "example_of", "contrasts_with", "follows", "duplicates"):
            deps[key] = _replace_kp_refs(deps.get(key), split_map)
        item["dependencies"] = deps
    render_plan = plan.get("selected_render_plan")
    if isinstance(render_plan, dict):
        render_plan["selected_kp_ids"] = _replace_kp_refs(render_plan.get("selected_kp_ids"), split_map)
        render_plan["render_units"] = [
            unit for unit in (render_plan.get("render_units") or [])
            if unit.get("kp_id") not in split_map
        ]


def repair_long_knowledge_points(plan, max_kp_duration=90):
    """Split only knowledge_points whose fragment-sum duration is too long."""
    if not isinstance(plan, dict):
        return plan
    original_points = list(plan.get("knowledge_points") or [])
    debug = {
        "original_kp_count": len(original_points),
        "original_long_kps": [],
        "repaired_kp_count": len(original_points),
        "split_kps": [],
        "fallback_used": False,
        "fallback_reason": None,
    }
    repaired_points = []
    split_map = {}
    resolved_by_fragment_sum = False
    split_applied = False
    for item in original_points:
        kp_item = dict(item)
        duration = calculate_kp_duration(kp_item)
        span = float(kp_item.get("continuous_span_duration") or 0)
        frag_sum = float(kp_item.get("fragment_sum_duration") or duration)
        if span > max_kp_duration and frag_sum <= max_kp_duration:
            resolved_by_fragment_sum = True
            debug["original_long_kps"].append({
                "kp_id": kp_item.get("kp_id"),
                "continuous_span_duration": round(span, 3),
                "fragment_sum_duration": round(frag_sum, 3),
                "reason": "span was too long but fragments sum is valid",
            })
            repaired_points.append(kp_item)
            continue
        if duration <= max_kp_duration:
            repaired_points.append(kp_item)
            continue
        debug["original_long_kps"].append({
            "kp_id": kp_item.get("kp_id"),
            "continuous_span_duration": round(span, 3),
            "fragment_sum_duration": round(frag_sum, 3),
            "reason": "fragment_sum_duration too large, split needed",
        })
        items = _candidate_split_items(kp_item, plan, duration)
        if not items:
            repaired_points.append(kp_item)
            debug["fallback_used"] = True
            debug["fallback_reason"] = f"{kp_item.get('kp_id')}: unable_to_split_long_knowledge_point"
            continue
        groups = _group_split_items(items, max_kp_duration=max_kp_duration)
        if len(groups) <= 1:
            repaired_points.append(kp_item)
            debug["fallback_used"] = True
            debug["fallback_reason"] = f"{kp_item.get('kp_id')}: split_produced_single_group"
            continue
        new_kps = [_make_split_kp(kp_item, group, idx + 1, len(groups)) for idx, group in enumerate(groups)]
        if any(calculate_kp_duration(new_kp) > max_kp_duration for new_kp in new_kps):
            repaired_points.append(kp_item)
            debug["fallback_used"] = True
            debug["fallback_reason"] = f"{kp_item.get('kp_id')}: split_still_too_long"
            continue
        split_applied = True
        new_ids = [new_kp.get("kp_id") for new_kp in new_kps]
        split_map[kp_item.get("kp_id")] = new_ids
        debug["split_kps"].append({
            "original_kp_id": kp_item.get("kp_id"),
            "new_kp_ids": new_ids,
        })
        repaired_points.extend(new_kps)

    plan["knowledge_points"] = repaired_points
    _update_kp_references_after_split(plan, split_map)
    debug["repaired_kp_count"] = len(repaired_points)
    plan["long_kp_repair"] = {
        "split_repair_applied": split_applied,
        "duration_recalculated_by_fragments": resolved_by_fragment_sum,
        "split_map": split_map,
        "debug": debug,
    }
    plan["llm_plan_repaired"] = bool(split_applied or resolved_by_fragment_sum)
    if split_applied:
        plan["llm_plan_repair_reason"] = "split_long_knowledge_points"
    elif resolved_by_fragment_sum:
        plan["llm_plan_repair_reason"] = "duration_recalculated_by_fragments"
    else:
        plan["llm_plan_repair_reason"] = "none"
    return plan


def _first_ids_by_type(points, wanted, fallback_ids, limit=4):
    ids = [kp.get("kp_id") for kp in points if kp.get("kp_type") in wanted and kp.get("kp_id")]
    for kid in fallback_ids:
        if kid not in ids:
            ids.append(kid)
    return ids[:limit]


def repair_assembly_paths(plan):
    if not isinstance(plan, dict):
        return plan
    points = plan.get("knowledge_points") or []
    kp_ids = [kp.get("kp_id") for kp in points if kp.get("kp_id")]
    if len(kp_ids) < 3:
        for path in plan.get("assembly_paths") or []:
            path["assembly_status"] = "insufficient_knowledge_points"
        plan["assembly_status"] = "insufficient_knowledge_points"
        return plan

    boss_ids = _first_ids_by_type(points, {"problem", "business_value", "decision", "summary", "concept"}, kp_ids, limit=4)
    tutorial_ids = _first_ids_by_type(points, {"concept", "principle", "operation", "implementation", "pitfall", "workflow", "tool_usage"}, kp_ids, limit=4)
    demo_ids = _first_ids_by_type(points, {"workflow", "tool_usage", "operation", "implementation", "pitfall", "case"}, list(reversed(kp_ids)), limit=4)
    if len(set(tuple(x) for x in (boss_ids, tutorial_ids, demo_ids))) < 3 and len(kp_ids) >= 3:
        boss_ids = kp_ids[:3]
        tutorial_ids = [kp_ids[0], kp_ids[2], kp_ids[1], *kp_ids[3:4]]
        demo_ids = [kp_ids[1], kp_ids[2], *kp_ids[3:4]]

    total_duration = int(sum(_estimate_kp_duration(kp) for kp in points))
    path_specs = [
        ("path_001", "老板汇报版：痛点到落地价值", "boss_report", boss_ids, "痛点 → 价值 → 当前进展 → 下一步", "用最短时间说明这段内容的价值和风险。"),
        ("path_002", "教学讲解版：概念到注意事项", "tutorial", tutorial_ids, "概念 → 原理 → 操作 → 示例 → 注意事项", "让观众理解知识结构并知道后续要补什么。"),
        ("path_003", "实操演示版：配置到排错", "operation_demo", demo_ids, "目标 → 配置 → 操作 → 结果 → 排错", "保留可执行线索，适合做演示前说明。"),
    ]
    plan["assembly_paths"] = [
        {
            "path_id": path_id,
            "path_title": title,
            "path_goal": goal,
            "recommended_for": recommended_for,
            "ordered_kp_ids": list(dict.fromkeys(ids))[:4],
            "narrative_structure": structure,
            "estimated_duration": total_duration,
            "opening_sentence": "先把可验证的信息讲清楚，再提示不确定内容。",
            "closing_sentence": "如果要生成正式视频，建议只使用已经有证据支撑的知识点。",
            "why_recommended": goal,
            "assembly_status": "ok",
        }
        for path_id, title, recommended_for, ids, structure, goal in path_specs
    ]
    for idx, hook in enumerate(plan.get("big_hooks") or []):
        hook["recommended_kp_ids"] = plan["assembly_paths"][min(idx, 2)]["ordered_kp_ids"]
        hook["optional_kp_ids"] = [kid for kid in kp_ids if kid not in hook["recommended_kp_ids"]]
    plan["assembly_status"] = "ok"
    return plan


def repair_semantic_plan_for_render(plan, source_segments=None):
    """Run all deterministic repairs needed before rendering."""
    plan = repair_plan_if_too_few_knowledge_points(plan)
    plan = repair_invalid_fragment_timestamps(plan, source_segments=source_segments)
    plan = repair_long_knowledge_points(plan)
    plan = repair_subtitle_lines_from_voice_script(plan)
    plan = repair_assembly_paths(plan)
    _repair_semantic_plan(plan)
    return plan


def validate_render_quality(knowledge_modules, selected_render_plan=None, require_assembly=False):
    reasons = []
    if not isinstance(knowledge_modules, dict) or not knowledge_modules.get("knowledge_points"):
        return False, ["缺少 knowledge_points"], {"knowledge_point_count": 0}
    points = knowledge_modules.get("knowledge_points") or []
    point_count = len(points)
    invalid_fragment_count = 0
    empty_tts_count = 0
    empty_subtitle_count = 0
    subtitle_source_mismatch_count = 0
    for kp_item in points:
        if not (kp_item.get("voice_script") or "").strip():
            empty_tts_count += 1
        if not kp_item.get("subtitle_lines"):
            empty_subtitle_count += 1
        for frag in kp_item.get("fragments") or []:
            if not _valid_interval(frag.get("start"), frag.get("end"), min_duration=1.5):
                invalid_fragment_count += 1
            if frag.get("subtitle_text_source") not in (None, "subtitle_lines"):
                subtitle_source_mismatch_count += 1
    if point_count < 1:
        reasons.append("selected knowledge_points 数量为 0")
    if invalid_fragment_count:
        reasons.append(f"时间戳无效：{invalid_fragment_count} 个 fragment start >= end 或时长不足")
    if empty_tts_count:
        reasons.append(f"TTS 文本为空：{empty_tts_count} 个 knowledge_point")
    if empty_subtitle_count:
        reasons.append(f"subtitle_lines 为空：{empty_subtitle_count} 个 knowledge_point")
    if subtitle_source_mismatch_count:
        reasons.append("字幕未从 subtitle_lines 生成")

    render_units = (selected_render_plan or {}).get("render_units") or []
    if selected_render_plan is not None:
        if not render_units:
            reasons.append("selected_render_plan.render_units 为空")
        for unit in render_units:
            if not unit.get("kp_id"):
                reasons.append(f"{unit.get('unit_id')}: 缺少 kp_id")
            if not (unit.get("voice_script") or "").strip():
                reasons.append(f"{unit.get('unit_id')}: TTS 文本为空")
            if not unit.get("subtitle_lines"):
                reasons.append(f"{unit.get('unit_id')}: subtitle_lines 为空")
            if unit.get("tts_text_source") != "voice_script":
                reasons.append(f"{unit.get('unit_id')}: tts_text_source 不是 voice_script")
            if unit.get("subtitle_text_source") != "subtitle_lines":
                reasons.append(f"{unit.get('unit_id')}: subtitle_text_source 不是 subtitle_lines")
    paths = knowledge_modules.get("assembly_paths") or []
    path_lengths = [len(p.get("ordered_kp_ids") or []) for p in paths]
    if require_assembly and (not paths or max(path_lengths or [0]) < 2):
        reasons.append("组合路径无效")
    metrics = {
        "knowledge_point_count": point_count,
        "invalid_fragment_count": invalid_fragment_count,
        "assembly_path_avg_length": round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else 0,
    }
    return not reasons, reasons, metrics



def _unit_role_for_segment(seg):
    text = (seg.get("clean_text") or seg.get("raw_text") or "").lower()
    if seg.get("is_filler") or len(text) < 6:
        return "noise"
    if any(k in text for k in ("怎么", "步骤", "点击", "配置", "上传", "选择", "生成", "调用", "排错", "测试")):
        return "operation_step"
    if any(k in text for k in ("比如", "例如", "案例", "演示", "测试")):
        return "example"
    if any(k in text for k in ("问题", "痛点", "难点", "为什么", "卡住")):
        return "problem_setup"
    if any(k in text for k in ("总结", "所以", "最后", "整体")):
        return "summary"
    return "concept_intro"


def _type_for_role(role, text):
    text = (text or "").lower()
    if role == "operation_step":
        return "implementation" if any(k in text for k in ("配置", "接口", "实现", "代码", "参数")) else "operation"
    if role == "example":
        return "case"
    if role == "problem_setup":
        return "problem"
    if role == "summary":
        return "summary"
    if any(k in text for k in ("价值", "老板", "业务", "效率", "成本", "落地")):
        return "business_value"
    if any(k in text for k in ("风险", "限制", "注意", "失败", "错误", "排错")):
        return "pitfall"
    if any(k in text for k in ("流程", "链路", "先", "再", "然后")):
        return "workflow"
    return "concept"


def build_semantic_hierarchical_fallback(cleaned_segments, project_id=""):
    units, atoms, points, discarded, usable_units = [], [], [], [], []
    for idx, seg in enumerate(cleaned_segments or []):
        uid = f"unit_{idx + 1:03d}"
        raw = seg.get("raw_text") or seg.get("clean_text") or ""
        clean = _clean_tts_text(seg.get("clean_text") or raw)
        role = _unit_role_for_segment(seg)
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", 0) or 0)
        unit = {
            "unit_id": uid, "start": start, "end": end, "raw_text": raw, "clean_text": clean,
            "unit_role": role, "contains_action": role == "operation_step",
            "contains_concept": role in {"concept_intro", "problem_setup", "summary"},
            "contains_example": role == "example",
            "noise_level": 0.85 if role == "noise" else (0.35 if len(clean) < 12 else 0.1),
        }
        units.append(unit)
        if role == "noise":
            discarded.append({"unit_id": uid, "reason": "口水话或信息密度较低，没有形成独立知识点。", "discard_type": "filler"})
        else:
            usable_units.append(unit)
    if not usable_units and units:
        usable_units = [u for u in units if u.get("clean_text")]
    for idx, unit in enumerate(usable_units):
        atom_type = _type_for_role(unit.get("unit_role"), unit.get("clean_text"))
        text = unit.get("clean_text") or unit.get("raw_text") or ""
        atoms.append({
            "atom_id": f"atom_{idx + 1:03d}", "source_unit_ids": [unit["unit_id"]],
            "atom_title": text[:28] or f"知识原子 {idx + 1}", "atom_type": atom_type,
            "atom_text": text, "evidence_text": unit.get("raw_text") or text,
            "start": unit.get("start", 0), "end": unit.get("end", 0), "confidence": 0.72,
        })
    buckets, current = [], []
    for atom in atoms:
        current.append(atom)
        if sum(_duration(a.get("start"), a.get("end")) for a in current) >= 8 or len(current) >= 2:
            buckets.append(current); current = []
    if current:
        (buckets[-1].extend(current) if buckets and sum(_duration(a.get("start"), a.get("end")) for a in current) < 5 else buckets.append(current))
    for idx, bucket in enumerate(buckets):
        kp_id = f"kp_{idx + 1:03d}"
        first = bucket[0]
        kp_type = first.get("atom_type") if first.get("atom_type") in KNOWLEDGE_TYPES else "concept"
        start = min(float(a.get("start", 0) or 0) for a in bucket)
        end = max(float(a.get("end", 0) or 0) for a in bucket)
        clean_text = " ".join(a.get("atom_text", "") for a in bucket).strip()
        voice_script = _clean_tts_text(clean_text)
        if voice_script and voice_script[-1] not in "。！？":
            voice_script += "。"
        points.append({
            "kp_id": kp_id, "kp_title": first.get("atom_title") or f"知识点 {idx + 1}",
            "kp_type": kp_type, "kp_summary": voice_script[:80],
            "source_atom_ids": [a.get("atom_id") for a in bucket if a.get("atom_id")],
            "source_unit_ids": [uid for a in bucket for uid in (a.get("source_unit_ids") or [])],
            "fragments": [{"fragment_id": f"frag_{idx + 1:03d}", "start": start, "end": end, "clean_text": clean_text, "reason": "该连续片段表达了一个可独立选择的小知识点。"}],
            "voice_script": voice_script, "subtitle_lines": split_subtitle_lines(voice_script),
            "scores": {"importance": 3 if kp_type in {"transition", "summary"} else 4, "clarity": 0.72, "clip_value": 0.72, "standalone": 0.65 if idx else 0.8, "business_value": 0.7 if kp_type == "business_value" else 0.45, "operation_value": 0.75 if kp_type in {"operation", "implementation"} else 0.35, "novelty": 0.5},
            "dependencies": {"requires": [] if idx == 0 else [points[idx - 1]["kp_id"]] if kp_type in {"operation", "implementation", "case"} else [], "recommended_before": [], "recommended_after": [f"kp_{idx + 2:03d}"] if idx + 1 < len(buckets) else [], "supports": [], "example_of": [], "contrasts_with": [], "follows": [points[idx - 1]["kp_id"]] if idx else [], "duplicates": [], "can_stand_alone": idx == 0 or kp_type not in {"operation", "implementation", "case"}},
            "selection_reason": "适合作为一个可勾选的小知识点。", "notices": [],
        })
    kp_ids = [kp["kp_id"] for kp in points]
    total_duration = int(sum(_estimate_kp_duration(kp) for kp in points))
    main_topic = (points[0]["kp_title"] if points else "录屏知识讲解")
    operation_ids = [kp["kp_id"] for kp in points if kp["kp_type"] in {"operation", "implementation", "workflow", "tool_usage", "pitfall"}]
    hooks = [
        {"hook_id": "hook_001", "hook_title": f"这段录屏最值得提炼的核心知识：{main_topic[:18]}", "hook_type": "boss_report", "opening_hook": "真正有价值的不是把录屏剪短，而是把里面的知识结构讲清楚。", "hook_summary": "把录屏中的痛点、方法和价值组织成适合汇报的短视频。", "target_audience": "老板 / 项目负责人 / 业务同事", "recommended_kp_ids": kp_ids[:4], "optional_kp_ids": kp_ids[4:], "excluded_unit_ids": [d["unit_id"] for d in discarded], "estimated_duration": total_duration, "why_it_works": "先讲价值，再讲关键内容，适合快速汇报。"},
        {"hook_id": "hook_002", "hook_title": f"从概念到操作，讲清楚：{main_topic[:22]}", "hook_type": "tutorial", "opening_hook": "如果只看操作，很容易漏掉背后的概念和前置条件。", "hook_summary": "按概念、原理、操作和注意事项组织成教学讲解。", "target_audience": "学习者 / 执行同事 / 新手用户", "recommended_kp_ids": kp_ids, "optional_kp_ids": [], "excluded_unit_ids": [d["unit_id"] for d in discarded], "estimated_duration": total_duration, "why_it_works": "从理解到复现，降低观看门槛。"},
        {"hook_id": "hook_003", "hook_title": f"照着做一遍：{main_topic[:22]}", "hook_type": "operation_demo", "opening_hook": "这条视频只保留目标、步骤、结果和排错线索。", "hook_summary": "偏向实操演示，突出可复现步骤。", "target_audience": "操作者 / 实施人员", "recommended_kp_ids": operation_ids or kp_ids[:3], "optional_kp_ids": [kp for kp in kp_ids if kp not in (operation_ids or kp_ids[:3])], "excluded_unit_ids": [d["unit_id"] for d in discarded], "estimated_duration": total_duration, "why_it_works": "筛掉背景噪音，保留能跟做的部分。"},
    ]
    paths = [
        {"path_id": "path_001", "path_title": "老板汇报版：从价值到落地", "path_goal": "用最短时间说明这个知识点为什么值得做。", "recommended_for": "boss_report", "ordered_kp_ids": hooks[0]["recommended_kp_ids"], "narrative_structure": "痛点 → 价值 → 当前进展 → 后续落地", "estimated_duration": total_duration, "opening_sentence": hooks[0]["opening_hook"], "closing_sentence": "最后可以把这些知识点组合成一条更适合汇报的短视频。", "why_recommended": "适合对非技术受众解释内容价值。"},
        {"path_id": "path_002", "path_title": "教学讲解版：从概念到操作", "path_goal": "让观众理解并能复现操作。", "recommended_for": "tutorial", "ordered_kp_ids": kp_ids, "narrative_structure": "概念 → 原理 → 操作 → 示例 → 注意事项", "estimated_duration": total_duration, "opening_sentence": hooks[1]["opening_hook"], "closing_sentence": "掌握概念和步骤之后，再去操作会更稳。", "why_recommended": "适合完整学习。"},
        {"path_id": "path_003", "path_title": "实操演示版：目标到排错", "path_goal": "保留可执行步骤和结果验证。", "recommended_for": "operation_demo", "ordered_kp_ids": hooks[2]["recommended_kp_ids"], "narrative_structure": "目标 → 步骤 → 关键配置 → 结果 → 排错", "estimated_duration": total_duration, "opening_sentence": hooks[2]["opening_hook"], "closing_sentence": "如果中间失败，优先检查配置、素材和接口返回。", "why_recommended": "适合跟做和复盘。"},
    ]
    return {"project_id": project_id, "slicing_mode": "semantic_hierarchical_planning", "source_summary": {"main_topic": main_topic, "speaker_intent": "围绕录屏内容进行讲解、说明或演示。", "content_type": "mixed", "suitable_video_styles": ["boss_report", "tutorial", "operation_demo"], "overall_quality_notes": "该计划由 deterministic 语义兜底生成，建议配置 LLM 获得更精细的语义关系。"}, "semantic_units": units, "knowledge_atoms": atoms, "knowledge_points": points, "big_hooks": hooks, "assembly_paths": paths, "discarded_content": discarded}


def _segments_total_duration(segments):
    starts, ends = [], []
    for seg in segments or []:
        try:
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", 0) or 0)
        except Exception:
            continue
        if start < end:
            starts.append(start)
            ends.append(end)
    return max(ends) - min(starts) if starts and ends else 0.0


def chunked_planning_reasons(cleaned_segments, full_text, max_segments=80,
                             max_chars=12000, max_duration_seconds=900,
                             previous_error=None):
    reasons = []
    if len(cleaned_segments or []) > max_segments:
        reasons.append("segments too many")
    if len(full_text or "") > max_chars:
        reasons.append("full_text too long")
    if _segments_total_duration(cleaned_segments) > max_duration_seconds:
        reasons.append("duration too long")
    err = (previous_error or "").lower()
    if any(key in err for key in ("context", "maximum context", "too large", "duration", "schema")):
        reasons.append("previous plan error unstable")
    return reasons


def should_use_chunked_planning(cleaned_segments, full_text, max_segments=80,
                                max_chars=12000, max_duration_seconds=900,
                                previous_error=None):
    return bool(chunked_planning_reasons(
        cleaned_segments, full_text,
        max_segments=max_segments,
        max_chars=max_chars,
        max_duration_seconds=max_duration_seconds,
        previous_error=previous_error,
    ))


def _split_text_for_planning_chunks(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + max_chars)
        if end < len(text):
            punct = max([text.rfind(p, cursor, end + 1) for p in "。！？；\n"] or [-1])
            if punct > cursor + max_chars * 0.55:
                end = punct + 1
        chunks.append(text[cursor:end].strip())
        cursor = end
    return [chunk for chunk in chunks if chunk]


def _normalise_segments_for_chunks(cleaned_segments, max_chunk_chars):
    segs = []
    cursor = 0.0
    real_time = any(_duration(seg.get("start"), seg.get("end")) > 0 for seg in cleaned_segments or [])
    for idx, seg in enumerate(cleaned_segments or []):
        text = (
            seg.get("clean_text") or seg.get("text") or seg.get("raw_text")
            or seg.get("cleaned_text") or ""
        ).strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", 0) or 0)
        except Exception:
            start, end = 0.0, 0.0
        if not real_time or end <= start:
            dur = max(1.0, len(text) / 5.0)
            start, end = cursor, cursor + dur
            cursor = end
            estimated = True
        else:
            estimated = bool(seg.get("estimated_timestamps"))
            cursor = max(cursor, end)
        parts = _split_text_for_planning_chunks(text, max(1000, max_chunk_chars))
        if len(parts) <= 1:
            item = dict(seg)
            item.update({
                "segment_id": seg.get("segment_id") or f"seg_{idx + 1:04d}",
                "start": start,
                "end": end,
                "clean_text": text,
                "raw_text": seg.get("raw_text") or seg.get("text") or text,
                "estimated_timestamps": estimated,
            })
            segs.append(item)
            continue
        total_len = sum(len(part) for part in parts) or 1
        part_start = start
        for part_idx, part in enumerate(parts):
            part_duration = max(1.0, (end - start) * (len(part) / total_len))
            part_end = end if part_idx == len(parts) - 1 else part_start + part_duration
            item = dict(seg)
            item.update({
                "segment_id": f"{seg.get('segment_id') or f'seg_{idx + 1:04d}'}_part_{part_idx + 1:02d}",
                "start": round(part_start, 3),
                "end": round(part_end, 3),
                "clean_text": part,
                "raw_text": part,
                "estimated_timestamps": True,
            })
            segs.append(item)
            part_start = part_end
    return segs


def split_segments_into_planning_chunks(cleaned_segments, target_chunk_seconds=300,
                                        max_chunk_seconds=420, overlap_seconds=20,
                                        max_chunk_chars=6000):
    segments = _normalise_segments_for_chunks(cleaned_segments or [], max_chunk_chars=max_chunk_chars)
    chunks = []
    idx = 0
    while idx < len(segments):
        chunk_segments = []
        char_count = 0
        start = float(segments[idx].get("start", 0) or 0)
        end = start
        j = idx
        while j < len(segments):
            seg = segments[j]
            seg_text = seg.get("clean_text") or seg.get("raw_text") or ""
            seg_end = float(seg.get("end", 0) or 0)
            next_duration = seg_end - start
            next_chars = char_count + len(seg_text)
            if chunk_segments and (next_duration > max_chunk_seconds or next_chars > max_chunk_chars):
                break
            chunk_segments.append(seg)
            char_count = next_chars
            end = max(end, seg_end)
            j += 1
            if chunk_segments and (end - start >= target_chunk_seconds or char_count >= max_chunk_chars * 0.85):
                break
        if not chunk_segments:
            chunk_segments.append(segments[idx])
            j = idx + 1
            end = float(segments[idx].get("end", start) or start)
            char_count = len(segments[idx].get("clean_text") or "")
        chunk_id = f"chunk_{len(chunks) + 1:03d}"
        text = "\n".join(seg.get("clean_text") or seg.get("raw_text") or "" for seg in chunk_segments).strip()
        chunks.append({
            "chunk_id": chunk_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "segment_ids": [seg.get("segment_id") for seg in chunk_segments if seg.get("segment_id")],
            "segments": chunk_segments,
            "text": text,
            "char_count": len(text),
            "duration": round(max(0.0, end - start), 3),
            "overlap_with_previous": bool(chunks),
            "overlap_with_next": False,
        })
        if j >= len(segments):
            break
        overlap_start_time = max(start, end - float(overlap_seconds or 0))
        next_idx = j
        for back_idx in range(j - 1, idx - 1, -1):
            if float(segments[back_idx].get("start", 0) or 0) <= overlap_start_time:
                next_idx = back_idx
                break
        if next_idx <= idx:
            next_idx = j
        if chunks:
            chunks[-1]["overlap_with_next"] = next_idx < j
        idx = next_idx
    return chunks


def _prefixed_id(chunk_id, raw_id, kind, idx):
    raw = str(raw_id or f"{kind}_{idx + 1:03d}")
    return raw if raw.startswith(f"{chunk_id}_") else f"{chunk_id}_{raw}"


def _normalise_local_chunk_plan(plan, chunk, fallback_used=False, error=None):
    chunk_id = chunk.get("chunk_id") or "chunk_000"
    local_units = plan.get("local_semantic_units") or plan.get("semantic_units") or []
    local_atoms = plan.get("local_knowledge_atoms") or plan.get("knowledge_atoms") or []
    local_kps = plan.get("local_candidate_knowledge_points") or plan.get("knowledge_points") or []
    unit_id_map = {}
    atom_id_map = {}
    units = []
    for idx, unit in enumerate(local_units):
        old_id = unit.get("unit_id") or f"unit_{idx + 1:03d}"
        new_id = _prefixed_id(chunk_id, old_id, "unit", idx)
        unit_id_map[old_id] = new_id
        units.append({
            **unit,
            "unit_id": new_id,
            "chunk_id": chunk_id,
            "start": unit.get("start", chunk.get("start", 0)),
            "end": unit.get("end", chunk.get("end", 0)),
        })
    atoms = []
    for idx, atom in enumerate(local_atoms):
        old_id = atom.get("atom_id") or f"atom_{idx + 1:03d}"
        new_id = _prefixed_id(chunk_id, old_id, "atom", idx)
        atom_id_map[old_id] = new_id
        atoms.append({
            **atom,
            "atom_id": new_id,
            "chunk_id": chunk_id,
            "source_unit_ids": [unit_id_map.get(uid, _prefixed_id(chunk_id, uid, "unit", idx)) for uid in (atom.get("source_unit_ids") or [])],
            "atom_type": atom.get("atom_type") if atom.get("atom_type") in KNOWLEDGE_TYPES else "concept",
            "evidence_text": atom.get("evidence_text") or atom.get("atom_text") or "",
            "start": atom.get("start", chunk.get("start", 0)),
            "end": atom.get("end", chunk.get("end", 0)),
        })
    points = []
    for idx, item in enumerate(local_kps):
        old_id = item.get("local_kp_id") or item.get("kp_id") or f"kp_{idx + 1:03d}"
        local_id = _prefixed_id(chunk_id, old_id, "kp", idx)
        fragments = []
        for frag_idx, frag in enumerate(item.get("fragments") or []):
            start = frag.get("start", chunk.get("start", 0))
            end = frag.get("end", chunk.get("end", 0))
            fragments.append({
                **frag,
                "fragment_id": _prefixed_id(chunk_id, frag.get("fragment_id"), "frag", frag_idx),
                "start": start,
                "end": end,
                "clean_text": frag.get("clean_text") or frag.get("cleaned_text") or item.get("kp_summary") or "",
            })
        if not fragments:
            fragments = [{
                "fragment_id": f"{local_id}_frag_01",
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
                "clean_text": item.get("kp_summary") or item.get("voice_script") or chunk.get("text", ""),
                "reason": "chunk-level fallback fragment",
            }]
        voice_script = _clean_tts_text(item.get("voice_script") or item.get("kp_summary") or "")
        if voice_script and voice_script[-1] not in "。！？":
            voice_script += "。"
        kp_type = item.get("kp_type") if item.get("kp_type") in KNOWLEDGE_TYPES else "concept"
        point = {
            "local_kp_id": local_id,
            "kp_id": local_id,
            "chunk_id": chunk_id,
            "kp_title": item.get("kp_title") or f"{chunk_id} 知识点 {idx + 1}",
            "kp_type": kp_type,
            "kp_summary": item.get("kp_summary") or voice_script[:90],
            "source_atom_ids": [atom_id_map.get(aid, _prefixed_id(chunk_id, aid, "atom", idx)) for aid in (item.get("source_atom_ids") or [])],
            "source_unit_ids": [unit_id_map.get(uid, _prefixed_id(chunk_id, uid, "unit", idx)) for uid in (item.get("source_unit_ids") or [])],
            "fragments": fragments,
            "voice_script": voice_script,
            "subtitle_lines": generate_subtitle_lines_from_voice_script(voice_script),
            "scores": item.get("scores") or _kp_score_defaults(kp_type, 0.55 if fallback_used else 0.72),
            "dependencies": item.get("dependencies") or {"requires": [], "recommended_before": [], "recommended_after": [], "supports": [], "example_of": [], "contrasts_with": [], "follows": [], "duplicates": [], "can_stand_alone": bool(item.get("can_stand_alone", True))},
            "selection_reason": item.get("selection_reason") or "从局部语义块中抽取的候选知识点。",
            "notices": item.get("notices") or (["该 chunk 使用 deterministic local fallback。"] if fallback_used else []),
        }
        calculate_kp_duration(point)
        points.append(point)
    return {
        "chunk_id": chunk_id,
        "status": "fallback" if fallback_used else "success",
        "error": error,
        "local_summary": plan.get("local_summary") or plan.get("source_summary") or {},
        "local_semantic_units": units,
        "local_knowledge_atoms": atoms,
        "local_candidate_knowledge_points": points,
        "local_discarded_content": plan.get("local_discarded_content") or plan.get("discarded_content") or [],
    }


def _fallback_chunk_plan(chunk):
    fallback = build_semantic_hierarchical_fallback(chunk.get("segments") or [], project_id=chunk.get("chunk_id", ""))
    return _normalise_local_chunk_plan(fallback, chunk, fallback_used=True)


def _all_local_items(chunk_plans, key):
    items = []
    for plan in chunk_plans or []:
        items.extend(plan.get(key) or [])
    return items


def _renumber_global_kps(points):
    out = []
    id_map = {}
    for idx, item in enumerate(points or []):
        old_id = item.get("kp_id") or item.get("local_kp_id") or f"kp_{idx + 1:03d}"
        new_id = f"kp_{idx + 1:03d}"
        id_map[old_id] = new_id
        kp_item = dict(item)
        kp_item["kp_id"] = new_id
        kp_item["merged_from"] = item.get("merged_from") or [old_id]
        deps = kp_item.get("dependencies") if isinstance(kp_item.get("dependencies"), dict) else {}
        for dep_key in ("requires", "recommended_before", "recommended_after", "supports", "example_of", "contrasts_with", "follows", "duplicates"):
            deps[dep_key] = [id_map.get(x, x) for x in (deps.get(dep_key) or [])]
        deps["can_stand_alone"] = bool(deps.get("can_stand_alone", not deps.get("requires")))
        kp_item["dependencies"] = deps
        calculate_kp_duration(kp_item)
        out.append(kp_item)
    return out, id_map


def _deterministic_merge_chunk_plans(chunk_plans, project_id=""):
    units = _all_local_items(chunk_plans, "local_semantic_units")
    atoms = _all_local_items(chunk_plans, "local_knowledge_atoms")
    local_points = _all_local_items(chunk_plans, "local_candidate_knowledge_points")
    merged_by_title = {}
    merge_decisions = []
    duplicates = []
    for point in local_points:
        title_key = re.sub(r"\s+", "", (point.get("kp_title") or "")[:18].lower())
        if not title_key:
            title_key = point.get("local_kp_id") or point.get("kp_id")
        if title_key not in merged_by_title:
            merged_by_title[title_key] = dict(point)
            merged_by_title[title_key]["merged_from"] = [point.get("local_kp_id") or point.get("kp_id")]
            merged_by_title[title_key]["source_chunk_ids"] = [point.get("chunk_id")] if point.get("chunk_id") else []
            continue
        target = merged_by_title[title_key]
        target["fragments"] = (target.get("fragments") or []) + (point.get("fragments") or [])
        target["source_atom_ids"] = list(dict.fromkeys((target.get("source_atom_ids") or []) + (point.get("source_atom_ids") or [])))
        target["source_unit_ids"] = list(dict.fromkeys((target.get("source_unit_ids") or []) + (point.get("source_unit_ids") or [])))
        target["merged_from"].append(point.get("local_kp_id") or point.get("kp_id"))
        if point.get("chunk_id") and point.get("chunk_id") not in target["source_chunk_ids"]:
            target["source_chunk_ids"].append(point.get("chunk_id"))
        duplicates.append(point.get("local_kp_id") or point.get("kp_id"))
    points, _ = _renumber_global_kps(list(merged_by_title.values()))
    for point in points:
        if len(point.get("merged_from") or []) > 1:
            merge_decisions.append({
                "merged_kp_id": point.get("kp_id"),
                "merged_from": point.get("merged_from"),
                "reason": "标题和语义相近，合并为跨 chunk 知识点。",
            })
    main_topic = (points[0].get("kp_title") if points else "长录屏知识讲解")
    return {
        "project_id": project_id,
        "slicing_mode": "semantic_hierarchical_planning",
        "source_summary": {
            "main_topic": main_topic,
            "speaker_intent": "围绕长录屏内容进行分块分析和全局重组。",
            "content_type": "mixed",
            "suitable_video_styles": ["boss_report", "tutorial", "operation_demo", "short_video"],
            "overall_quality_notes": "该计划由长文本分块分析合并生成。",
        },
        "semantic_units": units,
        "knowledge_atoms": atoms,
        "knowledge_points": points,
        "big_hooks": [],
        "assembly_paths": [],
        "discarded_content": _all_local_items(chunk_plans, "local_discarded_content"),
        "merge_decisions": merge_decisions,
        "discarded_duplicates": duplicates,
    }


def _build_default_hooks_for_points(plan):
    points = plan.get("knowledge_points") or []
    kp_ids = [kp.get("kp_id") for kp in points if kp.get("kp_id")]
    if not kp_ids:
        plan["big_hooks"] = []
        plan["assembly_paths"] = []
        return plan
    topic = ((plan.get("source_summary") or {}).get("main_topic") or points[0].get("kp_title") or "这段录屏").strip()
    recommended = kp_ids[:5]
    operation_ids = [kp.get("kp_id") for kp in points if kp.get("kp_type") in {"operation", "implementation", "workflow", "tool_usage", "case", "pitfall"}][:5]
    plan["big_hooks"] = [
        {
            "hook_id": "hook_001",
            "hook_title": f"为什么这段录屏真正值得看的是：{topic[:24]}",
            "hook_type": "boss_report",
            "opening_hook": "长录屏最有价值的不是时长，而是里面能被提炼出来的决策和方法。",
            "hook_summary": "从全局知识点中选出适合快速汇报的主线。",
            "target_audience": "老板 / 项目负责人 / 业务同事",
            "recommended_kp_ids": recommended[:4],
            "optional_kp_ids": [kid for kid in kp_ids if kid not in recommended[:4]],
            "excluded_unit_ids": [],
            "estimated_duration": int(sum(calculate_kp_duration(kp) for kp in points[:4])),
            "why_it_works": "先抓主题价值，再选关键知识点，适合把长内容压缩成一条完整短视频。",
        },
        {
            "hook_id": "hook_002",
            "hook_title": f"从概念到操作，讲清楚：{topic[:24]}",
            "hook_type": "tutorial",
            "opening_hook": "如果只看片段，很容易漏掉前后关系；这条视频按知识顺序讲清楚。",
            "hook_summary": "按概念、原理、操作和注意事项组织教学讲解。",
            "target_audience": "学习者 / 执行同事",
            "recommended_kp_ids": recommended,
            "optional_kp_ids": [kid for kid in kp_ids if kid not in recommended],
            "excluded_unit_ids": [],
            "estimated_duration": int(sum(calculate_kp_duration(kp) for kp in points[:5])),
            "why_it_works": "把长视频拆成可理解的小知识点，再按学习路径重组。",
        },
        {
            "hook_id": "hook_003",
            "hook_title": f"照着做一遍：{topic[:24]}",
            "hook_type": "operation_demo",
            "opening_hook": "这条视频只保留目标、关键步骤、结果和排错线索。",
            "hook_summary": "偏向实操演示，突出可复现步骤。",
            "target_audience": "操作者 / 实施人员",
            "recommended_kp_ids": operation_ids or recommended[:4],
            "optional_kp_ids": [kid for kid in kp_ids if kid not in (operation_ids or recommended[:4])],
            "excluded_unit_ids": [],
            "estimated_duration": int(sum(calculate_kp_duration(kp) for kp in points[:4])),
            "why_it_works": "筛掉背景噪音，保留能跟做的部分。",
        },
    ]
    return plan


def _read_prompt_template(prompts_dir, filename):
    path = Path(prompts_dir or "prompts") / filename
    return path.read_text(encoding="utf-8")


def _emit_progress(progress_callback, stage, message, **extra):
    if callable(progress_callback):
        try:
            progress_callback({"stage": stage, "message": message, **extra})
        except Exception:
            pass


def _write_planning_heartbeat(debug_dir, start_time, current_stage,
                              current_chunk=None, total_chunks=None,
                              last_success=None, last_error=None):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_stage": current_stage,
        "current_chunk": current_chunk,
        "total_chunks": total_chunks,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "last_success": last_success,
        "last_error": last_error,
    }
    try:
        (debug_dir / "planning_heartbeat.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return payload


def _is_valid_cached_chunk_plan(plan, chunk_id):
    return (
        isinstance(plan, dict)
        and plan.get("chunk_id") == chunk_id
        and isinstance(plan.get("local_candidate_knowledge_points"), list)
        and isinstance(plan.get("local_knowledge_atoms"), list)
    )


def plan_knowledge_modules_chunked(cleaned_segments, full_text, voice_style, project_dir,
                                   llm_client, prompts_dir="prompts",
                                   progress_callback=None):
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = project_dir / "chunk_plans"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = project_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    planning_start = time.time()
    reasons = chunked_planning_reasons(cleaned_segments, full_text)
    _emit_progress(progress_callback, "splitting_chunks", "正在切分 chunk")
    _write_planning_heartbeat(debug_dir, planning_start, "splitting_chunks")
    chunks = split_segments_into_planning_chunks(cleaned_segments or [])
    (project_dir / "planning_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    chunk_template = None
    merge_template = None
    hook_template = None
    for filename, var_name in (
        ("plan_chunk_knowledge.md", "chunk"),
        ("merge_global_knowledge.md", "merge"),
        ("plan_hooks_from_global_kps.md", "hook"),
    ):
        try:
            template = _read_prompt_template(prompts_dir, filename)
        except Exception:
            template = None
        if var_name == "chunk":
            chunk_template = template
        elif var_name == "merge":
            merge_template = template
        else:
            hook_template = template

    chunk_plans = []
    failed_chunks = []
    cached_chunk_count = 0
    for chunk in chunks:
        local_plan = None
        error = None
        current_index = len(chunk_plans) + 1
        elapsed = time.time() - planning_start
        _emit_progress(
            progress_callback,
            "chunk_planning",
            f"正在分析 chunk {current_index}/{len(chunks)}",
            current_chunk=current_index,
            total_chunks=len(chunks),
        )
        _write_planning_heartbeat(
            debug_dir, planning_start, "chunk_planning",
            current_chunk=current_index, total_chunks=len(chunks),
            last_success=(chunk_plans[-1].get("chunk_id") if chunk_plans else None),
            last_error=None,
        )
        cache_path = chunk_dir / f"{chunk['chunk_id']}_plan.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if _is_valid_cached_chunk_plan(cached, chunk["chunk_id"]):
                    local_plan = cached
                    local_plan["cache_hit"] = True
                    cached_chunk_count += 1
            except Exception as exc:
                error = f"cached chunk plan invalid: {exc}"
        if not local_plan and elapsed > PLANNING_MAX_TOTAL_SECONDS:
            error = f"planning total timeout after {PLANNING_MAX_TOTAL_SECONDS}s"
            failed_chunks.append({"chunk_id": chunk.get("chunk_id"), "error": error})
            local_plan = _fallback_chunk_plan(chunk)
            local_plan["error"] = error
            local_plan["timeout_fallback"] = True
            chunk_plans.append(local_plan)
            cache_path.write_text(json.dumps(local_plan, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_planning_heartbeat(
                debug_dir, planning_start, "chunk_planning_timeout",
                current_chunk=current_index, total_chunks=len(chunks),
                last_success=(chunk_plans[-2].get("chunk_id") if len(chunk_plans) > 1 else None),
                last_error=error,
            )
            continue
        if not local_plan and chunk_template:
            chunk_segments_json = json.dumps(chunk.get("segments") or [], ensure_ascii=False, indent=2)
            prompt = (
                chunk_template
                .replace("{{CHUNK_ID}}", chunk.get("chunk_id") or "")
                .replace("{{CHUNK_START}}", str(chunk.get("start", 0)))
                .replace("{{CHUNK_END}}", str(chunk.get("end", 0)))
                .replace("{{CHUNK_TEXT}}", chunk.get("text") or "")
                .replace("{{CHUNK_SEGMENTS_JSON}}", chunk_segments_json)
                .replace("{{VOICE_STYLE}}", voice_style or "讲解风格")
            )
            call = _call_llm_json_for_planning(llm_client, prompt, stage="llm_plan_chunk")
            (chunk_dir / f"{chunk['chunk_id']}_prompt.md").write_text(prompt, encoding="utf-8")
            (chunk_dir / f"{chunk['chunk_id']}_raw_response.txt").write_text(call.get("raw_response") or "", encoding="utf-8")
            if isinstance(call.get("result"), dict):
                try:
                    local_plan = _normalise_local_chunk_plan(call["result"], chunk)
                except Exception as exc:
                    error = f"normalise failed: {exc}"
            else:
                error = call.get("error") or "chunk LLM did not return JSON"
        elif not local_plan:
            error = "plan_chunk_knowledge.md missing"
        if not local_plan:
            failed_chunks.append({"chunk_id": chunk.get("chunk_id"), "error": error})
            local_plan = _fallback_chunk_plan(chunk)
            local_plan["error"] = error
            local_plan["chunk_fallback_used"] = True
        elif local_plan.get("status") == "fallback" and not any(item.get("chunk_id") == chunk.get("chunk_id") for item in failed_chunks):
            failed_chunks.append({
                "chunk_id": chunk.get("chunk_id"),
                "error": local_plan.get("error") or "cached fallback chunk plan",
            })
        chunk_plans.append(local_plan)
        cache_path.write_text(
            json.dumps(local_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_planning_heartbeat(
            debug_dir, planning_start, "chunk_planning",
            current_chunk=current_index, total_chunks=len(chunks),
            last_success=chunk.get("chunk_id") if local_plan.get("status") == "success" else None,
            last_error=error,
        )

    global_merge_status = "fallback"
    hook_plan_status = "fallback"
    global_plan = None
    local_atoms = _all_local_items(chunk_plans, "local_knowledge_atoms")
    local_points = _all_local_items(chunk_plans, "local_candidate_knowledge_points")
    source_summary_seed = {
        "main_topic": "长录屏知识讲解",
        "speaker_intent": "对长内容进行分块分析后重组为短视频知识点。",
        "content_type": "mixed",
        "suitable_video_styles": ["boss_report", "tutorial", "operation_demo", "short_video"],
        "overall_quality_notes": "长文本模式自动启用。",
    }
    can_continue_llm = (time.time() - planning_start) <= PLANNING_MAX_TOTAL_SECONDS
    _emit_progress(progress_callback, "global_merge", "正在合并全局知识点")
    _write_planning_heartbeat(debug_dir, planning_start, "global_merge", total_chunks=len(chunks))
    if merge_template and can_continue_llm:
        merge_prompt = (
            merge_template
            .replace("{{LOCAL_KNOWLEDGE_ATOMS_JSON}}", json.dumps(local_atoms, ensure_ascii=False, indent=2))
            .replace("{{LOCAL_CANDIDATE_KPS_JSON}}", json.dumps(local_points, ensure_ascii=False, indent=2))
            .replace("{{SOURCE_SUMMARY_JSON}}", json.dumps(source_summary_seed, ensure_ascii=False, indent=2))
            .replace("{{VOICE_STYLE}}", voice_style or "讲解风格")
        )
        merge_call = _call_llm_json_for_planning(llm_client, merge_prompt, stage="llm_global_merge")
        (debug_dir / "debug_global_merge_prompt.md").write_text(merge_prompt, encoding="utf-8")
        (debug_dir / "debug_global_merge_raw_response.txt").write_text(merge_call.get("raw_response") or "", encoding="utf-8")
        if isinstance(merge_call.get("result"), dict) and merge_call["result"].get("knowledge_points"):
            merged = merge_call["result"]
            points, _ = _renumber_global_kps(merged.get("knowledge_points") or [])
            global_plan = {
                "project_id": project_dir.name,
                "slicing_mode": "semantic_hierarchical_planning",
                "source_summary": merged.get("source_summary") or source_summary_seed,
                "semantic_units": _all_local_items(chunk_plans, "local_semantic_units"),
                "knowledge_atoms": local_atoms,
                "knowledge_points": points,
                "big_hooks": [],
                "assembly_paths": [],
                "discarded_content": _all_local_items(chunk_plans, "local_discarded_content"),
                "merge_decisions": merged.get("merge_decisions") or [],
                "discarded_duplicates": merged.get("discarded_duplicates") or [],
            }
            global_merge_status = "success"
    if not global_plan:
        global_plan = _deterministic_merge_chunk_plans(chunk_plans, project_id=project_dir.name)

    _emit_progress(progress_callback, "hook_planning", "正在生成大选题")
    _write_planning_heartbeat(debug_dir, planning_start, "hook_planning", total_chunks=len(chunks))
    can_continue_llm = (time.time() - planning_start) <= PLANNING_MAX_TOTAL_SECONDS
    if hook_template and global_plan.get("knowledge_points") and can_continue_llm:
        hook_prompt = (
            hook_template
            .replace("{{GLOBAL_KNOWLEDGE_POINTS_JSON}}", json.dumps(global_plan.get("knowledge_points") or [], ensure_ascii=False, indent=2))
            .replace("{{SOURCE_SUMMARY_JSON}}", json.dumps(global_plan.get("source_summary") or {}, ensure_ascii=False, indent=2))
            .replace("{{VOICE_STYLE}}", voice_style or "讲解风格")
        )
        hook_call = _call_llm_json_for_planning(llm_client, hook_prompt, stage="llm_hook_plan")
        (debug_dir / "debug_hook_plan_prompt.md").write_text(hook_prompt, encoding="utf-8")
        (debug_dir / "debug_hook_plan_raw_response.txt").write_text(hook_call.get("raw_response") or "", encoding="utf-8")
        if isinstance(hook_call.get("result"), dict):
            hooks = hook_call["result"].get("big_hooks") or []
            paths = hook_call["result"].get("assembly_paths") or []
            if hooks and paths:
                global_plan["big_hooks"] = hooks
                global_plan["assembly_paths"] = paths
                hook_plan_status = "success"
    if not global_plan.get("big_hooks") or not global_plan.get("assembly_paths"):
        _build_default_hooks_for_points(global_plan)

    _emit_progress(progress_callback, "repairing_plan", "正在修复计划")
    _write_planning_heartbeat(debug_dir, planning_start, "repairing_plan", total_chunks=len(chunks))
    global_plan["used_chunked_planning"] = True
    global_plan["chunk_count"] = len(chunks)
    global_plan["successful_chunk_count"] = len(chunks) - len(failed_chunks)
    global_plan["failed_chunk_count"] = len(failed_chunks)
    global_plan["failed_chunks"] = [item.get("chunk_id") for item in failed_chunks]
    global_plan["failed_chunk_details"] = failed_chunks
    global_plan["cached_chunk_count"] = cached_chunk_count
    global_plan["chunk_fallback_used"] = bool(failed_chunks)
    global_plan["chunked_planning_reason"] = " / ".join(reasons or ["manual chunked planning"])
    global_plan["global_merge_status"] = global_merge_status
    global_plan["hook_plan_status"] = hook_plan_status
    global_plan["chunk_plan_dir"] = str(chunk_dir)
    global_plan = repair_long_knowledge_points(global_plan)
    global_plan = repair_invalid_fragment_timestamps(global_plan, source_segments=cleaned_segments)
    global_plan = repair_subtitle_lines_from_voice_script(global_plan)
    global_plan = repair_assembly_paths(global_plan)
    _repair_semantic_plan(global_plan)
    default_hook = (global_plan.get("big_hooks") or [{}])[0]
    default_kp_ids = default_hook.get("recommended_kp_ids") or []
    selected_plan = build_selected_render_plan(
        default_kp_ids, [], global_plan, fragment_order="user_order", voice_style=voice_style, title="长文本分块规划默认合集"
    )
    selected_plan["render_output_mode"] = "single_complete_video"
    selected_plan["selected_hook_id"] = default_hook.get("hook_id")
    selected_plan["selected_hook_ids"] = [default_hook.get("hook_id")] if default_hook.get("hook_id") else []
    selected_plan["selected_kp_ids"] = [unit.get("kp_id") for unit in selected_plan.get("render_units") or []]
    selected_plan["final_video_title"] = default_hook.get("hook_title")
    selected_plan["final_video_opening_hook"] = default_hook.get("opening_hook")
    selected_plan["final_video_structure"] = "按所选知识点顺序讲解"
    selected_plan["visible_output_count"] = 1
    global_plan["selected_render_plan"] = selected_plan
    _emit_progress(progress_callback, "writing_results", "正在写入结果")
    _write_planning_heartbeat(debug_dir, planning_start, "writing_results", total_chunks=len(chunks))
    (debug_dir / "debug_chunked_planning.json").write_text(json.dumps({
        "used_chunked_planning": True,
        "chunk_count": len(chunks),
        "successful_chunk_count": len(chunks) - len(failed_chunks),
        "failed_chunk_count": len(failed_chunks),
        "cached_chunk_count": cached_chunk_count,
        "chunk_fallback_used": bool(failed_chunks),
        "chunked_planning_reason": global_plan["chunked_planning_reason"],
        "global_merge_status": global_merge_status,
        "hook_plan_status": hook_plan_status,
        "failed_chunks": failed_chunks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return global_plan


def _repair_semantic_plan(plan):
    points = plan.get("knowledge_points") or []
    for idx, kp in enumerate(points):
        kp["kp_id"] = kp.get("kp_id") or f"kp_{idx + 1:03d}"
        kp["kp_type"] = kp.get("kp_type") if kp.get("kp_type") in KNOWLEDGE_TYPES else "concept"
        voice_script = _clean_tts_text(kp.get("voice_script") or kp.get("kp_summary") or "")
        if voice_script and voice_script[-1] not in "。！？":
            voice_script += "。"
        kp["voice_script"] = voice_script
        kp["subtitle_lines"] = generate_subtitle_lines_from_voice_script(voice_script)
        kp["subtitle_lines_repaired_from_voice_script"] = True
        deps = kp.get("dependencies") if isinstance(kp.get("dependencies"), dict) else {}
        for key in ("requires", "recommended_before", "recommended_after", "supports", "example_of", "contrasts_with", "follows", "duplicates"):
            deps[key] = [x for x in (deps.get(key) or []) if isinstance(x, str)]
        deps["can_stand_alone"] = bool(deps.get("can_stand_alone", not deps.get("requires")))
        kp["dependencies"] = deps
        for j, frag in enumerate(kp.get("fragments") or []):
            frag["fragment_id"] = frag.get("fragment_id") or f"{kp['kp_id']}_frag_{j + 1:02d}"
            frag["voice_script"] = kp["voice_script"]
            frag["subtitle_lines"] = kp["subtitle_lines"]
            frag["subtitle_lines_repaired_from_voice_script"] = True
            frag["tts_text_source"] = "voice_script"
            frag["subtitle_text_source"] = "subtitle_lines"
    valid_kp_ids = {kp.get("kp_id") for kp in points if kp.get("kp_id")}
    for idx, hook in enumerate(plan.get("big_hooks") or []):
        hook["hook_id"] = hook.get("hook_id") or f"hook_{idx + 1:03d}"
        hook["hook_type"] = hook.get("hook_type") if hook.get("hook_type") in HOOK_TYPES else "short_video"
        hook["recommended_kp_ids"] = [x for x in (hook.get("recommended_kp_ids") or []) if x in valid_kp_ids]
        hook["optional_kp_ids"] = [x for x in (hook.get("optional_kp_ids") or []) if x in valid_kp_ids]
    if not plan.get("big_hooks") or not plan.get("assembly_paths"):
        fallback = build_semantic_hierarchical_fallback([], project_id=plan.get("project_id", ""))
        plan["big_hooks"] = plan.get("big_hooks") or fallback["big_hooks"]
        plan["assembly_paths"] = plan.get("assembly_paths") or fallback["assembly_paths"]
        for path in plan["assembly_paths"]:
            if not path.get("ordered_kp_ids"):
                path["ordered_kp_ids"] = [kp.get("kp_id") for kp in points]


def normalize_semantic_plan(plan, cleaned_segments=None, project_id=""):
    if not isinstance(plan, dict):
        return build_semantic_hierarchical_fallback(cleaned_segments or [], project_id=project_id)
    if isinstance(plan.get("knowledge_modules"), list) and not plan.get("knowledge_points"):
        plan = {"knowledge_points": [
            {"kp_id": f"kp_{idx + 1:03d}", "kp_title": m.get("suggested_title") or m.get("module_title") or f"知识点 {idx + 1}", "kp_type": "concept", "kp_summary": m.get("summary", ""), "source_atom_ids": [], "source_unit_ids": [], "fragments": m.get("fragments") or [], "voice_script": m.get("voice_script") or m.get("merged_cleaned_text") or "", "subtitle_lines": [], "scores": {}, "dependencies": {"requires": [], "can_stand_alone": True}, "selection_reason": m.get("hook", ""), "notices": []}
            for idx, m in enumerate(plan["knowledge_modules"] or [])
        ]}
    semantic = dict(plan)
    semantic["project_id"] = semantic.get("project_id") or project_id
    semantic["slicing_mode"] = "semantic_hierarchical_planning"
    fallback = build_semantic_hierarchical_fallback(cleaned_segments or [], project_id=project_id)
    for key in ("source_summary", "semantic_units", "knowledge_atoms", "knowledge_points", "big_hooks", "assembly_paths", "discarded_content"):
        if not semantic.get(key):
            semantic[key] = fallback.get(key, {} if key == "source_summary" else [])
    _repair_semantic_plan(semantic)
    return semantic


def validate_semantic_plan(plan, leak_check_fn=None):
    leak_check = leak_check_fn or _default_looks_like_prompt_leak
    errors = []
    if not isinstance(plan, dict):
        return False, ["semantic plan must be an object"]
    points = plan.get("knowledge_points")
    if not isinstance(points, list) or not points:
        return False, ["knowledge_points must be a non-empty list"]
    kp_ids = {kp.get("kp_id") for kp in points}
    for kp in points:
        kid = kp.get("kp_id") or "unknown"
        if kp.get("kp_type") not in KNOWLEDGE_TYPES:
            errors.append(f"{kid}: invalid kp_type")
        if not kp.get("voice_script"):
            errors.append(f"{kid}: missing voice_script")
        if not kp.get("subtitle_lines"):
            errors.append(f"{kid}: missing subtitle_lines")
        if leak_check(kp.get("voice_script") or ""):
            errors.append(f"{kid}: voice_script looks like prompt leak")
        duration = calculate_kp_duration(kp)
        if duration > 90:
            errors.append(f"{kid}: duration {duration:.1f}s is too large")
        for required in (kp.get("dependencies") or {}).get("requires") or []:
            if required not in kp_ids:
                errors.append(f"{kid}: requires unknown kp_id {required}")
    if len(plan.get("assembly_paths") or []) < 3:
        errors.append("assembly_paths must contain at least 3 plans")
    return len(errors) == 0, errors


def build_timeline_clips_from_modules(knowledge_modules, mode="knowledge"):
    """Convert knowledge modules into the clip dict shape used by
    generate_videos_from_plan. Each module becomes one clip that carries a
    'fragments' list for the new fragment-aware renderer.
    """
    if isinstance(knowledge_modules, dict) and knowledge_modules.get("knowledge_points"):
        clips = []
        for idx, kp in enumerate(knowledge_modules.get("knowledge_points") or []):
            frags = kp.get("fragments") or []
            if not frags:
                continue
            start = min(float(f.get("start", 0) or 0) for f in frags)
            end = max(float(f.get("end", 0) or 0) for f in frags)
            total_dur = sum(_duration(f.get("start"), f.get("end")) for f in frags)
            enriched = []
            for f in frags:
                frag = dict(f)
                frag["source_text"] = frag.get("source_text") or frag.get("clean_text") or ""
                frag["cleaned_text"] = frag.get("cleaned_text") or frag.get("clean_text") or ""
                frag["voice_script"] = kp.get("voice_script") or frag.get("voice_script") or frag.get("cleaned_text") or ""
                frag["subtitle_lines"] = kp.get("subtitle_lines") or frag.get("subtitle_lines") or split_subtitle_lines(frag["voice_script"])
                frag["tts_text_source"] = "voice_script"
                frag["subtitle_text_source"] = "subtitle_lines"
                frag["kp_id"] = kp.get("kp_id")
                enriched.append(frag)
            clips.append({
                "clip_id": f"clip_{idx + 1:03d}",
                "kp_id": kp.get("kp_id"),
                "title": kp.get("kp_title") or f"知识点 {idx + 1}",
                "hook": kp.get("selection_reason", ""),
                "summary": kp.get("kp_summary", ""),
                "topic_key": kp.get("kp_type", ""),
                "knowledge_points": [kp.get("kp_title", "")],
                "start": start,
                "end": end,
                "duration": round(total_dur, 1),
                "source_text": " ".join(f.get("source_text", "") for f in enriched).strip(),
                "cleaned_clip_text": " ".join(f.get("cleaned_text", "") for f in enriched).strip(),
                "voice_script": kp.get("voice_script", ""),
                "subtitle_lines": kp.get("subtitle_lines") or split_subtitle_lines(kp.get("voice_script", "")),
                "tts_text_source": "voice_script",
                "subtitle_text_source": "subtitle_lines",
                "fragments": enriched,
                "render_mode": "concat_fragments",
            })
        return clips
    clips = []
    for idx, m in enumerate(knowledge_modules or []):
        mid = m.get("module_id") or f"module_{idx + 1:03d}"
        cid = f"clip_{idx + 1:03d}"
        frags = m.get("fragments") or []
        if not frags:
            continue
        start = min(float(f.get("start", 0) or 0) for f in frags)
        end = max(float(f.get("end", 0) or 0) for f in frags)
        total_dur = sum((float(f.get("end", 0) or 0) - float(f.get("start", 0) or 0))
                        for f in frags)
        clips.append({
            "clip_id": cid,
            "module_id": mid,
            "title": m.get("suggested_title") or m.get("module_title") or cid,
            "hook": m.get("hook", ""),
            "summary": m.get("summary", ""),
            "topic_key": m.get("topic_key", ""),
            "knowledge_points": m.get("knowledge_points", []),
            # NOTE: start/end are advisory only. The renderer must use
            # `fragments` to cut, not start/end, otherwise content from the
            # gap between fragments would be included.
            "start": start,
            "end": end,
            "duration": round(total_dur, 1),
            "source_text": m.get("merged_source_text", ""),
            "cleaned_clip_text": m.get("merged_cleaned_text", ""),
            "voice_script": m.get("voice_script", ""),
            "fragments": frags,
            "render_mode": m.get("render_mode", "concat_fragments"),
        })
    return clips


def build_selected_render_plan(selected_module_ids, selected_fragment_ids,
                                knowledge_modules, fragment_order="source_time",
                                voice_style="讲解风格",
                                title="自定义合集"):
    """Build a selected_render_plan.json payload.

    Selection rules:
      - If selected_fragment_ids is non-empty, pick exactly those fragments
        (regardless of selected_module_ids).
      - Else if selected_module_ids is non-empty, pick all fragments from those
        modules.
      - Else: pick everything (caller already warned the user).

    fragment_order:
      - "source_time": ascending by start
      - "user_order": preserve the order of selected_fragment_ids (when given),
                      otherwise the order of selected_module_ids then fragments.
    """
    if isinstance(knowledge_modules, dict) and knowledge_modules.get("knowledge_points"):
        plan = knowledge_modules
        points_by_id = {kp.get("kp_id"): kp for kp in plan.get("knowledge_points") or []}
        kp_ids = list(selected_module_ids or [])
        if not kp_ids:
            kp_ids = [kp.get("kp_id") for kp in plan.get("knowledge_points") or []]
        if fragment_order == "source_time":
            kp_ids = sorted(
                kp_ids,
                key=lambda kid: min(
                    [float(f.get("start", 0) or 0) for f in (points_by_id.get(kid, {}).get("fragments") or [])] or [0]
                ),
            )
        render_units = []
        for idx, kid in enumerate(kp_ids):
            kp = points_by_id.get(kid)
            if not kp:
                continue
            source_fragments = []
            for frag in kp.get("fragments") or []:
                source_fragments.append({
                    "fragment_id": frag.get("fragment_id"),
                    "start": frag.get("start"),
                    "end": frag.get("end"),
                    "clean_text": frag.get("clean_text") or frag.get("cleaned_text"),
                    "reason": frag.get("reason"),
                })
            render_units.append({
                "unit_id": f"render_{idx + 1:03d}",
                "hook_id": None,
                "kp_id": kid,
                "title": kp.get("kp_title"),
                "source_fragments": source_fragments,
                "voice_script": kp.get("voice_script"),
                "subtitle_lines": kp.get("subtitle_lines") or split_subtitle_lines(kp.get("voice_script", "")),
                "tts_text_source": "voice_script",
                "subtitle_text_source": "subtitle_lines",
                "render_mode": "fragment_concat",
            })
        return {
            "selected_hook_ids": [],
            "selected_assembly_path_id": None,
            "selected_kp_ids": [unit["kp_id"] for unit in render_units],
            "voice_style": voice_style,
            "title": title,
            "fragment_order": fragment_order,
            "render_units": render_units,
        }

    modules_by_id = {m.get("module_id"): m for m in (knowledge_modules or [])}

    picks = []  # list of (module_id, fragment_dict)
    if selected_fragment_ids:
        order_index = {fid: i for i, fid in enumerate(selected_fragment_ids)}
        for m in (knowledge_modules or []):
            for f in m.get("fragments") or []:
                if f.get("fragment_id") in order_index:
                    picks.append((m.get("module_id"), f))
    elif selected_module_ids:
        for mid in selected_module_ids:
            m = modules_by_id.get(mid)
            if not m:
                continue
            for f in m.get("fragments") or []:
                picks.append((mid, f))
    else:
        for m in (knowledge_modules or []):
            for f in m.get("fragments") or []:
                picks.append((m.get("module_id"), f))

    if fragment_order == "source_time":
        picks.sort(key=lambda mf: float(mf[1].get("start", 0) or 0))
    elif fragment_order == "user_order" and selected_fragment_ids:
        order_index = {fid: i for i, fid in enumerate(selected_fragment_ids)}
        picks.sort(key=lambda mf: order_index.get(mf[1].get("fragment_id"), 1e9))
    # else: preserve insertion order

    chosen_fragments = []
    chosen_module_ids = []
    seen_modules = set()
    for mid, f in picks:
        if mid and mid not in seen_modules:
            chosen_module_ids.append(mid)
            seen_modules.add(mid)
        chosen_fragments.append(dict(f))

    render_id = "render_" + ("_".join(chosen_module_ids[:2]) or "selected")
    return {
        "render_jobs": [
            {
                "render_id": render_id,
                "title": title,
                "selected_module_ids": chosen_module_ids,
                "selected_fragment_ids": [f.get("fragment_id") for f in chosen_fragments],
                "fragment_order": fragment_order,
                "output_mode": "concat_fragments",
                "voice_style": voice_style,
                "fragments": chosen_fragments,
            }
        ]
    }


def validate_knowledge_modules(modules, leak_check_fn=None):
    """Sanity-check a knowledge_modules list. Returns (ok, errors)."""
    leak_check = leak_check_fn or _default_looks_like_prompt_leak
    errors = []
    if not isinstance(modules, list) or not modules:
        return False, ["knowledge_modules 必须是非空 list"]
    for i, m in enumerate(modules):
        mid = m.get("module_id") or f"#{i}"
        if not m.get("module_id"):
            errors.append(f"{mid}: 缺少 module_id")
        frags = m.get("fragments")
        if not isinstance(frags, list) or not frags:
            errors.append(f"{mid}: fragments 为空")
            continue
        for j, f in enumerate(frags):
            fid = f.get("fragment_id") or f"{mid}.#{j}"
            try:
                start = float(f.get("start", 0) or 0)
                end = float(f.get("end", 0) or 0)
            except Exception:
                errors.append(f"{fid}: start/end 不是数字")
                continue
            if end <= start:
                errors.append(f"{fid}: end ({end}) 必须 > start ({start})")
            if leak_check(f.get("source_text") or ""):
                errors.append(f"{fid}.source_text 疑似提示词回声")
            if leak_check(f.get("cleaned_text") or ""):
                errors.append(f"{fid}.cleaned_text 疑似提示词回声")
        if leak_check(m.get("voice_script") or ""):
            errors.append(f"{mid}.voice_script 疑似提示词回声")
        # Anti "spanning gap" check: if a module spans >5 minutes but the sum
        # of its fragment durations is <50% of that span, the LLM probably
        # made a mistake. We only warn.
        try:
            span_start = min(float(f.get("start", 0) or 0) for f in frags)
            span_end = max(float(f.get("end", 0) or 0) for f in frags)
            span = span_end - span_start
            total = sum(float(f.get("end", 0) or 0) - float(f.get("start", 0) or 0)
                        for f in frags)
            if span > 300 and total < 0.5 * span:
                errors.append(
                    f"{mid}: fragments 跨度 {span:.1f}s 但实际只有 {total:.1f}s 内容 — "
                    "请检查是否被错误地合并为大区间"
                )
        except Exception:
            pass

    return (len(errors) == 0), errors
