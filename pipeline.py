"""Clip-plan + video-generation pipeline (knowledge-module aware).

Two clip-plan modes:
  * "时间线连续切片" → legacy: one LLM call, 2-3 contiguous clips.
  * "知识点模块切片" → new: clean_segments → cluster modules → fragments.

Vocabulary (do NOT alias or rename):
  transcript_full_text  : raw text from STT / manual / file.
  transcript_segments   : list[{start,end,text}] from STT.
  prompt_template       : raw text of a prompts/*.md file.
  llm_prompt            : prompt_template with placeholders substituted.
  llm_raw_response      : raw string returned by LLM.
  llm_result            : parsed dict from LLM (or None on failure).
  cleaned_text          : full-text cleaned for the user — NEVER the prompt.
  cleaned_segments      : per-segment cleaning + topic_tags + knowledge_point.
  knowledge_modules     : clustered modules; each has fragments[].
  clips                 : flat list compatible with the video renderer. A
                          module-based clip carries `fragments` for the
                          fragment-aware renderer.
  fragments             : contiguous-segment slices of a module. Multiple
                          non-contiguous fragments per module is the norm.

NEVER write prompt_template / llm_prompt into any user-facing field.
NEVER cut a clip from min(fragment.start) to max(fragment.end) — that would
include the gap between fragments, defeating the whole point.
"""
import json
import os
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

from services.audio_sample_extractor import extract_voice_samples_from_video
from services.ffmpeg_utils import cut_video, extract_audio, concat_videos, run_cmd
from services.llm_client import call_llm_json, get_llm_env_status
from services.stt_service import transcribe_with_elevenlabs
from services.subtitle_utils import create_subtitles_from_tts_script, split_text_for_subtitles
from services.tts_elevenlabs import generate_tts
from services.video_composer import compose_final_video
from services.voice_clone_elevenlabs import clone_voice_with_elevenlabs
from services.layout_engine import LayoutConfig, probe_video_size, compute_layout, to_dict
from services.voice_style import DEFAULT_VOICE_STYLE, get_voice_style_preset
from services.seedance_client import (
    DEFAULT_SEEDANCE_MODEL,
    SEEDANCE_PRIVACY_SUGGESTED_ACTION,
    SEEDANCE_PRIVACY_USER_MESSAGE,
    is_seedance_privacy_error_code,
)
from services.digital_human import (
    DIGITAL_HUMAN_MODE_LIPSYNC_DISABLED,
    DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
    DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL,
    DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST,
    DIGITAL_HUMAN_MODE_STATIC,
    create_digital_human,
)

from services import knowledge_planner as kp


LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120") or 120)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-leak detector
# ─────────────────────────────────────────────────────────────────────────────
_LEAK_MARKERS = (
    "你是一个录屏讲解短视频策划助手",
    "你是一个短视频内容策划专家",
    "请返回 JSON",
    "{{FULL_TEXT}}",
    "{{SEGMENTS_JSON}}",
    "{{CLEANED_SEGMENTS_JSON}}",
    "{{VOICE_STYLE}}",
    "不要凭空编造",
    "输出 JSON",
    "source_segment_indexes 必须",
    "只返回 JSON，不要返回 markdown",
    "不要返回 markdown，不要返回解释",
)


def looks_like_prompt_leak(text):
    if not text or not isinstance(text, str):
        return False
    return any(m in text for m in _LEAK_MARKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt input compression (long-input safety)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_segments_for_llm(segments, max_segments=80, target_seconds=30):
    if not segments:
        return [], False
    if len(segments) <= max_segments:
        return list(segments), False
    has_real = any(
        (float(s.get("end", 0) or 0) - float(s.get("start", 0) or 0)) > 0
        for s in segments
    )
    bucketed = []
    if has_real:
        bucket_text = []
        bucket_start = float(segments[0].get("start", 0) or 0)
        bucket_end = bucket_start
        for seg in segments:
            bucket_text.append(seg.get("text", "") or "")
            bucket_end = float(seg.get("end", 0) or 0)
            if (bucket_end - bucket_start) >= target_seconds:
                bucketed.append({"start": round(bucket_start, 3),
                                 "end": round(bucket_end, 3),
                                 "text": "".join(bucket_text).strip()})
                bucket_text = []
                bucket_start = bucket_end
        if bucket_text:
            bucketed.append({"start": round(bucket_start, 3),
                             "end": round(bucket_end, 3),
                             "text": "".join(bucket_text).strip()})
    else:
        group = max(1, (len(segments) + max_segments - 1) // max_segments)
        for i in range(0, len(segments), group):
            chunk = segments[i:i + group]
            bucketed.append({"start": 0.0, "end": 0.0,
                             "text": "".join((s.get("text", "") or "") for s in chunk).strip()})
    if len(bucketed) > max_segments:
        step = max(1, len(bucketed) // max_segments)
        bucketed = bucketed[::step][:max_segments]
    return bucketed, True


def prepare_full_text_for_llm(full_text, max_chars=12000):
    if not full_text:
        return "", False
    if len(full_text) <= max_chars:
        return full_text, False
    return full_text[:max_chars], True


# ─────────────────────────────────────────────────────────────────────────────
# Strict retry prompt (legacy timeline mode only)
# ─────────────────────────────────────────────────────────────────────────────
_STRICT_RETRY_PROMPT = """你必须只返回一个 JSON 对象。
不要返回 markdown。不要返回解释。不要返回代码块。
不要复述我的任务。不要输出任何 JSON 之外的文字。

JSON 结构必须是：
{
  "cleaned_text": "string",
  "clips": [
    {
      "clip_id": "clip_001",
      "start": 0,
      "end": 90,
      "title": "string",
      "hook": "string",
      "summary": "string",
      "source_segment_indexes": [0, 1],
      "source_text": "string",
      "cleaned_clip_text": "string",
      "voice_script": "string",
      "action_tags": []
    }
  ]
}

下面是视频转写文本：
{{FULL_TEXT}}

下面是带时间戳 segments：
{{SEGMENTS_JSON}}
"""


def build_strict_retry_prompt(transcript_full_text, segments_json):
    return (
        _STRICT_RETRY_PROMPT
        .replace("{{FULL_TEXT}}", transcript_full_text or "")
        .replace("{{SEGMENTS_JSON}}", segments_json or "[]")
    )


def _classify_llm_response(call_result):
    if not isinstance(call_result, dict):
        return False, "call_result 不是 dict", None
    if call_result.get("error"):
        return False, str(call_result.get("error")), None
    parsed = call_result.get("result")
    if not isinstance(parsed, dict):
        return False, "未提取到 JSON object", None
    clips = parsed.get("clips")
    if not isinstance(clips, list) or not clips:
        return False, "clips 字段缺失或为空", None
    cleaned = parsed.get("cleaned_text") or ""
    if cleaned and looks_like_prompt_leak(cleaned):
        return False, "cleaned_text 疑似提示词回声", None
    for c in clips:
        cid = c.get("clip_id", "?")
        for field in ("source_text", "cleaned_clip_text", "voice_script"):
            v = c.get(field) or ""
            if v and looks_like_prompt_leak(v):
                return False, f"{cid}.{field} 疑似提示词回声", None
    return True, None, parsed


def _segments_have_real_timestamps(segments):
    return bool(segments) and any(
        (float(s.get("end", 0) or 0) - float(s.get("start", 0) or 0)) > 0
        for s in segments
    )


def build_fallback_clips_from_segments(transcript_segments, transcript_full_text):
    """Legacy timeline-mode deterministic fallback."""
    clips = []
    if _segments_have_real_timestamps(transcript_segments):
        bucket_start_idx = 0
        bucket_start_time = float(transcript_segments[0].get("start", 0) or 0)
        bucket_parts = []
        last_end = bucket_start_time
        for i, seg in enumerate(transcript_segments):
            bucket_parts.append(seg.get("text", "") or "")
            last_end = float(seg.get("end", 0) or 0)
            is_last = (i == len(transcript_segments) - 1)
            if (last_end - bucket_start_time) >= 90 or is_last:
                src = "".join(bucket_parts).strip()
                if src and last_end > bucket_start_time:
                    cn = len(clips) + 1
                    clips.append({
                        "clip_id": f"clip_{cn:03d}",
                        "start": float(bucket_start_time),
                        "end": float(last_end),
                        "duration": float(last_end - bucket_start_time),
                        "title": f"片段 {cn}",
                        "hook": src[:30],
                        "summary": src[:80],
                        "source_text": src,
                        "cleaned_clip_text": src,
                        "voice_script": src,
                        "source_segment_indexes": list(range(bucket_start_idx, i + 1)),
                        "estimated_timestamps": False,
                    })
                bucket_start_idx = i + 1
                if not is_last:
                    bucket_start_time = float(transcript_segments[i + 1].get("start", last_end) or last_end)
                bucket_parts = []
        return clips
    text = (transcript_full_text or "").strip()
    if not text:
        return []
    n = 3 if len(text) > 400 else 2
    chunk_size = max(1, len(text) // n)
    cursor_time = 0.0
    for i in range(n):
        s_i = i * chunk_size
        e_i = (i + 1) * chunk_size if i < n - 1 else len(text)
        piece = text[s_i:e_i].strip()
        if not piece:
            continue
        est_dur = max(1.0, len(piece) / 4.0)
        cn = len(clips) + 1
        clips.append({
            "clip_id": f"clip_{cn:03d}",
            "start": float(cursor_time),
            "end": float(cursor_time + est_dur),
            "duration": float(est_dur),
            "title": f"片段 {cn}",
            "hook": piece[:30],
            "summary": piece[:80],
            "source_text": piece,
            "cleaned_clip_text": piece,
            "voice_script": piece,
            "source_segment_indexes": [],
            "estimated_timestamps": True,
        })
        cursor_time += est_dur
    return clips


# ─────────────────────────────────────────────────────────────────────────────
# Plan-result validator
# ─────────────────────────────────────────────────────────────────────────────
def validate_plan_result(plan_result):
    full_text = (plan_result.get("transcript_full_text")
                 or plan_result.get("full_text") or "")
    if not full_text or not full_text.strip():
        raise ValueError("transcript_full_text 为空")
    if looks_like_prompt_leak(full_text):
        raise ValueError(
            "检测到提示词模板被当作结果文本（transcript_full_text），已中止切片计划生成。"
        )
    cleaned = plan_result.get("cleaned_text", "") or ""
    if looks_like_prompt_leak(cleaned):
        raise ValueError("检测到提示词模板被当作结果文本（cleaned_text），已中止切片计划生成。")
    clips = plan_result.get("clips", [])
    if not isinstance(clips, list) or len(clips) < 1:
        raise ValueError(f"clips 必须是非空 list（当前类型 {type(clips).__name__}）")
    for i, c in enumerate(clips):
        cid = c.get("clip_id", f"#{i}")
        src = c.get("source_text", "") or ""
        if not src.strip():
            raise ValueError(f"{cid} 的 source_text 为空")
        if looks_like_prompt_leak(src):
            raise ValueError(f"检测到提示词模板被当作 {cid}.source_text。")
        if looks_like_prompt_leak(c.get("cleaned_clip_text", "") or ""):
            raise ValueError(f"检测到提示词模板被当作 {cid}.cleaned_clip_text。")
        if looks_like_prompt_leak(c.get("voice_script", "") or ""):
            raise ValueError(f"检测到提示词模板被当作 {cid}.voice_script。")
        try:
            start = float(c.get("start", 0) or 0)
            end = float(c.get("end", 0) or 0)
        except Exception:
            raise ValueError(f"{cid} 的 start/end 不是数字")
        is_estimated = bool(c.get("estimated_timestamps", False))
        has_fragments = bool(c.get("fragments"))
        # Knowledge-module clips: start/end is advisory; fragments dictate cuts.
        if end <= start and not is_estimated and not has_fragments:
            raise ValueError(f"{cid}: end ({end}) 必须 > start ({start})")
        if has_fragments:
            for j, f in enumerate(c["fragments"]):
                fid = f.get("fragment_id", f"{cid}.#{j}")
                try:
                    fs = float(f.get("start", 0) or 0)
                    fe = float(f.get("end", 0) or 0)
                except Exception:
                    raise ValueError(f"{fid} 的 start/end 不是数字")
                f_est = bool(f.get("estimated_timestamps", False))
                if fe <= fs and not f_est:
                    raise ValueError(f"{fid}: end ({fe}) 必须 > start ({fs})")
                if looks_like_prompt_leak(f.get("source_text") or ""):
                    raise ValueError(f"检测到提示词模板被当作 {fid}.source_text。")
                if looks_like_prompt_leak(f.get("cleaned_text") or ""):
                    raise ValueError(f"检测到提示词模板被当作 {fid}.cleaned_text。")


def _save_upload(file_bytes, dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(file_bytes)
    return dest


def _merge_source_text(segments, indexes):
    if not segments or not indexes:
        return ""
    parts = []
    for i in indexes:
        if 0 <= i < len(segments):
            parts.append(segments[i].get("text", ""))
    return "".join(parts)


def _write_debug_files(debug_dir, **fields):
    """Persist a per-run debug snapshot. The filename extension is inferred
    from the value type / key name:
      * dict / list values  → .json
      * keys containing 'prompt' → .md
      * everything else      → .txt
    The caller passes raw key names (no dots); we append the extension here.
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, value in fields.items():
        if isinstance(value, (dict, list)):
            ext = ".json"
        elif "prompt" in name:
            ext = ".md"
        else:
            ext = ".txt"
        path = debug_dir / f"{name}{ext}"
        try:
            if isinstance(value, (dict, list)):
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                path.write_text(value if value is not None else "", encoding="utf-8")
        except Exception as e:
            print(f"[debug] failed to write {path}: {e}")


def _emit_progress(progress_callback, stage, message, **extra):
    if callable(progress_callback):
        try:
            progress_callback({"stage": stage, "message": message, **extra})
        except Exception:
            pass


def _write_planning_heartbeat(debug_dir, start_time, current_stage,
                              current_chunk=None, total_chunks=None,
                              last_success=None, last_error=None):
    payload = {
        "current_stage": current_stage,
        "current_chunk": current_chunk,
        "total_chunks": total_chunks,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "last_success": last_success,
        "last_error": last_error,
    }
    try:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        (Path(debug_dir) / "planning_heartbeat.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return payload


def _call_llm_json_with_timeout(prompt, stage="llm"):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(call_llm_json, prompt, stage=stage)
        try:
            return future.result(timeout=LLM_REQUEST_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            future.cancel()
            diagnostic = {
                "success": False,
                "stage": stage,
                "error_type": "request_timeout",
                "error_message": f"LLM request timeout after {LLM_REQUEST_TIMEOUT_SECONDS}s",
                "selected_key_source": get_llm_env_status().get("selected_key_source"),
                "model": get_llm_env_status().get("llm_model"),
            }
            return {
                "result": None,
                "raw_response": "",
                "error": diagnostic["error_message"],
                "model": diagnostic["model"],
                "success": False,
                "diagnostic": diagnostic,
                **diagnostic,
            }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _llm_error_for_report(*diagnostics):
    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict) and diagnostic.get("error_type"):
            return {
                "llm_error_type": diagnostic.get("error_type"),
                "llm_error_message": diagnostic.get("error_message"),
            }
    return {"llm_error_type": None, "llm_error_message": None}


def _llm_selected_key_source(*diagnostics):
    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict) and diagnostic.get("selected_key_source"):
            return diagnostic.get("selected_key_source")
    return get_llm_env_status().get("selected_key_source")


def _estimate_digital_human_target_duration(clips, selected_fragment_ids=None):
    durations = []
    for clip in clips or []:
        fragments = clip.get("fragments") or []
        if fragments:
            selected = [
                frag for frag in fragments
                if selected_fragment_ids is None or frag.get("fragment_id") in selected_fragment_ids
            ]
            total = 0.0
            longest = 0.0
            for frag in selected:
                try:
                    duration = float(frag.get("end", 0) or 0) - float(frag.get("start", 0) or 0)
                except Exception:
                    duration = 0.0
                if duration > 0:
                    total += duration
                    longest = max(longest, duration)
            if total > 0:
                durations.append(max(total, longest))
            continue
        try:
            duration = float(clip.get("duration") or 0)
        except Exception:
            duration = 0.0
        if duration <= 0:
            try:
                duration = float(clip.get("end", 0) or 0) - float(clip.get("start", 0) or 0)
            except Exception:
                duration = 0.0
        if duration > 0:
            durations.append(duration)
    return max(durations) if durations else 5.0


def _prepare_digital_human(provider_key, avatar_path, voice_style, clips, project_dir,
                           result_warnings, force_regenerate_seedance_avatar=False,
                           quality_mode="fast", avatar_video_mode="preview_loop",
                           target_duration=None,
                           fallback_experimental_i2v_to_fast=False):
    first_clip = (clips or [{}])[0] if clips else {}
    first_script = (
        first_clip.get("voice_script")
        or first_clip.get("cleaned_clip_text")
        or first_clip.get("source_text")
        or ""
    )
    if provider_key == DIGITAL_HUMAN_MODE_LIPSYNC_DISABLED:
        result_warnings.append(
            "精准口型同步数字人当前未启用：doubao-seedance-1-0-pro-fast-251015 "
            "不支持 image_audio_to_talking_head，已回退静态头像。"
        )
    try:
        result = create_digital_human(
            provider_key,
            avatar_image_path=avatar_path,
            style=voice_style,
            voice_script=first_script,
            clip_id=first_clip.get("clip_id", "digital_human_cache"),
            output_dir=project_dir,
            force_regenerate=force_regenerate_seedance_avatar,
            quality_mode=quality_mode,
            target_duration=target_duration,
            avatar_video_mode=avatar_video_mode,
        )
        report = result.to_report()
        report["requested_provider"] = provider_key
        report["actual_provider"] = provider_key if result.success else "static"
        report["fallback_to_static_avatar"] = bool(
            provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC and not result.success
        )
        report["fallback_from_2_0_i2v_to_fast"] = False
        seedance_debug = report.get("debug") or {}
        experimental_privacy_failure = (
            provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL
            and not result.success
            and _seedance_privacy_guard_triggered(seedance_debug)
        )
        if experimental_privacy_failure and fallback_experimental_i2v_to_fast:
            fallback_result = create_digital_human(
                DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST,
                avatar_image_path=avatar_path,
                style=voice_style,
                voice_script=first_script,
                clip_id=first_clip.get("clip_id", "digital_human_cache"),
                output_dir=project_dir,
                force_regenerate=force_regenerate_seedance_avatar,
                quality_mode="fast",
                target_duration=target_duration,
                avatar_video_mode=avatar_video_mode,
            )
            fallback_report = fallback_result.to_report()
            fallback_debug = fallback_report.get("debug") or {}
            merged_debug = dict(fallback_debug)
            merged_debug.update({
                "seedance_model_requested": seedance_debug.get("seedance_model_requested"),
                "seedance_requested_model": (
                    seedance_debug.get("seedance_requested_model")
                    or seedance_debug.get("seedance_model_requested")
                ),
                "requested_model_candidates": seedance_debug.get("requested_model_candidates"),
                "seedance_model_candidates": seedance_debug.get("seedance_model_candidates"),
                "seedance_error_type": seedance_debug.get("seedance_error_type"),
                "seedance_error_code": seedance_debug.get("seedance_error_code"),
                "seedance_error_message": seedance_debug.get("seedance_error_message"),
                "suggested_action": seedance_debug.get("suggested_action"),
                "privacy_guard_triggered": True,
                "privacy_guard_message": seedance_debug.get("privacy_guard_message") or SEEDANCE_PRIVACY_USER_MESSAGE,
                "fallback_from_2_0_i2v_to_fast": True,
                "fallback_reason": SEEDANCE_PRIVACY_USER_MESSAGE,
                "experimental_seedance_error": seedance_debug.get("seedance_error"),
                "experimental_seedance_attempt_errors": seedance_debug.get("seedance_attempt_errors"),
                "fallback_fast_seedance_attempt_errors": fallback_debug.get("seedance_attempt_errors"),
                "seedance_attempt_errors": seedance_debug.get("seedance_attempt_errors"),
            })
            fallback_report["debug"] = merged_debug
            fallback_report["requested_provider"] = provider_key
            fallback_report["actual_provider"] = (
                DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST if fallback_result.success else "static"
            )
            fallback_report["fallback_to_static_avatar"] = False
            fallback_report["fallback_from_2_0_i2v_to_fast"] = True
            if fallback_result.success:
                result_warnings.append(
                    "Seedance 2.0 image_to_video 已触发真人图片隐私风控，已按勾选项自动回退到 Seedance fast 保留头像模式。"
                )
            else:
                result_warnings.append(f"Seedance fast 保留头像模式自动回退也失败: {fallback_result.error}")
            return fallback_report
        if experimental_privacy_failure:
            result_warnings.append(SEEDANCE_PRIVACY_USER_MESSAGE)
        if provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC and not result.success:
            result_warnings.append(f"Seedance 动态讲解人像生成失败，已回退静态头像: {result.error}")
        return report
    except Exception as e:
        provider = "seedance_image_to_video" if provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC else "static"
        if provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC:
            result_warnings.append(f"Seedance 动态讲解人像初始化失败，已回退静态头像: {e}")
        return {
            "talking_head_video_path": None,
            "success": provider_key != DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
            "error": str(e) if provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC else None,
            "debug": {"fallback": "static_avatar"},
            "provider": provider,
            "requested_provider": provider_key,
            "actual_provider": "static",
            "fallback_to_static_avatar": provider_key == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
            "is_lip_sync": False,
            "uses_audio_driven_lipsync": False,
            "quality_mode": quality_mode,
            "avatar_video_mode": avatar_video_mode,
            "target_duration": target_duration,
            "fallback_from_2_0_i2v_to_fast": False,
        }


def _talking_head_file_info(talking_head_video_path):
    path = Path(talking_head_video_path) if talking_head_video_path else None
    exists = bool(path and path.exists())
    size = path.stat().st_size if exists else 0
    duration = None
    if exists and size > 0:
        try:
            import subprocess
            from services.ffmpeg_utils import _cmd
            result = subprocess.run(
                _cmd([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ]),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0:
                duration = float((result.stdout or "0").strip() or 0)
            else:
                duration = None
        except Exception:
            duration = None
    return exists, size, duration


def _media_duration(path):
    p = Path(path) if path else None
    if not p or not p.exists() or p.stat().st_size <= 0:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(p),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode == 0:
            return float((result.stdout or "0").strip() or 0)
    except Exception:
        return None
    return None


def _norm_render_text(text):
    return "".join(str(text or "").split())


def _normalise_audio_mode(audio_mode):
    text = str(audio_mode or "").strip()
    if text in {
        "keep_original_audio",
        "existing_elevenlabs_voice_id",
        "upload_voice_sample_clone",
        "clone_from_source_video_audio",
    }:
        return text
    if text.startswith("保留"):
        return "keep_original_audio"
    if text.startswith("使用已有"):
        return "existing_elevenlabs_voice_id"
    if text.startswith("上传声音样本"):
        return "upload_voice_sample_clone"
    if text.startswith("从原视频原声"):
        return "clone_from_source_video_audio"
    return text or "keep_original_audio"


def _audio_mode_label(audio_mode_key):
    labels = {
        "keep_original_audio": "保留原视频原声",
        "existing_elevenlabs_voice_id": "使用已有 ElevenLabs voice_id 替换原声",
        "upload_voice_sample_clone": "上传声音样本，通过 ElevenLabs API 云端克隆后替换原声",
        "clone_from_source_video_audio": "从原视频原声自动提取样本并克隆后替换原声",
    }
    return labels.get(audio_mode_key, audio_mode_key)


def _resolve_tts_text(render_item):
    candidates = [
        ("voice_script", render_item.get("voice_script")),
        ("voice_script", render_item.get("knowledge_point_voice_script")),
        ("clean_text", render_item.get("clean_text")),
        ("clean_text", render_item.get("cleaned_text")),
        ("clean_text", render_item.get("cleaned_clip_text")),
        ("clean_text", render_item.get("source_text")),
        ("raw_text", render_item.get("raw_text")),
        ("raw_text", render_item.get("text")),
    ]
    for source, value in candidates:
        text = str(value or "").strip()
        if text:
            return text, source, source == "raw_text"
    return "", "empty", False


def _resolve_subtitle_lines(render_item, tts_text):
    lines = render_item.get("subtitle_lines") or []
    lines = [str(line or "").strip() for line in lines if str(line or "").strip()]
    if lines and _norm_render_text("".join(lines)) == _norm_render_text(tts_text):
        return lines, "subtitle_lines", True
    if tts_text:
        return split_text_for_subtitles(tts_text), "voice_script", True
    clean_text = (
        render_item.get("clean_text")
        or render_item.get("cleaned_text")
        or render_item.get("cleaned_clip_text")
        or ""
    )
    if clean_text:
        return split_text_for_subtitles(clean_text), "clean_text", _norm_render_text(clean_text) == _norm_render_text(tts_text)
    raw_text = render_item.get("raw_text") or render_item.get("source_text") or render_item.get("text") or ""
    if raw_text:
        return split_text_for_subtitles(raw_text), "raw_text", _norm_render_text(raw_text) == _norm_render_text(tts_text)
    return [], "empty", False


def _adjust_video_duration_for_tts(input_video, source_duration, tts_duration, output_video):
    """Return (video_path, adjustment) after matching picture duration to TTS."""
    try:
        source_duration = float(source_duration or 0)
        tts_duration = float(tts_duration or 0)
    except Exception:
        return input_video, "none"
    if source_duration <= 0 or tts_duration <= 0 or abs(source_duration - tts_duration) <= 0.3:
        return input_video, "none"
    output_video = str(output_video)
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    if tts_duration < source_duration:
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-t", f"{tts_duration:.3f}",
            "-c:v", "libx264",
            "-an",
            "-movflags", "+faststart",
            output_video,
        ])
        return output_video, "trim_video"
    extra = max(tts_duration - source_duration, 0.0)
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", f"tpad=stop_mode=clone:stop_duration={extra:.3f}",
        "-t", f"{tts_duration:.3f}",
        "-c:v", "libx264",
        "-an",
        "-movflags", "+faststart",
        output_video,
    ])
    return output_video, "freeze_tail"


def _source_video_metadata_path(project_dir):
    return Path(project_dir) / "source_video.json"


def _read_source_video_metadata(project_dir):
    path = _source_video_metadata_path(project_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_source_video_metadata(project_dir, metadata):
    _source_video_metadata_path(project_dir).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_source_video_path(project_dir, video_name, source_video_path=None):
    if source_video_path and Path(source_video_path).exists():
        return str(Path(source_video_path))
    metadata = _read_source_video_metadata(project_dir)
    candidate = metadata.get("source_video_path")
    if candidate and Path(candidate).exists():
        return str(Path(candidate))
    return str(Path(project_dir) / "uploads" / video_name)


def _digital_human_clip_report(digital_human_report, talking_head_video_path=None,
                               final_composer_used_talking_head=False,
                               compose_debug=None):
    report = digital_human_report or {}
    compose_debug = compose_debug or {}
    seedance_debug = report.get("debug") or {}
    seedance_error = _select_seedance_error(seedance_debug)
    privacy_guard_triggered = _seedance_privacy_guard_triggered(seedance_debug)
    seedance_error_message = (
        seedance_debug.get("seedance_error_message")
        or seedance_error.get("error_message")
        or seedance_error.get("message")
    )
    fallback_reason = seedance_debug.get("fallback_reason")
    if privacy_guard_triggered and not fallback_reason:
        fallback_reason = SEEDANCE_PRIVACY_USER_MESSAGE
    exists, size, duration = _talking_head_file_info(talking_head_video_path)
    requested = report.get("requested_provider") or DIGITAL_HUMAN_MODE_STATIC
    fallback = bool(report.get("fallback_to_static_avatar"))
    actual = report.get("actual_provider") or report.get("provider") or "static"
    if fallback or not talking_head_video_path:
        actual = "static" if requested != DIGITAL_HUMAN_MODE_STATIC else "static"
    return {
        "digital_human_requested_provider": requested,
        "digital_human_actual_provider": actual,
        "digital_human_provider": report.get("provider") or "static",
        "digital_human_success": bool(report.get("success")),
        "talking_head_video_path": talking_head_video_path,
        "talking_head_video_exists": exists,
        "talking_head_video_size": size,
        "talking_head_video_duration": duration,
        "fallback_to_static_avatar": fallback,
        "fallback_from_2_0_i2v_to_fast": bool(
            report.get("fallback_from_2_0_i2v_to_fast")
            or seedance_debug.get("fallback_from_2_0_i2v_to_fast")
        ),
        "is_lip_sync": False,
        "uses_audio_driven_lipsync": False,
        "digital_human_error": report.get("error"),
        "seedance_debug": seedance_debug,
        "seedance_cache_hit": seedance_debug.get("seedance_cache_hit", seedance_debug.get("cache_hit")),
        "cache_hit": seedance_debug.get("cache_hit"),
        "cache_key": seedance_debug.get("cache_key"),
        "seedance_prompt": seedance_debug.get("seedance_prompt", seedance_debug.get("prompt")),
        "seedance_negative_prompt": seedance_debug.get("seedance_negative_prompt"),
        "seedance_model_requested": seedance_debug.get("seedance_model_requested"),
        "seedance_model_used": seedance_debug.get("seedance_model_used"),
        "seedance_generation_type": seedance_debug.get("seedance_generation_type"),
        "identity_preserved_from_avatar": bool(seedance_debug.get("identity_preserved_from_avatar")),
        "reason": seedance_debug.get("reason"),
        "seedance_requested_model": seedance_debug.get("seedance_requested_model") or seedance_debug.get("seedance_model_requested"),
        "seedance_used_model": seedance_debug.get("seedance_used_model") or seedance_debug.get("seedance_model_used"),
        "seedance_quality_mode": seedance_debug.get("quality_mode"),
        "requested_model_candidates": seedance_debug.get("requested_model_candidates"),
        "seedance_model_candidates": seedance_debug.get("seedance_model_candidates"),
        "selected_model": seedance_debug.get("seedance_model_used"),
        "fallback_model": seedance_debug.get("seedance_model_used") if seedance_debug.get("fallback_reason") else None,
        "fallback_reason": fallback_reason,
        "seedance_fallback_reason": fallback_reason,
        "privacy_guard_triggered": privacy_guard_triggered,
        "seedance_error_code": seedance_debug.get("seedance_error_code") or seedance_error.get("error_code"),
        "seedance_error_message": seedance_error_message,
        "suggested_action": seedance_debug.get("suggested_action"),
        "generation_mode": seedance_debug.get("generation_mode"),
        "target_duration": seedance_debug.get("target_duration"),
        "generated_duration": seedance_debug.get("generated_duration"),
        "chunk_count": seedance_debug.get("chunk_count"),
        "seedance_create_task_status_code": _seedance_error_value(seedance_error, "create_task", "status_code"),
        "seedance_create_task_error_code": _seedance_error_value(seedance_error, "create_task", "error_code"),
        "seedance_create_task_error_message": _seedance_error_value(seedance_error, "create_task", "error_message"),
        "seedance_create_task_response_text": _seedance_error_value(seedance_error, "create_task", "response_text"),
        "seedance_poll_status_code": _seedance_error_value(seedance_error, "poll_task", "status_code"),
        "seedance_poll_error_code": _seedance_error_value(seedance_error, "poll_task", "error_code"),
        "seedance_poll_response_text": _seedance_error_value(seedance_error, "poll_task", "response_text"),
        "seedance_request_payload_sanitized": seedance_error.get("request_payload_sanitized"),
        "seedance_error_detail": seedance_error,
        "final_composer_used_talking_head": bool(final_composer_used_talking_head),
        "final_composer_used_window_style": compose_debug.get("final_composer_used_window_style"),
        "used_full_length_avatar_video": bool(compose_debug.get("used_full_length_avatar_video")),
        "used_loop_avatar_video": bool(compose_debug.get("used_loop_avatar_video")),
        "used_static_avatar_fallback": bool(compose_debug.get("used_static_avatar_fallback")),
    }


def _seedance_error_value(seedance_error, stage, key):
    if not isinstance(seedance_error, dict):
        return None
    if seedance_error.get("stage") == stage:
        return seedance_error.get(key)
    return None


def _seedance_error_candidates(seedance_debug):
    candidates = []
    direct_error = seedance_debug.get("seedance_error")
    if isinstance(direct_error, dict):
        candidates.append(direct_error)
    for attempt in seedance_debug.get("seedance_attempt_errors") or []:
        if isinstance(attempt, dict):
            error = attempt.get("error")
            if isinstance(error, dict):
                candidates.append(error)
    return candidates


def _select_seedance_error(seedance_debug):
    candidates = _seedance_error_candidates(seedance_debug)
    for error in candidates:
        if is_seedance_privacy_error_code(error.get("error_code")):
            return error
    return candidates[0] if candidates else {}


def _seedance_privacy_guard_triggered(seedance_debug):
    if seedance_debug.get("privacy_guard_triggered"):
        return True
    return any(
        is_seedance_privacy_error_code(error.get("error_code"))
        for error in _seedance_error_candidates(seedance_debug)
    )


def _seedance_privacy_report_fields(digital_human_report):
    seedance_debug = (digital_human_report or {}).get("debug") or {}
    seedance_error = _select_seedance_error(seedance_debug)
    triggered = _seedance_privacy_guard_triggered(seedance_debug)
    error_code = seedance_debug.get("seedance_error_code") or seedance_error.get("error_code")
    error_message = (
        seedance_debug.get("seedance_error_message")
        or seedance_error.get("error_message")
        or seedance_error.get("message")
    )
    return {
        "seedance_error_type": "privacy_person_image" if triggered else seedance_debug.get("seedance_error_type"),
        "seedance_error_code": error_code,
        "seedance_error_message": error_message,
        "suggested_action": SEEDANCE_PRIVACY_SUGGESTED_ACTION if triggered else seedance_debug.get("suggested_action"),
        "privacy_guard_triggered": bool(triggered),
    }


def _read_prompt_file(prompts_dir, filename):
    p = Path(prompts_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f"prompt 文件不存在: {p}")
    return p.read_text(encoding="utf-8")


def get_transcript_from_source(input_mode, video_path,
                               manual_text=None, transcript_file_text=None,
                               work_dir=None):
    if input_mode == "auto":
        audio_base = Path(work_dir) if work_dir else Path(video_path).parent
        audio_base.mkdir(parents=True, exist_ok=True)
        audio_path = str(audio_base / "_extracted_audio.wav")
        extract_audio(video_path, audio_path)
        result = transcribe_with_elevenlabs(audio_path)
        Path(audio_path).unlink(missing_ok=True)
        return {"source": "elevenlabs_stt", "language": "zh",
                "full_text": result["full_text"], "segments": result["segments"]}
    if input_mode == "manual":
        text = (manual_text or "").strip()
        return {"source": "manual_text", "language": "zh",
                "full_text": text,
                "segments": [{"start": 0.0, "end": 0.0, "text": text}] if text else []}
    if input_mode == "file":
        text = (transcript_file_text or "").strip()
        return {"source": "uploaded_transcript_file", "language": "zh",
                "full_text": text,
                "segments": [{"start": 0.0, "end": 0.0, "text": text}] if text else []}
    raise ValueError(f"未知的 input_mode: {input_mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge-module pipeline (mode = "知识点模块切片")
# ─────────────────────────────────────────────────────────────────────────────
def _generate_knowledge_modules(
    transcript_full_text,
    transcript_segments,
    voice_style,
    project_dir,
    debug_dir,
    prompts_dir,
    progress_callback=None,
):
    """Run the two-stage knowledge-module planner. Always returns a dict:

      {
        "cleaned_text": str,
        "cleaned_segments": list[dict],
        "knowledge_modules": list[dict],
        "warnings": list[str],
        "errors": list[str],
        "llm_clean_status": "success" | "fallback" | "failed",
        "llm_plan_status":  "success" | "fallback" | "failed",
        "llm_clean_error":  str | None,
        "llm_plan_error":   str | None,
      }
    """
    warnings = []
    errors = []
    res = {
        "cleaned_text": "",
        "cleaned_segments": [],
        "knowledge_modules": [],
        "warnings": warnings,
        "errors": errors,
        "llm_clean_status": "unknown",
        "llm_plan_status": "unknown",
        "llm_clean_error": None,
        "llm_plan_error": None,
        "llm_clean_diagnostic": None,
        "llm_plan_diagnostic": None,
        "llm_selected_key_source": get_llm_env_status().get("selected_key_source"),
        "llm_model": get_llm_env_status().get("llm_model"),
        "llm_plan_repaired": False,
        "llm_plan_repair_reason": "none",
        "llm_plan_validation_debug": None,
        "used_chunked_planning": False,
        "chunk_count": 0,
        "successful_chunk_count": 0,
        "failed_chunk_count": 0,
        "failed_chunks": [],
        "chunk_fallback_used": False,
        "cached_chunk_count": 0,
        "chunked_planning_reason": None,
        "global_merge_status": None,
        "hook_plan_status": None,
    }

    project_dir = Path(project_dir)
    modules_dir = project_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    planning_start = time.time()

    # Stage 1: clean + tag segments
    try:
        clean_template = _read_prompt_file(prompts_dir, "clean_segments.md")
    except Exception as e:
        warnings.append(f"读取 clean_segments.md 失败，使用 deterministic 清洗：{e}")
        clean_template = None

    cleaned_text = ""
    cleaned_segments = []

    if clean_template:
        _emit_progress(progress_callback, "cleaning_text", "正在清洗文本")
        _write_planning_heartbeat(debug_dir, planning_start, "cleaning_text")
        # Compress before sending (same as legacy mode).
        compressed_segments, _ = prepare_segments_for_llm(transcript_segments)
        capped_full_text, _ = prepare_full_text_for_llm(transcript_full_text)
        clean_call = kp.clean_transcript_segments_with_llm(
            compressed_segments, capped_full_text, _call_llm_json_with_timeout, clean_template,
        )
        _write_debug_files(
            debug_dir,
            debug_llm_clean_prompt=clean_call.get("llm_prompt") or "",
            debug_llm_clean_raw_response=clean_call.get("raw_response") or "",
        )
        res["llm_clean_diagnostic"] = clean_call.get("diagnostic")
        res["llm_selected_key_source"] = _llm_selected_key_source(clean_call.get("diagnostic"))
        res["llm_model"] = clean_call.get("model") or res["llm_model"]
        if isinstance(clean_call.get("result"), dict):
            parsed = clean_call["result"]
            cleaned_text = parsed.get("cleaned_text") or ""
            cs_candidate = parsed.get("cleaned_segments") or []
            # Leak guard on the LLM-returned cleaned content.
            leak = False
            if looks_like_prompt_leak(cleaned_text):
                leak = True
            for s in cs_candidate:
                if looks_like_prompt_leak(s.get("clean_text") or ""):
                    leak = True; break
            if leak:
                warnings.append("clean_segments LLM 响应疑似提示词回声，使用 deterministic 兜底。")
                cleaned_text = ""
                cs_candidate = []
                res["llm_clean_diagnostic"] = {
                    **(res.get("llm_clean_diagnostic") or {}),
                    "success": False,
                    "stage": "llm_clean",
                    "error_type": "json_schema_error",
                    "error_message": "clean_segments LLM response looked like prompt echo",
                }
            cleaned_segments = cs_candidate
            res["llm_clean_status"] = "success" if cleaned_segments else "fallback"
            if not cleaned_segments:
                res["llm_clean_error"] = "cleaned_segments 为空"
        else:
            res["llm_clean_status"] = "fallback"
            res["llm_clean_error"] = clean_call.get("error") or "未提取到 JSON"
    else:
        res["llm_clean_status"] = "fallback"
        res["llm_clean_error"] = "prompt 模板缺失"

    # Deterministic fallback for stage 1
    if not cleaned_segments:
        _emit_progress(progress_callback, "cleaning_text", "正在使用规则兜底清洗文本")
        cleaned_segments = kp.build_fallback_cleaned_segments(transcript_segments)
        if not cleaned_text:
            cleaned_text = "".join((s.get("clean_text") or "") for s in cleaned_segments
                                   if not s.get("is_filler"))
        warnings.append("clean_segments 使用 deterministic 兜底（已基于规则去口语化）。")

    if looks_like_prompt_leak(cleaned_text):
        errors.append("检测到 cleaned_text 含提示词残留，已中止知识模块规划。")
        return res

    res["cleaned_text"] = cleaned_text
    res["cleaned_segments"] = cleaned_segments

    # Persist cleaned_segments.json + cleaned_text.txt (re-saved by caller too).
    (project_dir / "cleaned_segments.json").write_text(
        json.dumps(cleaned_segments, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Persist cleaning_report.json so downstream tools can audit how aggressive
    # the filler/repetition removal was for this project.
    try:
        cleaning_report = kp.build_cleaning_report(transcript_segments, cleaned_segments)
        cleaning_report["llm_clean_status"] = res.get("llm_clean_status")
        cleaning_report["llm_clean_error"] = res.get("llm_clean_error")
        cleaning_report["planning_input_text"] = "cleaned_segments"
        cleaning_report["raw_text_used_for_planning"] = False
        (project_dir / "cleaning_report.json").write_text(
            json.dumps(cleaning_report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        res["cleaning_report"] = cleaning_report
    except Exception as e:
        warnings.append(f"写入 cleaning_report.json 失败：{e}")

    # Stage 2: cluster into modules
    try:
        plan_template = _read_prompt_file(prompts_dir, "plan_knowledge_modules.md")
    except Exception as e:
        warnings.append(f"读取 plan_knowledge_modules.md 失败，使用 deterministic 聚类：{e}")
        plan_template = None

    knowledge_modules = None
    chunk_reasons = kp.chunked_planning_reasons(cleaned_segments, transcript_full_text)
    if chunk_reasons:
        try:
            knowledge_modules = kp.plan_knowledge_modules_chunked(
                cleaned_segments,
                transcript_full_text,
                voice_style,
                project_dir,
                _call_llm_json_with_timeout,
                prompts_dir=prompts_dir,
                progress_callback=progress_callback,
            )
            res["used_chunked_planning"] = True
            res["chunk_count"] = knowledge_modules.get("chunk_count", 0)
            res["successful_chunk_count"] = knowledge_modules.get("successful_chunk_count", 0)
            res["failed_chunk_count"] = knowledge_modules.get("failed_chunk_count", 0)
            res["failed_chunks"] = knowledge_modules.get("failed_chunks") or []
            res["chunk_fallback_used"] = bool(knowledge_modules.get("chunk_fallback_used"))
            res["cached_chunk_count"] = knowledge_modules.get("cached_chunk_count", 0)
            res["chunked_planning_reason"] = knowledge_modules.get("chunked_planning_reason") or " / ".join(chunk_reasons)
            res["global_merge_status"] = knowledge_modules.get("global_merge_status")
            res["hook_plan_status"] = knowledge_modules.get("hook_plan_status")
            res["llm_plan_repaired"] = bool(knowledge_modules.get("llm_plan_repaired"))
            res["llm_plan_repair_reason"] = knowledge_modules.get("llm_plan_repair_reason") or "none"
            res["llm_plan_status"] = "chunked_success"
            warnings.append(
                f"长文本模式已启用：系统已将内容分成 {res['chunk_count']} 个语义块进行分析。"
            )
        except Exception as e:
            warnings.append(f"长文本分块规划失败，回到单次规划或 deterministic 兜底：{e}")
            res["llm_plan_error"] = str(e)

    if knowledge_modules is None and plan_template:
        _emit_progress(progress_callback, "single_planning", "正在生成知识点计划")
        _write_planning_heartbeat(debug_dir, planning_start, "single_planning")
        plan_call = kp.plan_knowledge_modules_with_llm(
            cleaned_segments, _call_llm_json_with_timeout, plan_template, voice_style=voice_style,
        )
        _write_debug_files(
            debug_dir,
            debug_llm_plan_prompt=plan_call.get("llm_prompt") or "",
            debug_llm_plan_raw_response=plan_call.get("raw_response") or "",
        )
        res["llm_plan_diagnostic"] = plan_call.get("diagnostic")
        res["llm_selected_key_source"] = _llm_selected_key_source(
            res.get("llm_clean_diagnostic"),
            plan_call.get("diagnostic"),
        )
        res["llm_model"] = plan_call.get("model") or res["llm_model"]
        if isinstance(plan_call.get("result"), dict):
            cand = kp.normalize_semantic_plan(
                plan_call["result"],
                cleaned_segments=cleaned_segments,
                project_id=project_dir.name,
            )
            cand = kp.repair_long_knowledge_points(cand, max_kp_duration=90)
            validation_debug = (cand.get("long_kp_repair") or {}).get("debug") or {}
            res["llm_plan_validation_debug"] = validation_debug
            _write_debug_files(debug_dir, debug_llm_plan_validation=validation_debug)
            ok, errs = kp.validate_semantic_plan(cand, leak_check_fn=looks_like_prompt_leak)
            if ok:
                knowledge_modules = cand
                repair_reason = cand.get("llm_plan_repair_reason") or "none"
                res["llm_plan_repaired"] = bool(cand.get("llm_plan_repaired"))
                res["llm_plan_repair_reason"] = repair_reason
                res["llm_plan_status"] = "repaired_success" if res["llm_plan_repaired"] else "success"
                if res["llm_plan_repaired"]:
                    warnings.append("LLM 已返回知识点计划，系统已自动修复过长知识点或按 fragment 总时长重新计算。")
            else:
                long_errors = [e for e in errs if "duration" in e and "too large" in e]
                if long_errors:
                    warnings.append("LLM 已返回知识点计划，但部分知识点过长；自动拆分后仍未通过校验，才使用 deterministic 兜底。")
                else:
                    warnings.append("knowledge_modules 校验失败，使用 deterministic 聚类。详情见 errors。")
                res["llm_plan_status"] = "fallback"
                res["llm_plan_error"] = "; ".join(errs[:5])
                res["llm_plan_diagnostic"] = {
                    **(res.get("llm_plan_diagnostic") or {}),
                    "success": False,
                    "stage": "llm_plan",
                    "error_type": "json_schema_error",
                    "error_message": res["llm_plan_error"],
                }
        else:
            res["llm_plan_status"] = "fallback"
            res["llm_plan_error"] = plan_call.get("error") or "未提取到 JSON"
    elif knowledge_modules is None:
        res["llm_plan_status"] = "fallback"
        res["llm_plan_error"] = "prompt 模板缺失"

    if (
        knowledge_modules is None
        and plan_template
        and not chunk_reasons
        and kp.should_use_chunked_planning(
            cleaned_segments,
            transcript_full_text,
            previous_error=res.get("llm_plan_error"),
        )
    ):
        try:
            knowledge_modules = kp.plan_knowledge_modules_chunked(
                cleaned_segments,
                transcript_full_text,
                voice_style,
                project_dir,
                _call_llm_json_with_timeout,
                prompts_dir=prompts_dir,
                progress_callback=progress_callback,
            )
            res["used_chunked_planning"] = True
            res["chunk_count"] = knowledge_modules.get("chunk_count", 0)
            res["successful_chunk_count"] = knowledge_modules.get("successful_chunk_count", 0)
            res["failed_chunk_count"] = knowledge_modules.get("failed_chunk_count", 0)
            res["failed_chunks"] = knowledge_modules.get("failed_chunks") or []
            res["chunk_fallback_used"] = bool(knowledge_modules.get("chunk_fallback_used"))
            res["cached_chunk_count"] = knowledge_modules.get("cached_chunk_count", 0)
            res["chunked_planning_reason"] = knowledge_modules.get("chunked_planning_reason") or "previous plan error unstable"
            res["global_merge_status"] = knowledge_modules.get("global_merge_status")
            res["hook_plan_status"] = knowledge_modules.get("hook_plan_status")
            res["llm_plan_repaired"] = bool(knowledge_modules.get("llm_plan_repaired"))
            res["llm_plan_repair_reason"] = knowledge_modules.get("llm_plan_repair_reason") or "none"
            res["llm_plan_status"] = "chunked_success"
            warnings.append("单次知识点规划不稳定，已自动切换为长文本分块规划。")
        except Exception as e:
            warnings.append(f"自动切换长文本分块规划失败，使用 deterministic 兜底：{e}")

    if not knowledge_modules:
        knowledge_modules = kp.build_semantic_hierarchical_fallback(
            cleaned_segments,
            project_id=project_dir.name,
        )
        warnings.append("knowledge_modules 使用 deterministic 兜底（按 topic_tags / 时间桶聚类）。")

    if not knowledge_modules:
        errors.append("无法生成 knowledge_modules（cleaned_segments 也是空的）。")
        return res

    if isinstance(knowledge_modules, dict):
        _emit_progress(progress_callback, "repairing_plan", "正在修复计划")
        _write_planning_heartbeat(debug_dir, planning_start, "repairing_plan")
        knowledge_modules = kp.repair_semantic_plan_for_render(
            knowledge_modules,
            source_segments=cleaned_segments,
        )
        knowledge_modules["llm_plan_repaired"] = bool(res.get("llm_plan_repaired") or knowledge_modules.get("llm_plan_repaired"))
        repair_reason = res.get("llm_plan_repair_reason")
        if not repair_reason or repair_reason == "none":
            repair_reason = knowledge_modules.get("llm_plan_repair_reason") or "none"
        knowledge_modules["llm_plan_repair_reason"] = (
            repair_reason
        )
        knowledge_modules["used_deterministic_fallback"] = bool(
            res.get("llm_clean_status") != "success"
            or res.get("llm_plan_status") not in {"success", "repaired_success", "chunked_success"}
        )
        for key in (
            "used_chunked_planning",
            "chunk_count",
            "successful_chunk_count",
            "failed_chunk_count",
            "failed_chunks",
            "chunk_fallback_used",
            "cached_chunk_count",
            "chunked_planning_reason",
            "global_merge_status",
            "hook_plan_status",
        ):
            if res.get(key) is not None:
                knowledge_modules[key] = res.get(key)

    res["knowledge_modules"] = knowledge_modules
    _emit_progress(progress_callback, "writing_results", "正在写入结果")
    _write_planning_heartbeat(debug_dir, planning_start, "writing_results")
    (project_dir / "knowledge_modules.json").write_text(
        json.dumps(knowledge_modules, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Per-knowledge-point markdown summary for human review.
    review_items = (
        knowledge_modules.get("knowledge_points", [])
        if isinstance(knowledge_modules, dict)
        else knowledge_modules
    )
    for m in review_items:
        mid = m.get("kp_id") or m.get("module_id", "module_unknown")
        lines = [
            f"# {m.get('kp_title') or m.get('module_title', mid)}",
            "",
            f"- type: `{m.get('kp_type') or m.get('topic_key', '')}`",
            f"- id: `{mid}`",
            f"- estimated_duration: {sum((float(f.get('end', 0) or 0) - float(f.get('start', 0) or 0)) for f in (m.get('fragments') or [])):.1f}s",
            "",
            "## Hook",
            m.get("selection_reason") or m.get("hook", ""),
            "",
            "## Summary",
            m.get("kp_summary") or m.get("summary", ""),
            "",
            "## Knowledge Points",
        ]
        for kpoint in (m.get("knowledge_points") or [m.get("kp_title", "")]):
            lines.append(f"- {kpoint}")
        lines += ["", "## Fragments"]
        for f in (m.get("fragments") or []):
            lines.append(
                f"- `{f.get('fragment_id', '')}` "
                f"[{f.get('start', 0)} → {f.get('end', 0)}]  "
                f"reason: {f.get('reason', '')}"
            )
            lines.append(f"  - cleaned_text: {(f.get('clean_text') or f.get('cleaned_text') or '')[:160]}")
        lines += ["", "## Voice Script", m.get("voice_script", "") or ""]
        try:
            (modules_dir / f"{mid}.md").write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"[modules.md] write failed for {mid}: {e}")

    llm_error = _llm_error_for_report(
        res.get("llm_clean_diagnostic"),
        res.get("llm_plan_diagnostic"),
    )
    if llm_error.get("llm_error_type"):
        _write_debug_files(
            debug_dir,
            debug_llm_error={
                **llm_error,
                "llm_clean_status": res.get("llm_clean_status"),
                "llm_plan_status": res.get("llm_plan_status"),
                "llm_selected_key_source": res.get("llm_selected_key_source"),
                "llm_model": res.get("llm_model"),
                "clean": res.get("llm_clean_diagnostic"),
                "plan": res.get("llm_plan_diagnostic"),
            },
        )

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 entrypoint: generate clip plan
# ─────────────────────────────────────────────────────────────────────────────
def generate_clips_plan(
    video_bytes,
    video_name,
    avatar_bytes,
    avatar_name,
    input_mode,
    prompt_template,
    manual_text=None,
    transcript_file_text=None,
    clip_plan_mode="知识点模块切片",
    voice_style=DEFAULT_VOICE_STYLE,
    prompts_dir="prompts",
    source_video_path=None,
    video_input_mode="upload",
    source_video_size_bytes=None,
    progress_callback=None,
):
    """Build the clip plan.

    clip_plan_mode == "知识点模块切片" (default):
        Stage 1: clean_segments → cleaned_segments.json
        Stage 2: plan_knowledge_modules → knowledge_modules.json
        Stage 3: modules → clips.json (each clip carries `fragments`)
    clip_plan_mode == "时间线连续切片":
        Legacy single-shot LLM with strict retry, produces 2-3 contiguous clips.
    """
    project_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    project_dir = Path("outputs") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = project_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "project_id": project_id,
        "project_dir": str(project_dir),
        "transcript_source": input_mode,
        "clip_plan_mode": clip_plan_mode,
        "voice_style": voice_style,
        "warnings": [],
        "errors": [],
        "clips": [],
        "knowledge_modules": [],
        "cleaned_segments": [],
        "cleaned_text": "",
        "full_text": "",
        "transcript_full_text": "",
        "segments": [],
        "llm_status": "unknown",
        "llm_first_error": None,
        "llm_first_response_preview": "",
        "llm_retry_error": None,
        "llm_retry_response_preview": "",
        "llm_fallback_reason": None,
        "llm_model": None,
        "llm_error_type": None,
        "llm_error_message": None,
        "llm_selected_key_source": get_llm_env_status().get("selected_key_source"),
        "used_deterministic_fallback": False,
        "llm_clean_status": None,
        "llm_plan_status": None,
        "used_fallback_clips": False,
        "used_chunked_planning": False,
        "chunk_count": 0,
        "successful_chunk_count": 0,
        "failed_chunk_count": 0,
        "failed_chunks": [],
        "chunk_fallback_used": False,
        "cached_chunk_count": 0,
        "chunked_planning_reason": None,
        "global_merge_status": None,
        "hook_plan_status": None,
        "video_input_mode": video_input_mode,
        "source_video_path": None,
        "source_video_size_bytes": source_video_size_bytes,
    }

    # 1. Persist uploaded avatar and either persist a small uploaded video or
    # keep an external local/NAS video path without copying large source files.
    uploads_dir = project_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    if source_video_path:
        video_path = str(Path(source_video_path))
        try:
            source_video_size_bytes = Path(video_path).stat().st_size
        except Exception:
            pass
    else:
        video_path = str(uploads_dir / video_name)
        _save_upload(video_bytes, video_path)
        try:
            source_video_size_bytes = Path(video_path).stat().st_size
        except Exception:
            pass
    avatar_path = str(project_dir / "uploads" / avatar_name)
    _save_upload(avatar_bytes, avatar_path)
    source_video_meta = {
        "video_input_mode": video_input_mode or ("upload" if not source_video_path else "local_path"),
        "source_video_path": video_path,
        "source_video_name": video_name,
        "source_video_size_bytes": source_video_size_bytes,
        "copied_to_project_uploads": not bool(source_video_path),
    }
    _write_source_video_metadata(project_dir, source_video_meta)
    result.update({
        "video_input_mode": source_video_meta["video_input_mode"],
        "source_video_path": video_path,
        "source_video_size_bytes": source_video_size_bytes,
    })

    # 2. Transcript.
    try:
        transcript_result = get_transcript_from_source(
            input_mode, video_path, manual_text, transcript_file_text,
            work_dir=uploads_dir,
        )
    except Exception as e:
        result["errors"].append(f"转写失败: {e}")
        _write_debug_files(debug_dir,
            debug_transcript_full_text="",
            debug_pipeline_warnings={"warnings": result["warnings"], "errors": result["errors"]},
        )
        return result

    transcript_full_text = (transcript_result.get("full_text") or "").strip()
    transcript_segments = transcript_result.get("segments") or []

    if not transcript_full_text:
        result["errors"].append("转写文本为空")
        _write_debug_files(debug_dir,
            debug_transcript_full_text="",
            debug_transcript_segments=transcript_segments,
            debug_pipeline_warnings={"warnings": result["warnings"], "errors": result["errors"]},
        )
        return result

    if looks_like_prompt_leak(transcript_full_text):
        result["errors"].append("检测到提示词模板被当作 transcript_full_text。")
        return result

    (project_dir / "transcript.json").write_text(
        json.dumps(transcript_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (project_dir / "transcript.txt").write_text(transcript_full_text, encoding="utf-8")

    result["full_text"] = transcript_full_text
    result["transcript_full_text"] = transcript_full_text
    result["segments"] = transcript_segments

    if clip_plan_mode == "知识点模块切片":
        return _finish_knowledge_module_plan(
            result, project_dir, debug_dir,
            transcript_full_text, transcript_segments,
            voice_style, prompts_dir,
            progress_callback=progress_callback,
        )
    else:
        return _finish_timeline_plan(
            result, project_dir, debug_dir,
            transcript_full_text, transcript_segments,
            prompt_template,
        )


def _finish_knowledge_module_plan(result, project_dir, debug_dir,
                                   transcript_full_text, transcript_segments,
                                   voice_style, prompts_dir,
                                   progress_callback=None):
    km_out = _generate_knowledge_modules(
        transcript_full_text, transcript_segments,
        voice_style, project_dir, debug_dir, prompts_dir,
        progress_callback=progress_callback,
    )
    result["warnings"].extend(km_out.get("warnings", []))
    result["errors"].extend(km_out.get("errors", []))
    result["cleaned_text"] = km_out["cleaned_text"]
    result["cleaned_segments"] = km_out["cleaned_segments"]
    result["knowledge_modules"] = km_out["knowledge_modules"]
    result["cleaning_report"] = km_out.get("cleaning_report") or {}
    result["llm_clean_status"] = km_out.get("llm_clean_status")
    result["llm_plan_status"] = km_out.get("llm_plan_status")
    result["llm_model"] = km_out.get("llm_model")
    result["llm_selected_key_source"] = km_out.get("llm_selected_key_source")
    result["llm_plan_repaired"] = bool(km_out.get("llm_plan_repaired"))
    result["llm_plan_repair_reason"] = km_out.get("llm_plan_repair_reason") or "none"
    result["llm_plan_validation_debug"] = km_out.get("llm_plan_validation_debug")
    for key in (
        "used_chunked_planning",
        "chunk_count",
        "successful_chunk_count",
        "failed_chunk_count",
        "chunked_planning_reason",
        "global_merge_status",
        "hook_plan_status",
    ):
        result[key] = km_out.get(key)
    llm_error = _llm_error_for_report(
        km_out.get("llm_clean_diagnostic"),
        km_out.get("llm_plan_diagnostic"),
    )
    result.update(llm_error)
    result["llm_clean_diagnostic"] = km_out.get("llm_clean_diagnostic")
    result["llm_plan_diagnostic"] = km_out.get("llm_plan_diagnostic")
    plan_ok_statuses = {"success", "repaired_success", "chunked_success"}
    result["used_deterministic_fallback"] = bool(
        km_out.get("llm_clean_status") != "success"
        or km_out.get("llm_plan_status") not in plan_ok_statuses
    )

    # Roll up status for UI.
    if km_out.get("llm_clean_status") == "success" and km_out.get("llm_plan_status") == "success":
        result["llm_status"] = "ok"
    elif km_out.get("llm_clean_status") == "success" and km_out.get("llm_plan_status") == "chunked_success":
        result["llm_status"] = "chunked_success"
    elif km_out.get("llm_clean_status") == "success" and km_out.get("llm_plan_status") == "repaired_success":
        result["llm_status"] = "repaired_success"
    elif km_out.get("llm_clean_status") == "fallback" and km_out.get("llm_plan_status") in plan_ok_statuses:
        result["llm_status"] = "retry_success"
    else:
        result["llm_status"] = "fallback"
        result["llm_fallback_reason"] = (
            f"clean: {km_out.get('llm_clean_error')}; plan: {km_out.get('llm_plan_error')}"
        )
        result["used_fallback_clips"] = True

    if result["errors"]:
        _write_debug_files(debug_dir,
            debug_transcript_full_text=transcript_full_text,
            debug_transcript_segments=transcript_segments,
            debug_pipeline_warnings={
                "warnings": result["warnings"], "errors": result["errors"],
                "llm_clean_status": km_out.get("llm_clean_status"),
                "llm_plan_status": km_out.get("llm_plan_status"),
            },
        )
        return result

    # Convert modules → clips (one clip per module, with fragments).
    clips = kp.build_timeline_clips_from_modules(result["knowledge_modules"])
    result["clips"] = clips

    # Save clips.json with both fields.
    (project_dir / "cleaned_text.txt").write_text(result["cleaned_text"], encoding="utf-8")
    (project_dir / "clips.json").write_text(
        json.dumps({
            "cleaned_text": result["cleaned_text"],
            "clips": clips,
            "knowledge_modules": result["knowledge_modules"],
            "clip_plan_mode": "知识点模块切片",
            "voice_style": voice_style,
            "llm_plan_repaired": result.get("llm_plan_repaired"),
            "llm_plan_repair_reason": result.get("llm_plan_repair_reason"),
            "used_deterministic_fallback": result.get("used_deterministic_fallback"),
            "used_chunked_planning": result.get("used_chunked_planning"),
            "chunk_count": result.get("chunk_count"),
            "successful_chunk_count": result.get("successful_chunk_count"),
            "failed_chunk_count": result.get("failed_chunk_count"),
            "failed_chunks": result.get("failed_chunks"),
            "chunk_fallback_used": result.get("chunk_fallback_used"),
            "cached_chunk_count": result.get("cached_chunk_count"),
            "chunked_planning_reason": result.get("chunked_planning_reason"),
            "global_merge_status": result.get("global_merge_status"),
            "hook_plan_status": result.get("hook_plan_status"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Per-clip text files (compatible with timeline mode UI/editing).
    for clip in clips:
        cid = clip.get("clip_id", "unknown")
        (project_dir / f"{cid}_source_text.txt").write_text(
            clip.get("source_text", "") or "", encoding="utf-8")
        (project_dir / f"{cid}_cleaned_text.txt").write_text(
            clip.get("cleaned_clip_text", "") or "", encoding="utf-8")
        (project_dir / f"{cid}_voice_script.txt").write_text(
            clip.get("voice_script", "") or "", encoding="utf-8")

    # Default selected_render_plan now follows the first video_topic so the
    # initial render aligns with the topic the user sees in the UI. We still
    # capture big_hook metadata for backwards compatibility.
    default_topic = {}
    default_hook = {}
    default_kp_ids = []
    if isinstance(result["knowledge_modules"], dict):
        topics = result["knowledge_modules"].get("video_topics") or []
        if topics:
            default_topic = topics[0]
            default_kp_ids = default_topic.get("recommended_kp_ids") or []
        if not default_kp_ids:
            default_hook = (result["knowledge_modules"].get("big_hooks") or [{}])[0]
            default_kp_ids = default_hook.get("recommended_kp_ids") or []
    initial_plan = kp.build_selected_render_plan(
        default_kp_ids, [], result["knowledge_modules"],
        fragment_order="user_order" if default_kp_ids else "source_time", voice_style=voice_style,
        title=(default_topic.get("topic_title") or default_hook.get("hook_title") or "默认短视频方案"),
    )
    initial_plan["render_output_mode"] = "single_complete_video"
    initial_plan["selected_topic_id"] = default_topic.get("topic_id")
    initial_plan["selected_hook_id"] = default_topic.get("source_hook_id") or default_hook.get("hook_id")
    initial_plan["selected_hook_ids"] = [initial_plan["selected_hook_id"]] if initial_plan.get("selected_hook_id") else []
    initial_plan["selected_kp_ids"] = [unit.get("kp_id") for unit in initial_plan.get("render_units") or []]
    initial_plan["final_video_title"] = default_topic.get("topic_title") or default_hook.get("hook_title")
    initial_plan["final_video_opening_hook"] = default_topic.get("topic_hook") or default_hook.get("opening_hook")
    initial_plan["final_video_structure"] = "按所选知识点顺序讲解"
    initial_plan["video_type"] = default_topic.get("video_type") or "short_video"
    initial_plan["visible_output_count"] = 1
    (project_dir / "selected_render_plan.json").write_text(
        json.dumps(initial_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # Persist video_topics.json and render_jobs.json so future renderings can
    # reuse this plan without re-running STT or LLM planning.
    km_for_files = result["knowledge_modules"] if isinstance(result["knowledge_modules"], dict) else {}
    topics_payload = {
        "project_id": result["project_id"],
        "voice_style": voice_style,
        "source_summary": km_for_files.get("source_summary") or {},
        "video_topics_count": len(km_for_files.get("video_topics") or []),
        "video_topics": km_for_files.get("video_topics") or [],
        "video_topics_targets": km_for_files.get("video_topics_targets") or {},
        "user_visible_kp_ids": km_for_files.get("user_visible_kp_ids") or [],
        "advanced_kp_ids": km_for_files.get("advanced_kp_ids") or [],
        "kp_cap_summary": km_for_files.get("kp_cap_summary") or {},
    }
    (project_dir / "video_topics.json").write_text(
        json.dumps(topics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs_payload = kp.build_render_jobs_from_topics(
        km_for_files, voice_style=voice_style, project_id=result["project_id"],
    )
    jobs_payload["project_id"] = result["project_id"]
    jobs_payload["plan_reusable"] = bool(jobs_payload.get("render_jobs"))
    jobs_payload["video_topics_count"] = topics_payload["video_topics_count"]
    (project_dir / "render_jobs.json").write_text(
        json.dumps(jobs_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["video_topics_count"] = topics_payload["video_topics_count"]
    result["render_jobs_count"] = jobs_payload["render_jobs_count"]
    result["plan_reusable"] = jobs_payload["plan_reusable"]
    result["video_topics_path"] = str(project_dir / "video_topics.json")
    result["render_jobs_path"] = str(project_dir / "render_jobs.json")
    # Data-flow provenance: report.json downstream relies on these flags.
    result["planning_input_text"] = "cleaned_segments"
    result["raw_text_used_for_planning"] = False
    result["filler_words_removed"] = bool(
        (result.get("cleaning_report") or {}).get("filler_words_removed", True)
    )

    # Validate.
    try:
        validate_plan_result(result)
    except ValueError as ve:
        result["errors"].append(str(ve))

    _write_debug_files(debug_dir,
        debug_transcript_full_text=transcript_full_text,
        debug_transcript_segments=transcript_segments,
        debug_pipeline_warnings={
            "warnings": result["warnings"], "errors": result["errors"],
            "llm_status": result["llm_status"],
            "llm_clean_status": result["llm_clean_status"],
            "llm_plan_status": result["llm_plan_status"],
            "llm_fallback_reason": result["llm_fallback_reason"],
            "used_chunked_planning": result.get("used_chunked_planning"),
            "chunk_count": result.get("chunk_count"),
            "successful_chunk_count": result.get("successful_chunk_count"),
            "failed_chunk_count": result.get("failed_chunk_count"),
            "failed_chunks": result.get("failed_chunks"),
            "chunk_fallback_used": result.get("chunk_fallback_used"),
            "cached_chunk_count": result.get("cached_chunk_count"),
            "global_merge_status": result.get("global_merge_status"),
            "hook_plan_status": result.get("hook_plan_status"),
        },
    )
    return result


def _finish_timeline_plan(result, project_dir, debug_dir,
                          transcript_full_text, transcript_segments,
                          prompt_template):
    """Legacy single-shot LLM + strict retry, produces contiguous clips."""
    compressed_segments, segs_compressed = prepare_segments_for_llm(transcript_segments)
    if segs_compressed:
        result["warnings"].append(
            f"segments 数量过多（{len(transcript_segments)}），已合并为 "
            f"{len(compressed_segments)} 段后送 LLM。")
    capped_full_text, ft_truncated = prepare_full_text_for_llm(transcript_full_text)
    if ft_truncated:
        result["warnings"].append(
            f"full_text 过长（{len(transcript_full_text)} 字），已截断到 "
            f"{len(capped_full_text)} 字送 LLM。")
    segments_json = json.dumps(compressed_segments, ensure_ascii=False, indent=2)

    llm_prompt = (
        prompt_template
        .replace("{{FULL_TEXT}}", capped_full_text)
        .replace("{{SEGMENTS_JSON}}", segments_json)
    )
    if "{{FULL_TEXT}}" in llm_prompt or "{{SEGMENTS_JSON}}" in llm_prompt:
        result["errors"].append("LLM prompt 占位符未正确替换，已中止切片计划生成。")
        return result

    first_call = call_llm_json(llm_prompt, stage="llm_plan")
    llm_raw_response = first_call.get("raw_response") or ""
    result["llm_model"] = first_call.get("model")
    result["llm_selected_key_source"] = _llm_selected_key_source(first_call.get("diagnostic"))
    result.update(_llm_error_for_report(first_call.get("diagnostic")))
    first_ok, first_reason, first_parsed = _classify_llm_response(first_call)
    result["llm_first_error"] = None if first_ok else (first_reason or "未知错误")
    result["llm_first_response_preview"] = llm_raw_response[:500] if llm_raw_response else ""

    parsed = first_parsed
    retry_prompt = None
    retry_raw_response = ""
    used_retry = False

    if not first_ok:
        used_retry = True
        retry_prompt = build_strict_retry_prompt(capped_full_text, segments_json)
        retry_call = call_llm_json(retry_prompt, stage="llm_plan")
        retry_raw_response = retry_call.get("raw_response") or ""
        result["llm_selected_key_source"] = _llm_selected_key_source(
            first_call.get("diagnostic"),
            retry_call.get("diagnostic"),
        )
        result.update(_llm_error_for_report(retry_call.get("diagnostic"), first_call.get("diagnostic")))
        retry_ok, retry_reason, retry_parsed = _classify_llm_response(retry_call)
        result["llm_retry_error"] = None if retry_ok else (retry_reason or "未知错误")
        result["llm_retry_response_preview"] = retry_raw_response[:500] if retry_raw_response else ""
        if retry_ok:
            parsed = retry_parsed
            result["llm_status"] = "retry_success"

    used_fallback = False
    if parsed is not None and not used_retry:
        result["llm_status"] = "ok"
        cleaned_text = parsed.get("cleaned_text") or transcript_full_text
        clips = parsed.get("clips") or []
    elif parsed is not None and used_retry and result["llm_status"] == "retry_success":
        cleaned_text = parsed.get("cleaned_text") or transcript_full_text
        clips = parsed.get("clips") or []
    else:
        used_fallback = True
        result["llm_status"] = "fallback"
        result["llm_fallback_reason"] = (
            f"first: {result['llm_first_error']}"
            + (f" | retry: {result['llm_retry_error']}" if used_retry else "")
        )
        result["warnings"].append(
            f"LLM 两次均未给出可用 JSON，已使用 transcript 兜底切片。{result['llm_fallback_reason']}")
        cleaned_text = transcript_full_text
        clips = build_fallback_clips_from_segments(transcript_segments, transcript_full_text)

    for clip in clips:
        clip.setdefault("source_segment_indexes", [])
        if not clip.get("source_text"):
            merged = _merge_source_text(transcript_segments, clip.get("source_segment_indexes", []))
            clip["source_text"] = merged or ""
        if "cleaned_clip_text" not in clip or clip["cleaned_clip_text"] is None:
            clip["cleaned_clip_text"] = clip.get("source_text", "") or ""
        if "voice_script" not in clip or clip["voice_script"] is None:
            clip["voice_script"] = clip.get("cleaned_clip_text") or clip.get("source_text", "") or ""
        clip.pop("script", None)
        try:
            clip["duration"] = float(clip.get("end", 0) or 0) - float(clip.get("start", 0) or 0)
        except Exception:
            clip["duration"] = 0

    result["cleaned_text"] = cleaned_text
    result["clips"] = clips
    result["used_fallback_clips"] = used_fallback
    result["used_deterministic_fallback"] = used_fallback
    result["llm_clean_status"] = "success"
    result["llm_plan_status"] = "fallback" if used_fallback else "success"

    try:
        validate_plan_result(result)
    except ValueError as ve:
        result["errors"].append(str(ve))
        _write_debug_files(debug_dir,
            debug_transcript_full_text=transcript_full_text,
            debug_transcript_segments=transcript_segments,
            debug_llm_prompt=llm_prompt,
            debug_llm_retry_prompt=retry_prompt or "",
            debug_llm_raw_response=llm_raw_response,
            debug_llm_retry_raw_response=retry_raw_response,
            debug_llm_result=parsed if isinstance(parsed, dict) else {"_": None},
            debug_pipeline_warnings={"warnings": result["warnings"], "errors": result["errors"]},
        )
        return result

    (project_dir / "clips.json").write_text(
        json.dumps({"cleaned_text": cleaned_text, "clips": clips,
                    "clip_plan_mode": "时间线连续切片"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (project_dir / "cleaned_text.txt").write_text(cleaned_text, encoding="utf-8")
    for clip in clips:
        cid = clip.get("clip_id", "unknown")
        (project_dir / f"{cid}_source_text.txt").write_text(clip.get("source_text", "") or "", encoding="utf-8")
        (project_dir / f"{cid}_cleaned_text.txt").write_text(clip.get("cleaned_clip_text", "") or "", encoding="utf-8")
        (project_dir / f"{cid}_voice_script.txt").write_text(clip.get("voice_script", "") or "", encoding="utf-8")

    _write_debug_files(debug_dir,
        debug_transcript_full_text=transcript_full_text,
        debug_transcript_segments=transcript_segments,
        debug_llm_prompt=llm_prompt,
        debug_llm_retry_prompt=retry_prompt or "",
        debug_llm_raw_response=llm_raw_response,
        debug_llm_retry_raw_response=retry_raw_response,
        debug_llm_result=parsed if isinstance(parsed, dict) else {"_": None},
        debug_pipeline_warnings={
            "warnings": result["warnings"], "errors": result["errors"],
            "llm_status": result["llm_status"],
            "llm_first_error": result["llm_first_error"],
            "llm_retry_error": result["llm_retry_error"],
            "llm_fallback_reason": result["llm_fallback_reason"],
        },
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fragment-aware renderer
# ─────────────────────────────────────────────────────────────────────────────
def _render_clip_with_fragments(clip, video_path, avatar_path, layout_config,
                                 effective_voice_id, voice_style,
                                 raw_dir, srt_dir, voice_dir, final_dir,
                                 result_warnings, result_errors,
                                 digital_human_report=None,
                                 talking_head_video_path=None,
                                 talking_head_full_video_path=None,
                                 digital_human_window_style="direct"):
    """Render a clip that has multiple non-contiguous fragments.

    For each fragment we:
      1. cut the original video to a per-fragment mp4 (raw_dir/<cid>_<fid>_raw.mp4)
      2. make a per-fragment SRT
      3. (optional) generate TTS for the fragment's text
      4. compose fragment_composed.mp4 (avatar + subtitle + maybe voice)

    Then concat all composed fragments → final_dir/<cid>_final.mp4.
    Per-fragment TTS keeps the audio aligned with the picture and avoids the
    "one giant monolithic TTS" feel.
    """
    clip_id = clip.get("clip_id", "clip_unknown")
    fragments = clip.get("fragments") or []
    composed_paths = []
    fragment_reports = []
    any_tts_success = False

    for j, frag in enumerate(fragments):
        fid = frag.get("fragment_id", f"frag_{j + 1:04d}")
        try:
            fs = float(frag.get("start", 0) or 0)
            fe = float(frag.get("end", 0) or 0)
        except Exception:
            result_errors.append(f"{clip_id}/{fid}: start/end 非数字，跳过")
            continue
        if fe <= fs:
            result_errors.append(f"{clip_id}/{fid}: end<=start，跳过")
            continue

        # 1. cut the source video for this fragment
        raw_frag = str(raw_dir / f"{clip_id}_{fid}_raw.mp4")
        try:
            cut_video(video_path, fs, fe, raw_frag)
        except Exception as e:
            result_errors.append(f"{clip_id}/{fid}: 裁剪失败: {e}")
            continue

        # 2. Resolve semantic TTS/subtitle text. In semantic mode TTS must use
        # voice_script and subtitles must come from the same cleaned narration.
        render_item = dict(clip)
        render_item.update({k: v for k, v in frag.items() if v not in (None, "", [])})
        frag_text = (frag.get("cleaned_text") or frag.get("clean_text") or frag.get("source_text") or "").strip()
        tts_text, tts_text_source, raw_text_used_for_tts = _resolve_tts_text(render_item)
        subtitle_lines, subtitle_text_source, subtitle_same_as_tts = _resolve_subtitle_lines(render_item, tts_text)
        subtitle_text = "".join(subtitle_lines).strip()

        # 3. per-fragment TTS if requested. SRT is generated after this so its
        # timeline can use the TTS audio duration.
        voice_audio = None
        voice_settings_used = None
        tts_error = None
        tts_audio_duration = None
        video_for_compose = raw_frag
        duration_adjustment = "none"
        if effective_voice_id and tts_text:
            voice_path = str(voice_dir / f"{clip_id}_{fid}_voice.mp3")
            tts_result = generate_tts(
                tts_text, voice_path,
                voice_id=effective_voice_id,
                voice_style=voice_style,
            )
            voice_settings_used = tts_result.get("voice_settings")
            if tts_result.get("success"):
                voice_audio = tts_result.get("audio_path")
                tts_audio_duration = _media_duration(voice_audio)
                any_tts_success = True
                try:
                    video_for_compose, duration_adjustment = _adjust_video_duration_for_tts(
                        raw_frag,
                        fe - fs,
                        tts_audio_duration,
                        final_dir / f"{clip_id}_{fid}_video_matched_to_tts.mp4",
                    )
                except Exception as e:
                    result_warnings.append(f"{clip_id}/{fid}: 根据 TTS 时长调整画面失败，沿用原片段: {e}")
                    video_for_compose = raw_frag
                    duration_adjustment = "none"
            else:
                tts_error = tts_result.get("error")
                result_warnings.append(
                    f"{clip_id}/{fid}: TTS 失败，保留原声: {tts_error}")

        srt_path = str(srt_dir / f"{clip_id}_{fid}.srt")
        subtitle_duration = tts_audio_duration if tts_audio_duration and tts_audio_duration > 0 else (fe - fs)
        subtitle_duration_source = "tts_audio_duration" if tts_audio_duration and tts_audio_duration > 0 else "fragment_duration"
        try:
            create_subtitles_from_tts_script(
                voice_script=tts_text or frag_text or "...",
                subtitle_lines=subtitle_lines,
                audio_duration=subtitle_duration,
                output_srt_path=srt_path,
            )
        except Exception as e:
            result_warnings.append(f"{clip_id}/{fid}: 字幕生成失败: {e}")
            srt_path = None

        # 4. compose this fragment (avatar + subtitle + maybe TTS audio)
        composed_path = str(final_dir / f"{clip_id}_{fid}_composed.mp4")
        compose_debug = {}
        try:
            compose_debug = compose_final_video(
                main_video=video_for_compose,
                avatar_image_path=avatar_path,
                voice_audio=voice_audio,
                subtitle_srt=srt_path,
                title=clip.get("title", clip_id),
                output_path=composed_path,
                talking_head_video_path=talking_head_video_path,
                talking_head_full_video_path=talking_head_full_video_path,
                digital_human_window_style=digital_human_window_style,
                layout_config=layout_config,
                return_debug=True,
            )
            composed_paths.append(composed_path)
        except Exception as e:
            result_errors.append(f"{clip_id}/{fid}: 合成失败: {e}")
            continue

        fragment_reports.append({
            "fragment_id": fid,
            "start": fs,
            "end": fe,
            "raw_video": raw_frag,
            "video_for_compose": video_for_compose,
            "composed_video": composed_path,
            "voice_audio": voice_audio,
            "voice_settings": voice_settings_used,
            "tts_error": tts_error,
            "srt": srt_path,
            "cleaned_text": frag_text,
            "tts_text": tts_text,
            "voice_script": tts_text,
            "subtitle_lines": subtitle_lines,
            "tts_text_source": tts_text_source,
            "raw_text_used_for_tts": raw_text_used_for_tts,
            "subtitle_text_source": subtitle_text_source,
            "subtitle_uses_same_text_as_tts": bool(subtitle_same_as_tts and _norm_render_text(subtitle_text) == _norm_render_text(tts_text)),
            "subtitle_duration_source": subtitle_duration_source,
            "tts_audio_duration": tts_audio_duration,
            "subtitle_duration": subtitle_duration,
            "source_video_duration": fe - fs,
            "duration_adjustment": duration_adjustment,
            **_digital_human_clip_report(
                digital_human_report,
                talking_head_video_path,
                final_composer_used_talking_head=compose_debug.get("used_talking_head", False),
                compose_debug=compose_debug,
            ),
            "composer_filter_complex": compose_debug.get("filter_complex"),
        })

    if not composed_paths:
        return None, fragment_reports, any_tts_success

    # 5. concat all composed fragments → final clip
    final_path = str(final_dir / f"{clip_id}_final.mp4")
    try:
        concat_videos(composed_paths, final_path)
    except Exception as e:
        result_errors.append(f"{clip_id}: concat 失败: {e}")
        return None, fragment_reports, any_tts_success
    return final_path, fragment_reports, any_tts_success


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 entrypoint: render videos from an existing plan
# ─────────────────────────────────────────────────────────────────────────────
def generate_videos_from_plan(
    project_dir,
    video_name,
    avatar_name,
    audio_mode="保留原视频原声",
    layout_config=None,
    avatar_scale=0.14,
    avatar_margin_right=24,
    avatar_margin_bottom=100,
    subtitle_size=20,
    subtitle_margin_bottom=40,
    subtitle_max_width_ratio=0.55,
    title_size=28,
    voice_sample_paths=None,
    voice_name=None,
    remove_background_noise=False,
    voice_consent=False,
    voice_style=DEFAULT_VOICE_STYLE,
    render_plan_path=None,
    digital_human_provider=DIGITAL_HUMAN_MODE_STATIC,
    digital_human_mode=None,
    force_regenerate_seedance_avatar=False,
    seedance_quality_mode="fast",
    digital_human_window_style="card",
    digital_human_video_mode="preview_loop",
    fallback_experimental_i2v_to_fast=False,
    source_video_path=None,
    video_input_mode=None,
    source_video_size_bytes=None,
    source_audio_clone_target_seconds=90,
):
    """Render videos for the plan stored in project_dir.

    If a `render_plan_path` is provided (selected_render_plan.json), only the
    fragments referenced there will be rendered. Otherwise every clip in
    clips.json is rendered.

    Clips with `fragments` are rendered fragment-by-fragment then concatenated
    (so non-contiguous fragments never include the gap between them). Clips
    without `fragments` follow the legacy single-cut path.
    """
    project_dir = Path(project_dir)
    clips_path = project_dir / "clips.json"
    if digital_human_mode is not None and digital_human_provider == DIGITAL_HUMAN_MODE_STATIC:
        # Backwards compatibility for callers from the previous round.
        digital_human_provider = digital_human_mode

    if layout_config is None:
        layout_config = LayoutConfig(
            avatar_scale=avatar_scale,
            avatar_margin_right=avatar_margin_right,
            avatar_margin_bottom=avatar_margin_bottom,
            subtitle_size=subtitle_size,
            subtitle_margin_bottom=subtitle_margin_bottom,
            subtitle_max_width_ratio=subtitle_max_width_ratio,
            title_size=title_size,
        )

    audio_mode_key = _normalise_audio_mode(audio_mode)
    source_video_meta = _read_source_video_metadata(project_dir)
    result = {
        "project_dir": str(project_dir),
        "audio_mode": _audio_mode_label(audio_mode_key),
        "audio_mode_key": audio_mode_key,
        "video_input_mode": video_input_mode or source_video_meta.get("video_input_mode"),
        "source_video_path": source_video_path or source_video_meta.get("source_video_path"),
        "source_video_size_bytes": source_video_size_bytes or source_video_meta.get("source_video_size_bytes"),
        "voice_style": voice_style,
        "seedance_model": os.getenv("SEEDANCE_MODEL", DEFAULT_SEEDANCE_MODEL),
        "seedance_model_requested": None,
        "seedance_model_used": None,
        "seedance_generation_type": None,
        "identity_preserved_from_avatar": False,
        "is_lip_sync": False,
        "uses_audio_driven_lipsync": False,
        "reason": None,
        "seedance_error_type": None,
        "seedance_error_code": None,
        "seedance_error_message": None,
        "suggested_action": None,
        "privacy_guard_triggered": False,
        "seedance_model_probe_summary": {},
        "seedance_supported_modes": ["text_to_video", "image_to_video"],
        "seedance_lipsync_supported": False,
        "digital_human_mode": digital_human_provider,
        "digital_human_requested_provider": digital_human_provider,
        "digital_human_actual_provider": DIGITAL_HUMAN_MODE_STATIC,
        "talking_head_video_path": None,
        "force_regenerate_seedance_avatar": force_regenerate_seedance_avatar,
        "quality_mode": seedance_quality_mode,
        "digital_human_window_style": digital_human_window_style,
        "digital_human_video_mode": digital_human_video_mode,
        "fallback_from_2_0_i2v_to_fast": False,
        "fallback_experimental_i2v_to_fast_requested": bool(fallback_experimental_i2v_to_fast),
        "voice_settings_default": get_voice_style_preset(voice_style),
        "layout_config": to_dict(layout_config),
        "computed_layout": None,
        "layout_warnings": [],
        "voice_sample_file_names": [Path(p).name for p in (voice_sample_paths or [])],
        "voice_sample_file_sizes": {
            Path(p).name: Path(p).stat().st_size
            for p in (voice_sample_paths or []) if Path(p).exists()
        },
        "voice_clone_result": None,
        "voice_sample_extraction_result": None,
        "voice_sample_manifest_path": None,
        "source_audio_clone_target_seconds": source_audio_clone_target_seconds,
        "effective_voice_id": None,
        "voice_consent": voice_consent,
        "tts_status": "pending",
        "tts_errors": [],
        "warnings": [],
        "errors": [],
        "clips": [],
        "render_plan_used": render_plan_path,
        "output_mode": "multiple_kp_videos",
        "selected_topic_id": None,
        "selected_hook_id": None,
        "final_video_title": None,
        "final_video_opening_hook": None,
        "selected_kp_ids": [],
        "plan_reusable": True,
        "unit_count": 0,
        "intermediate_unit_videos": [],
        "final_complete_video_path": None,
        "final_video_count_visible_to_user": None,
        "digital_human_applied_per_unit": False,
        "final_video_has_digital_human": False,
        "tts_text_source": "voice_script",
        "subtitle_text_source": "subtitle_lines",
        "subtitle_uses_same_text_as_tts": True,
        "planning_input_text": "cleaned_segments",
        "raw_text_used_for_planning": False,
        "filler_words_removed": True,
        "llm_plan_repaired": False,
        "llm_plan_repair_reason": "none",
        "used_deterministic_fallback": False,
        "used_chunked_planning": False,
        "chunk_count": 0,
        "successful_chunk_count": 0,
        "failed_chunk_count": 0,
        "failed_chunks": [],
        "chunk_fallback_used": False,
        "cached_chunk_count": 0,
        "chunked_planning_reason": None,
        "global_merge_status": None,
        "hook_plan_status": None,
    }

    if not clips_path.exists():
        result["errors"].append("未找到 clips.json，请先生成切片计划")
        return result

    clips_data = json.loads(clips_path.read_text(encoding="utf-8"))
    clips = clips_data.get("clips", []) or []
    for key in (
        "used_chunked_planning",
        "chunk_count",
        "successful_chunk_count",
        "failed_chunk_count",
        "failed_chunks",
        "chunk_fallback_used",
        "cached_chunk_count",
        "chunked_planning_reason",
        "global_merge_status",
        "hook_plan_status",
        "used_deterministic_fallback",
    ):
        if key in clips_data:
            result[key] = clips_data.get(key)
    if not clips:
        result["errors"].append("clips.json 中没有片段")
        return result

    # Apply render plan filter (if any).
    selected_fragment_ids = None
    selected_kp_ids = None
    selected_kp_order = None
    render_plan = None
    render_output_mode = "multiple_kp_videos"
    selected_hook_id = None
    if render_plan_path:
        try:
            rp = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
            render_plan = rp
            render_output_mode = rp.get("render_output_mode") or "multiple_kp_videos"
            selected_hook_id = rp.get("selected_hook_id") or (rp.get("selected_hook_ids") or [None])[0]
            result["output_mode"] = render_output_mode
            result["selected_topic_id"] = rp.get("selected_topic_id")
            result["selected_hook_id"] = selected_hook_id
            result["final_video_title"] = rp.get("final_video_title") or rp.get("title")
            result["final_video_opening_hook"] = rp.get("final_video_opening_hook")
            if rp.get("render_units"):
                selected_kp_order = list(rp.get("selected_kp_ids") or [
                    unit.get("kp_id") for unit in (rp.get("render_units") or []) if unit.get("kp_id")
                ])
                selected_kp_ids = set(selected_kp_order)
                result["selected_kp_ids"] = selected_kp_order
                result["unit_count"] = len(selected_kp_order)
                rp_voice_style = rp.get("voice_style")
                if rp_voice_style:
                    voice_style = rp_voice_style
                    result["voice_style"] = voice_style
                    result["voice_settings_default"] = get_voice_style_preset(voice_style)
            jobs = rp.get("render_jobs") or []
            if jobs:
                first_job = jobs[0]
                selected_fragment_ids = set(first_job.get("selected_fragment_ids") or [])
                rp_voice_style = first_job.get("voice_style")
                if rp_voice_style:
                    voice_style = rp_voice_style
                    result["voice_style"] = voice_style
                    result["voice_settings_default"] = get_voice_style_preset(voice_style)
        except Exception as e:
            result["warnings"].append(f"读取 render_plan 失败，使用全部 clips: {e}")
            selected_fragment_ids = None

    semantic_plan = clips_data.get("knowledge_modules")
    if isinstance(semantic_plan, dict) and semantic_plan.get("knowledge_points"):
        result["llm_plan_repaired"] = bool(semantic_plan.get("llm_plan_repaired"))
        result["llm_plan_repair_reason"] = semantic_plan.get("llm_plan_repair_reason") or "none"
        result["used_deterministic_fallback"] = bool(semantic_plan.get("used_deterministic_fallback", False))
        for key in (
            "used_chunked_planning",
            "chunk_count",
            "successful_chunk_count",
            "failed_chunk_count",
            "failed_chunks",
            "chunk_fallback_used",
            "cached_chunk_count",
            "chunked_planning_reason",
            "global_merge_status",
            "hook_plan_status",
        ):
            if key in semantic_plan:
                result[key] = semantic_plan.get(key)
        ok, gate_reasons, gate_metrics = kp.validate_render_quality(
            semantic_plan,
            selected_render_plan=render_plan,
            require_assembly=False,
        )
        if render_plan is not None and not (
            render_plan.get("selected_topic_id")
            or selected_hook_id
            or render_plan.get("selected_kp_ids")
        ):
            ok = False
            gate_reasons.append("必须选择一个短视频方案")
        result["knowledge_plan_quality_gate"] = {
            "can_enter_video_generation": ok,
            "blocking_reasons": gate_reasons,
            **gate_metrics,
        }
        if not ok:
            result["errors"].append("知识点切片质量门禁未通过：" + "；".join(gate_reasons))
            (project_dir / "report.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

    video_path = _resolve_source_video_path(project_dir, video_name, source_video_path=source_video_path)
    avatar_path = str(project_dir / "uploads" / avatar_name)
    try:
        resolved_size = Path(video_path).stat().st_size
    except Exception:
        resolved_size = None
    result["source_video_path"] = video_path
    result["source_video_size_bytes"] = source_video_size_bytes or resolved_size or result.get("source_video_size_bytes")
    if not result.get("video_input_mode"):
        result["video_input_mode"] = "upload" if str(video_path).startswith(str(project_dir / "uploads")) else "local_or_nas_path"

    try:
        vw, vh = probe_video_size(video_path)
        computed = compute_layout(vw, vh, layout_config)
        result["computed_layout"] = computed
        result["layout_warnings"] = computed.get("warnings", [])
        try:
            from services.layout_engine import estimate_subtitle_layout_report
            result["subtitle_layout"] = estimate_subtitle_layout_report(computed)
        except Exception as e:
            result["layout_warnings"].append(f"subtitle_layout 计算失败: {e}")
    except Exception as e:
        result["layout_warnings"].append(f"布局计算失败: {e}")

    # Resolve TTS voice_id
    effective_voice_id = None
    if audio_mode_key == "existing_elevenlabs_voice_id":
        effective_voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip() or None
        if not effective_voice_id:
            result["warnings"].append("未配置 ELEVENLABS_VOICE_ID，已回退原视频原声")
    elif audio_mode_key == "upload_voice_sample_clone":
        if not voice_consent:
            result["warnings"].append("未确认声音授权，已回退原视频原声")
        elif not voice_sample_paths:
            result["warnings"].append("未上传声音样本，已回退原视频原声")
        else:
            valid = [p for p in voice_sample_paths if Path(p).exists() and Path(p).stat().st_size > 0]
            if not valid:
                result["warnings"].append("声音样本文件为空或不存在，已回退原视频原声")
            else:
                clone_result = clone_voice_with_elevenlabs(
                    sample_paths=valid,
                    voice_name=voice_name or "demo_voice_clone",
                    remove_background_noise=remove_background_noise,
                    description="Created by digital-human-demo",
                )
                result["voice_clone_result"] = {
                    "success": clone_result["success"],
                    "voice_id": clone_result["voice_id"],
                    "requires_verification": clone_result.get("requires_verification"),
                    "error": clone_result["error"],
                }
                if clone_result["success"]:
                    effective_voice_id = clone_result["voice_id"]
                    result["warnings"].append(f"已创建克隆音色 voice_id={effective_voice_id}")
                else:
                    result["warnings"].append(f"克隆失败，回退原声：{clone_result['error']}")
    elif audio_mode_key == "clone_from_source_video_audio":
        if not voice_consent:
            result["warnings"].append("未确认原视频声音授权，已回退原视频原声")
        else:
            transcript_segments = []
            transcript_path = project_dir / "transcript.json"
            if transcript_path.exists():
                try:
                    transcript_segments = (json.loads(transcript_path.read_text(encoding="utf-8")).get("segments") or [])
                except Exception as e:
                    result["warnings"].append(f"读取 transcript segments 失败，改用非静音检测抽样: {e}")
            voice_samples_dir = project_dir / "voice_samples"
            extraction = extract_voice_samples_from_video(
                video_path=video_path,
                output_dir=str(voice_samples_dir),
                target_total_seconds=int(source_audio_clone_target_seconds or 90),
                transcript_segments=transcript_segments,
            )
            result["voice_sample_extraction_result"] = {
                "success": extraction.get("success"),
                "sample_paths": extraction.get("sample_paths") or [],
                "merged_sample_path": extraction.get("merged_sample_path"),
                "total_duration": extraction.get("total_duration"),
                "error": extraction.get("error"),
                "debug": extraction.get("debug") or {},
            }
            result["voice_sample_manifest_path"] = str(voice_samples_dir / "voice_sample_manifest.json")
            if not extraction.get("success"):
                result["warnings"].append(f"原视频声音样本提取失败，已回退原声: {extraction.get('error')}")
            else:
                merged_sample = extraction.get("merged_sample_path")
                clone_result = clone_voice_with_elevenlabs(
                    sample_paths=[merged_sample],
                    voice_name=voice_name or f"source_video_voice_{project_dir.name}",
                    remove_background_noise=remove_background_noise,
                    description="Created from authorized source-video audio by digital-human-demo",
                )
                result["voice_clone_result"] = {
                    "success": clone_result["success"],
                    "voice_id": clone_result["voice_id"],
                    "requires_verification": clone_result.get("requires_verification"),
                    "error": clone_result["error"],
                    "status_code": (clone_result.get("debug") or {}).get("first_status_code"),
                    "response_text": (clone_result.get("debug") or {}).get("first_response_text"),
                }
                if clone_result["success"]:
                    effective_voice_id = clone_result["voice_id"]
                    result["warnings"].append(f"已从原视频声音创建克隆音色 voice_id={effective_voice_id}")
                else:
                    result["warnings"].append(f"原视频声音克隆失败，回退原声：{clone_result['error']}")

    result["effective_voice_id"] = effective_voice_id
    if audio_mode_key == "keep_original_audio":
        result["tts_status"] = "skipped"

    raw_dir = project_dir / "raw_clips"
    srt_dir = project_dir / "subtitles"
    voice_dir = project_dir / "voices"
    final_dir = project_dir / "final"
    intermediate_dir = project_dir / "intermediate_units"
    final_videos_dir = project_dir / "final_videos"
    for d in [raw_dir, srt_dir, voice_dir, final_dir, intermediate_dir, final_videos_dir]:
        d.mkdir(exist_ok=True)

    tts_any_success = False
    talking_head_video_path = None
    talking_head_full_video_path = None
    digital_human_target_duration = _estimate_digital_human_target_duration(
        clips,
        selected_fragment_ids=selected_fragment_ids,
    )
    digital_human_report = _prepare_digital_human(
        provider_key=digital_human_provider,
        avatar_path=avatar_path,
        voice_style=voice_style,
        clips=clips,
        project_dir=project_dir,
        result_warnings=result["warnings"],
        force_regenerate_seedance_avatar=force_regenerate_seedance_avatar,
        quality_mode=seedance_quality_mode,
        avatar_video_mode=digital_human_video_mode,
        target_duration=digital_human_target_duration,
        fallback_experimental_i2v_to_fast=fallback_experimental_i2v_to_fast,
    )
    result["digital_human_provider_result"] = digital_human_report
    seedance_debug = (digital_human_report.get("debug") or {}) if digital_human_report else {}
    result["digital_human_actual_provider"] = digital_human_report.get("actual_provider") or DIGITAL_HUMAN_MODE_STATIC
    result["seedance_generation_type"] = seedance_debug.get("seedance_generation_type")
    result["identity_preserved_from_avatar"] = bool(seedance_debug.get("identity_preserved_from_avatar"))
    result["is_lip_sync"] = bool(digital_human_report.get("is_lip_sync"))
    result["uses_audio_driven_lipsync"] = bool(digital_human_report.get("uses_audio_driven_lipsync"))
    result["fallback_from_2_0_i2v_to_fast"] = bool(
        digital_human_report.get("fallback_from_2_0_i2v_to_fast")
        or seedance_debug.get("fallback_from_2_0_i2v_to_fast")
    )
    result["reason"] = seedance_debug.get("reason")
    result.update(_seedance_privacy_report_fields(digital_human_report))
    if digital_human_report and digital_human_report.get("success"):
        talking_head_video_path = digital_human_report.get("talking_head_video_path")
        talking_head_full_video_path = seedance_debug.get("talking_head_full_video_path")
    result["talking_head_video_path"] = talking_head_video_path
    result["seedance_model_requested"] = seedance_debug.get("seedance_model_requested")
    result["seedance_model_used"] = seedance_debug.get("seedance_model_used")
    result["seedance_fallback_reason"] = seedance_debug.get("fallback_reason")
    result["requested_model_candidates"] = seedance_debug.get("requested_model_candidates")
    result["selected_model"] = seedance_debug.get("seedance_model_used")
    result["fallback_model"] = result["selected_model"] if result.get("seedance_fallback_reason") else None
    result["seedance_model_probe_summary"] = {
        "requested_quality_mode": seedance_quality_mode,
        "requested_model_candidates": result.get("requested_model_candidates"),
        "selected_model": result.get("selected_model"),
        "fallback_model": result.get("fallback_model"),
        "fallback_reason": result.get("seedance_fallback_reason"),
        "attempt_errors": seedance_debug.get("seedance_attempt_errors"),
    }
    result["digital_human_target_duration"] = digital_human_target_duration

    if selected_kp_order:
        order_index = {kid: idx for idx, kid in enumerate(selected_kp_order)}
        clips = sorted(clips, key=lambda c: order_index.get(c.get("kp_id"), len(order_index)))

    for clip in clips:
        clip_id = clip.get("clip_id", "unknown")
        if selected_kp_ids is not None and clip.get("kp_id") not in selected_kp_ids:
            continue
        # Apply render-plan filter to clips with fragments.
        if selected_fragment_ids is not None and clip.get("fragments"):
            filtered = [f for f in clip["fragments"]
                        if f.get("fragment_id") in selected_fragment_ids]
            if not filtered:
                continue
            clip = dict(clip)
            clip["fragments"] = filtered

        cr = {
            "clip_id": clip_id,
            "kp_id": clip.get("kp_id"),
            "module_id": clip.get("module_id"),
            "title": clip.get("title", ""),
            "hook": clip.get("hook", ""),
            "summary": clip.get("summary", ""),
            "topic_key": clip.get("topic_key", ""),
            "start": clip.get("start", 0),
            "end": clip.get("end", 0),
            "duration": clip.get("duration", 0),
            "source_text": clip.get("source_text", ""),
            "cleaned_clip_text": clip.get("cleaned_clip_text", ""),
            "voice_script": clip.get("voice_script", ""),
            "subtitle_lines": clip.get("subtitle_lines", []),
            "tts_text_source": clip.get("tts_text_source") or "voice_script",
            "tts_text": "",
            "raw_text_used_for_tts": False,
            "subtitle_text_source": clip.get("subtitle_text_source") or "subtitle_lines",
            "render_mode": "single" if not clip.get("fragments") else "concat_fragments",
            "fragments_rendered": [],
            "raw_video": None,
            "srt": None,
            "voice_id_used": None,
            "voice_audio": None,
            "voice_style": voice_style,
            "used_original_audio": True,
            "tts_error": None,
            "final_video": None,
            "error": None,
            **_digital_human_clip_report(digital_human_report, talking_head_video_path),
        }

        if clip.get("fragments"):
            # Fragment-aware render path
            final_path, frag_reports, frag_tts_any = _render_clip_with_fragments(
                clip, video_path, avatar_path, layout_config,
                effective_voice_id, voice_style,
                raw_dir, srt_dir, voice_dir, final_dir,
                result["warnings"], result["errors"],
                digital_human_report=digital_human_report,
                talking_head_video_path=talking_head_video_path,
                talking_head_full_video_path=talking_head_full_video_path,
                digital_human_window_style=digital_human_window_style,
            )
            cr["fragments_rendered"] = frag_reports
            cr["final_video"] = final_path
            if frag_reports:
                cr["tts_text_source"] = frag_reports[0].get("tts_text_source") or "voice_script"
                cr["subtitle_text_source"] = frag_reports[0].get("subtitle_text_source") or "subtitle_lines"
                cr["subtitle_uses_same_text_as_tts"] = all(
                    bool(fr.get("subtitle_uses_same_text_as_tts")) for fr in frag_reports
                )
                cr["raw_text_used_for_tts"] = any(bool(fr.get("raw_text_used_for_tts")) for fr in frag_reports)
                cr["tts_text"] = "\n".join(fr.get("tts_text") or "" for fr in frag_reports).strip()
                cr["duration_adjustment"] = [
                    fr.get("duration_adjustment") for fr in frag_reports
                ]
                cr["source_video_duration"] = sum(float(fr.get("source_video_duration") or 0) for fr in frag_reports)
                cr["tts_audio_duration"] = sum(float(fr.get("tts_audio_duration") or 0) for fr in frag_reports)
                cr["subtitle_duration_source"] = (
                    "tts_audio_duration"
                    if all(fr.get("subtitle_duration_source") == "tts_audio_duration" for fr in frag_reports)
                    else "mixed_or_fragment_duration"
                )
                cr["voice_audio"] = frag_reports[0].get("voice_audio")
                cr["srt"] = frag_reports[0].get("srt")
                cr["final_composer_used_talking_head"] = any(
                    bool(fr.get("final_composer_used_talking_head")) for fr in frag_reports
                )
                first_dh_fragment = frag_reports[0]
                for key in (
                    "final_composer_used_window_style",
                    "used_full_length_avatar_video",
                    "used_loop_avatar_video",
                    "used_static_avatar_fallback",
                ):
                    cr[key] = first_dh_fragment.get(key)
            if frag_tts_any:
                tts_any_success = True
                cr["used_original_audio"] = False
                cr["voice_id_used"] = effective_voice_id
            if not final_path:
                cr["error"] = "全部 fragment 都失败"
        else:
            # Legacy single-cut path (timeline mode)
            try:
                raw_path = str(raw_dir / f"{clip_id}_raw.mp4")
                cut_video(video_path, clip.get("start", 0), clip.get("end", 30), raw_path)
                cr["raw_video"] = raw_path
            except Exception as e:
                cr["error"] = f"裁剪失败: {e}"
                result["errors"].append(f"{clip_id}: 裁剪失败: {e}")
                result["clips"].append(cr)
                continue
            voice_audio = None
            tts_text, tts_text_source, raw_text_used_for_tts = _resolve_tts_text(clip)
            subtitle_lines, subtitle_text_source, subtitle_same_as_tts = _resolve_subtitle_lines(clip, tts_text)
            tts_audio_duration = None
            source_video_duration = float(clip.get("end", 30) or 30) - float(clip.get("start", 0) or 0)
            video_for_compose = cr["raw_video"]
            duration_adjustment = "none"
            if effective_voice_id:
                voice_path = str(voice_dir / f"{clip_id}_voice.mp3")
                tts_result = generate_tts(
                    tts_text, voice_path,
                    voice_id=effective_voice_id, voice_style=voice_style,
                )
                cr["voice_id_used"] = effective_voice_id
                if tts_result.get("success"):
                    voice_audio = tts_result["audio_path"]
                    tts_audio_duration = _media_duration(voice_audio)
                    tts_any_success = True
                    cr["voice_audio"] = voice_audio
                    cr["used_original_audio"] = False
                    try:
                        video_for_compose, duration_adjustment = _adjust_video_duration_for_tts(
                            cr["raw_video"],
                            source_video_duration,
                            tts_audio_duration,
                            final_dir / f"{clip_id}_video_matched_to_tts.mp4",
                        )
                    except Exception as e:
                        result["warnings"].append(f"{clip_id}: 根据 TTS 时长调整画面失败，沿用原片段: {e}")
                        video_for_compose = cr["raw_video"]
                        duration_adjustment = "none"
                else:
                    cr["tts_error"] = tts_result.get("error")
                    result["warnings"].append(f"{clip_id}: TTS 失败，回退原声：{tts_result.get('error')}")
            try:
                srt_path = str(srt_dir / f"{clip_id}.srt")
                subtitle_duration = tts_audio_duration if tts_audio_duration and tts_audio_duration > 0 else source_video_duration
                create_subtitles_from_tts_script(
                    voice_script=tts_text or "",
                    subtitle_lines=subtitle_lines,
                    audio_duration=subtitle_duration,
                    output_srt_path=srt_path,
                )
                cr["srt"] = srt_path
                cr["subtitle_duration_source"] = "tts_audio_duration" if tts_audio_duration else "clip_duration"
            except Exception as e:
                result["errors"].append(f"{clip_id}: 字幕失败: {e}")
            cr["tts_text"] = tts_text
            cr["tts_text_source"] = tts_text_source
            cr["raw_text_used_for_tts"] = raw_text_used_for_tts
            cr["subtitle_lines"] = subtitle_lines
            cr["subtitle_text_source"] = subtitle_text_source
            cr["subtitle_uses_same_text_as_tts"] = bool(
                subtitle_same_as_tts and _norm_render_text("".join(subtitle_lines)) == _norm_render_text(tts_text)
            )
            cr["source_video_duration"] = source_video_duration
            cr["tts_audio_duration"] = tts_audio_duration
            cr["duration_adjustment"] = duration_adjustment
            try:
                final_path = str(final_dir / f"{clip_id}_final.mp4")
                compose_debug = compose_final_video(
                    main_video=video_for_compose,
                    avatar_image_path=avatar_path,
                    voice_audio=voice_audio,
                    subtitle_srt=cr["srt"],
                    title=clip.get("title", clip_id),
                    output_path=final_path,
                    talking_head_video_path=talking_head_video_path,
                    talking_head_full_video_path=talking_head_full_video_path,
                    digital_human_window_style=digital_human_window_style,
                    layout_config=layout_config,
                    return_debug=True,
                )
                cr["final_video"] = final_path
                cr["final_composer_used_talking_head"] = compose_debug.get("used_talking_head", False)
                cr.update(_digital_human_clip_report(
                    digital_human_report,
                    talking_head_video_path,
                    final_composer_used_talking_head=compose_debug.get("used_talking_head", False),
                    compose_debug=compose_debug,
                ))
                cr["composer_filter_complex"] = compose_debug.get("filter_complex")
            except Exception as e:
                cr["error"] = f"合成失败: {e}"
                result["errors"].append(f"{clip_id}: 合成失败: {e}")

        result["clips"].append(cr)

    if render_output_mode == "single_complete_video":
        rendered_clips = [c for c in result["clips"] if c.get("final_video") and Path(c.get("final_video")).exists()]
        # Per-topic file naming keeps batch renderings from clobbering each
        # other when multiple topics share final_videos/ and intermediate_units/.
        selected_topic_id = (render_plan or {}).get("selected_topic_id") if render_plan_path else None
        topic_part = selected_topic_id or selected_hook_id or "selected_hook"
        safe_topic = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(topic_part)) or "selected_hook"
        intermediate_paths = []
        for idx, clip_report in enumerate(rendered_clips, 1):
            unit_path = intermediate_dir / f"{safe_topic}_unit_{idx:03d}.mp4"
            try:
                shutil.copyfile(clip_report["final_video"], unit_path)
                intermediate_paths.append(str(unit_path))
            except Exception as e:
                result["errors"].append(f"{safe_topic}_unit_{idx:03d}: 中间视频写入失败: {e}")
        if intermediate_paths:
            final_complete = final_videos_dir / f"{safe_topic}_complete_video.mp4"
            try:
                concat_videos(intermediate_paths, str(final_complete))
                result["final_complete_video_path"] = str(final_complete)
            except Exception as e:
                result["errors"].append(f"完整视频拼接失败: {e}")
        result["selected_topic_id"] = selected_topic_id
        result["topic_safe_id"] = safe_topic
        result["intermediate_unit_videos"] = intermediate_paths
        result["unit_count"] = len(intermediate_paths)
        result["final_video_count_visible_to_user"] = 1 if result.get("final_complete_video_path") else 0
        result["digital_human_applied_per_unit"] = bool(intermediate_paths)
        result["final_video_has_digital_human"] = any(
            bool(c.get("digital_human_success"))
            and (
                bool(c.get("final_composer_used_talking_head"))
                or bool(c.get("used_static_avatar_fallback"))
                or c.get("digital_human_actual_provider") in {
                    DIGITAL_HUMAN_MODE_STATIC,
                    DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
                    DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST,
                    DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL,
                }
            )
            for c in result["clips"]
        )
    else:
        result["final_video_count_visible_to_user"] = len([
            c for c in result["clips"] if c.get("final_video") and Path(c.get("final_video")).exists()
        ])
        result["unit_count"] = len(result["clips"])

    if audio_mode_key == "keep_original_audio":
        result["tts_status"] = "skipped"
    elif effective_voice_id:
        result["tts_status"] = "success" if tts_any_success else "all_failed"
    else:
        result["tts_status"] = "no_voice_id"

    rendered_reports = [c for c in result["clips"] if c.get("final_video")]
    if rendered_reports:
        result["raw_text_used_for_tts"] = any(bool(c.get("raw_text_used_for_tts")) for c in rendered_reports)
        result["subtitle_uses_same_text_as_tts"] = all(
            bool(c.get("subtitle_uses_same_text_as_tts")) for c in rendered_reports
        )
        result["tts_text_source"] = (
            "voice_script"
            if all(c.get("tts_text_source") == "voice_script" for c in rendered_reports)
            else "mixed"
        )
        result["subtitle_text_source"] = (
            "subtitle_lines"
            if all(c.get("subtitle_text_source") == "subtitle_lines" for c in rendered_reports)
            else "mixed"
        )

    (project_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch render: 一次内容计划，多次选择生成视频
# ─────────────────────────────────────────────────────────────────────────────
def generate_videos_from_render_jobs(
    project_dir,
    video_name,
    avatar_name,
    selected_render_jobs_path=None,
    selected_render_jobs=None,
    progress_callback=None,
    **kwargs,
):
    """Render multiple short videos from a precomputed render_jobs selection.

    No STT, no LLM planning happens here. The function reads
    selected_render_jobs.json (or the equivalent dict) and, for each job, writes
    a per-job render plan to ``selected_render_plans/<topic_id>.json`` then
    invokes ``generate_videos_from_plan`` once per job. Each job produces an
    independent ``final_complete_video.mp4``. Aggregated results land in
    ``report.json`` with ``plan_reusable=True``.
    """
    project_dir = Path(project_dir)
    plan_files_exist = (project_dir / "clips.json").exists() and (project_dir / "knowledge_modules.json").exists()
    batch_result = {
        "project_dir": str(project_dir),
        "plan_reusable": True,
        "planning_input_text": "cleaned_segments",
        "raw_text_used_for_planning": False,
        "filler_words_removed": True,
        "tts_text_source": "voice_script",
        "subtitle_text_source": "subtitle_lines",
        "subtitle_uses_same_text_as_tts": True,
        "selected_render_jobs_path": selected_render_jobs_path,
        "selected_render_jobs_count": 0,
        "successful_jobs": 0,
        "failed_jobs": 0,
        "jobs": [],
        "final_videos": [],
        "warnings": [],
        "errors": [],
    }
    if not plan_files_exist:
        batch_result["errors"].append("未找到 clips.json / knowledge_modules.json，请先生成内容计划")
        (project_dir / "report.json").write_text(
            json.dumps(batch_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return batch_result

    if selected_render_jobs is None:
        if not selected_render_jobs_path:
            batch_result["errors"].append("未提供 selected_render_jobs_path 或 selected_render_jobs")
            (project_dir / "report.json").write_text(
                json.dumps(batch_result, ensure_ascii=False, indent=2), encoding="utf-8")
            return batch_result
        try:
            selected_render_jobs = json.loads(Path(selected_render_jobs_path).read_text(encoding="utf-8"))
        except Exception as e:
            batch_result["errors"].append(f"读取 selected_render_jobs 失败: {e}")
            (project_dir / "report.json").write_text(
                json.dumps(batch_result, ensure_ascii=False, indent=2), encoding="utf-8")
            return batch_result

    # Schema validation (fail fast before any heavy rendering kicks off).
    ok, reasons, validation_metrics = kp.validate_selected_render_jobs(selected_render_jobs or {})
    batch_result["selected_render_jobs_validation"] = {
        "ok": ok,
        "reasons": reasons,
        **validation_metrics,
    }
    if not ok:
        batch_result["errors"].extend(reasons)
        (project_dir / "report.json").write_text(
            json.dumps(batch_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return batch_result

    jobs = (selected_render_jobs or {}).get("render_jobs") or []
    batch_result["selected_render_jobs_count"] = len(jobs)

    per_job_plan_dir = project_dir / "selected_render_plans"
    per_job_plan_dir.mkdir(exist_ok=True)

    def _notify_progress(stage, **extra):
        if callable(progress_callback):
            try:
                progress_callback({"stage": stage, **extra})
            except Exception:
                pass

    _notify_progress("batch_start", total_jobs=len(jobs))

    for job_index, job in enumerate(jobs, 1):
        topic_id = job.get("topic_id") or job.get("job_id") or "topic"
        safe_topic = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(topic_id)) or "topic"
        _notify_progress(
            "job_start",
            current_job=job_index,
            total_jobs=len(jobs),
            topic_id=topic_id,
            topic_title=job.get("topic_title") or topic_id,
            successful_jobs=batch_result["successful_jobs"],
            failed_jobs=batch_result["failed_jobs"],
        )
        per_job_plan_path = per_job_plan_dir / f"{safe_topic}.json"
        per_job_plan = {
            "selected_topic_id": topic_id,
            "selected_hook_id": job.get("selected_hook_id"),
            "selected_hook_ids": [job.get("selected_hook_id")] if job.get("selected_hook_id") else [],
            "selected_kp_ids": job.get("selected_kp_ids") or [],
            "render_units": job.get("render_units") or [],
            "render_output_mode": "single_complete_video",
            "fragment_order": job.get("fragment_order") or "user_order",
            "voice_style": job.get("voice_style"),
            "title": job.get("topic_title") or topic_id,
            "final_video_title": job.get("final_video_title") or job.get("topic_title") or topic_id,
            "final_video_opening_hook": job.get("final_video_opening_hook") or job.get("topic_hook") or "",
            "final_video_structure": job.get("final_video_structure") or "按所选知识点顺序讲解",
            "video_type": job.get("video_type") or "short_video",
            "visible_output_count": 1,
        }
        per_job_plan_path.write_text(
            json.dumps(per_job_plan, ensure_ascii=False, indent=2), encoding="utf-8")

        sub_kwargs = dict(kwargs)
        if job.get("voice_style"):
            sub_kwargs["voice_style"] = job["voice_style"]
        try:
            job_result = generate_videos_from_plan(
                project_dir=project_dir,
                video_name=video_name,
                avatar_name=avatar_name,
                render_plan_path=str(per_job_plan_path),
                **sub_kwargs,
            )
        except Exception as e:
            batch_result["failed_jobs"] += 1
            batch_result["jobs"].append({
                "job_id": job.get("job_id"),
                "topic_id": topic_id,
                "topic_title": job.get("topic_title"),
                "final_complete_video_path": None,
                "errors": [f"渲染异常: {e}"],
                "warnings": [],
            })
            batch_result["errors"].append(f"{topic_id}: 渲染异常 {e}")
            _notify_progress(
                "job_done",
                current_job=job_index,
                total_jobs=len(jobs),
                topic_id=topic_id,
                topic_title=job.get("topic_title") or topic_id,
                succeeded=False,
                successful_jobs=batch_result["successful_jobs"],
                failed_jobs=batch_result["failed_jobs"],
            )
            continue

        if not batch_result.get("subtitle_layout") and job_result.get("subtitle_layout"):
            batch_result["subtitle_layout"] = job_result["subtitle_layout"]
        final_video = job_result.get("final_complete_video_path")
        succeeded = bool(final_video) and Path(final_video).exists() and not job_result.get("errors")
        if succeeded:
            batch_result["successful_jobs"] += 1
            batch_result["final_videos"].append(final_video)
        else:
            batch_result["failed_jobs"] += 1
        batch_result["jobs"].append({
            "job_id": job.get("job_id"),
            "topic_id": topic_id,
            "topic_title": job.get("topic_title"),
            "video_type": job.get("video_type"),
            "selected_kp_ids": job.get("selected_kp_ids"),
            "estimated_duration": job.get("estimated_duration"),
            "render_plan_path": str(per_job_plan_path),
            "final_complete_video_path": final_video,
            "final_video_count_visible_to_user": job_result.get("final_video_count_visible_to_user"),
            "intermediate_unit_videos": job_result.get("intermediate_unit_videos") or [],
            "tts_text_source": job_result.get("tts_text_source"),
            "subtitle_text_source": job_result.get("subtitle_text_source"),
            "warnings": job_result.get("warnings") or [],
            "errors": job_result.get("errors") or [],
        })
        batch_result["warnings"].extend([f"{topic_id}: {w}" for w in (job_result.get("warnings") or [])])
        batch_result["errors"].extend([f"{topic_id}: {e}" for e in (job_result.get("errors") or [])])
        _notify_progress(
            "job_done",
            current_job=job_index,
            total_jobs=len(jobs),
            topic_id=topic_id,
            topic_title=job.get("topic_title") or topic_id,
            succeeded=succeeded,
            successful_jobs=batch_result["successful_jobs"],
            failed_jobs=batch_result["failed_jobs"],
            final_complete_video_path=final_video,
        )

    _notify_progress(
        "batch_done",
        total_jobs=len(jobs),
        successful_jobs=batch_result["successful_jobs"],
        failed_jobs=batch_result["failed_jobs"],
    )

    # Aggregate TTS / subtitle provenance across the batch so the on-disk
    # report still tells the truth when one job had to fall back to raw_text.
    job_tts_sources = {j.get("tts_text_source") for j in batch_result["jobs"] if j.get("tts_text_source")}
    job_subtitle_sources = {j.get("subtitle_text_source") for j in batch_result["jobs"] if j.get("subtitle_text_source")}
    if job_tts_sources and job_tts_sources != {"voice_script"}:
        batch_result["tts_text_source"] = "mixed"
        batch_result["warnings"].append("某些 topic 的 TTS 文本不是 voice_script，请检查 jobs.warnings。")
    if job_subtitle_sources and job_subtitle_sources != {"subtitle_lines"}:
        batch_result["subtitle_text_source"] = "mixed"
        batch_result["warnings"].append("某些 topic 的字幕不是来自 subtitle_lines，请检查 jobs.warnings。")
    batch_result["subtitle_uses_same_text_as_tts"] = (
        batch_result["tts_text_source"] == "voice_script"
        and batch_result["subtitle_text_source"] == "subtitle_lines"
    )

    # Read cleaning_report.json (written at plan time) to keep provenance
    # accurate even if user reused a plan after switching cleaning modes.
    cleaning_path = project_dir / "cleaning_report.json"
    if cleaning_path.exists():
        try:
            cleaning_report = json.loads(cleaning_path.read_text(encoding="utf-8"))
            batch_result["filler_words_removed"] = bool(cleaning_report.get("filler_words_removed", True))
            batch_result["planning_input_text"] = cleaning_report.get("planning_input_text") or "cleaned_segments"
            batch_result["raw_text_used_for_planning"] = bool(cleaning_report.get("raw_text_used_for_planning", False))
        except Exception:
            pass

    # The batch report overrides the per-job report.json written by the last
    # generate_videos_from_plan call so the on-disk report reflects the whole
    # batch rather than the final job alone.
    (project_dir / "report.json").write_text(
        json.dumps(batch_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return batch_result


# ─────────────────────────────────────────────────────────────────────────────
# Editable-clip persistence (unchanged from previous round)
# ─────────────────────────────────────────────────────────────────────────────
_EDITABLE_FIELDS = (
    "start", "end", "title", "hook", "summary",
    "source_text", "cleaned_clip_text", "voice_script",
)


def save_all_clips_changes(project_dir, clips_data):
    project_dir = Path(project_dir)
    clips_json_path = project_dir / "clips.json"
    if not clips_json_path.exists():
        raise FileNotFoundError(f"clips.json not found at {clips_json_path}")
    data = json.loads(clips_json_path.read_text(encoding="utf-8"))
    existing_clips = data.get("clips", []) or []
    by_id = {c.get("clip_id"): c for c in existing_clips}
    touched_ids = []
    for new_c in clips_data or []:
        cid = new_c.get("clip_id")
        if not cid or cid not in by_id:
            continue
        target = by_id[cid]
        for k in _EDITABLE_FIELDS:
            if k in new_c:
                target[k] = new_c[k]
        try:
            target["duration"] = float(target.get("end", 0) or 0) - float(target.get("start", 0) or 0)
        except Exception:
            target["duration"] = 0
        touched_ids.append(cid)
    clips_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for cid in touched_ids:
        c = by_id[cid]
        (project_dir / f"{cid}_source_text.txt").write_text(c.get("source_text", "") or "", encoding="utf-8")
        (project_dir / f"{cid}_cleaned_text.txt").write_text(c.get("cleaned_clip_text", "") or "", encoding="utf-8")
        (project_dir / f"{cid}_voice_script.txt").write_text(c.get("voice_script", "") or "", encoding="utf-8")
    return existing_clips


def save_selected_render_plan(project_dir, plan):
    """Persist selected_render_plan.json."""
    project_dir = Path(project_dir)
    (project_dir / "selected_render_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(project_dir / "selected_render_plan.json")
