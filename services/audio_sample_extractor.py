"""Extract clean-ish voice samples from a source video for voice cloning.

The extractor is deliberately conservative:
- it never writes a full-length WAV for long videos;
- it prefers transcript timestamps when available;
- otherwise it scans for non-silent ranges with ffmpeg silencedetect;
- it records every selected range in a manifest for report.json/debugging.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from services.ffmpeg_utils import _cmd, run_cmd


FILLER_PATTERNS = (
    "嗯",
    "啊",
    "额",
    "呃",
    "就是",
    "然后这个",
    "那个",
    "这个",
)


def extract_voice_samples_from_video(
    video_path: str,
    output_dir: str,
    target_total_seconds: int = 90,
    min_total_seconds: int = 45,
    max_total_seconds: int = 180,
    segment_min_seconds: float = 6.0,
    segment_max_seconds: float = 20.0,
    transcript_segments: list[dict[str, Any]] | None = None,
) -> dict:
    """Extract and merge source-video voice samples for ElevenLabs cloning."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "success": False,
        "sample_paths": [],
        "merged_sample_path": None,
        "total_duration": 0,
        "error": None,
        "debug": {
            "source_video_path": str(video_path),
            "selected_segments": [],
            "selection_method": None,
            "target_total_seconds": target_total_seconds,
            "min_total_seconds": min_total_seconds,
            "max_total_seconds": max_total_seconds,
            "segment_min_seconds": segment_min_seconds,
            "segment_max_seconds": segment_max_seconds,
        },
    }

    src = Path(video_path)
    if not src.exists():
        result["error"] = f"source video does not exist: {video_path}"
        _write_manifest(out_dir, result)
        return result

    target_total_seconds = int(max(min_total_seconds, min(target_total_seconds, max_total_seconds)))
    video_duration = _probe_duration(src)
    result["debug"]["source_video_duration"] = video_duration

    selected = _select_from_transcript_segments(
        transcript_segments or [],
        target_total_seconds=target_total_seconds,
        min_total_seconds=min_total_seconds,
        max_total_seconds=max_total_seconds,
        segment_min_seconds=segment_min_seconds,
        segment_max_seconds=segment_max_seconds,
    )
    if selected:
        result["debug"]["selection_method"] = "transcript_segments"
    else:
        selected = _select_from_silence_detection(
            src,
            video_duration=video_duration,
            target_total_seconds=target_total_seconds,
            min_total_seconds=min_total_seconds,
            max_total_seconds=max_total_seconds,
            segment_min_seconds=segment_min_seconds,
            segment_max_seconds=segment_max_seconds,
        )
        result["debug"]["selection_method"] = "silencedetect"

    if not selected:
        result["error"] = "no suitable voice segments found"
        _write_manifest(out_dir, result)
        return result

    sample_paths = []
    selected_reports = []
    for idx, seg in enumerate(selected, 1):
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        sample_path = out_dir / f"voice_sample_{idx:03d}.wav"
        try:
            _extract_audio_range(src, start, end, sample_path)
            duration = _probe_duration(sample_path) or (end - start)
            if duration <= 0:
                continue
            sample_paths.append(str(sample_path))
            selected_reports.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "reason": seg.get("reason") or "clean speech segment with enough duration",
                "text_preview": (seg.get("text") or "")[:120],
            })
        except Exception as exc:
            result["debug"].setdefault("extract_errors", []).append({
                "start": start,
                "end": end,
                "error": str(exc),
            })

    if not sample_paths:
        result["error"] = "voice segment extraction failed"
        _write_manifest(out_dir, result)
        return result

    merged_path = out_dir / "voice_clone_sample_merged.wav"
    try:
        _merge_wav_files(sample_paths, merged_path)
    except Exception as exc:
        result["error"] = f"merge voice samples failed: {exc}"
        _write_manifest(out_dir, result)
        return result

    # Keep a small extracted_audio.wav artifact for users/debugging without
    # materializing the whole original video's audio.
    extracted_audio_path = out_dir / "extracted_audio.wav"
    try:
        shutil.copyfile(merged_path, extracted_audio_path)
    except Exception:
        pass

    total_duration = _probe_duration(merged_path) or sum(seg["duration"] for seg in selected_reports)
    result.update({
        "success": bool(total_duration and total_duration >= min_total_seconds),
        "sample_paths": sample_paths,
        "merged_sample_path": str(merged_path),
        "total_duration": round(float(total_duration or 0), 3),
        "error": None if total_duration and total_duration >= min_total_seconds else (
            f"extracted sample duration {total_duration:.1f}s is below minimum {min_total_seconds}s"
        ),
    })
    result["debug"]["selected_segments"] = selected_reports
    result["debug"]["extracted_audio_path"] = str(extracted_audio_path)
    _write_manifest(out_dir, result)
    return result


def _select_from_transcript_segments(
    segments: list[dict[str, Any]],
    *,
    target_total_seconds: int,
    min_total_seconds: int,
    max_total_seconds: int,
    segment_min_seconds: float,
    segment_max_seconds: float,
) -> list[dict[str, Any]]:
    candidates = []
    for seg in segments:
        try:
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", 0) or 0)
        except Exception:
            continue
        duration = end - start
        if duration < segment_min_seconds:
            continue
        text = (seg.get("clean_text") or seg.get("cleaned_text") or seg.get("text") or "").strip()
        if len(text) < 12:
            continue
        filler_ratio = _filler_ratio(text)
        if filler_ratio > 0.35:
            continue
        score = len(text) * 0.08 + min(duration, segment_max_seconds) - filler_ratio * 20
        candidates.append({
            "start": start,
            "end": min(end, start + segment_max_seconds),
            "duration": min(duration, segment_max_seconds),
            "text": text,
            "score": score,
            "reason": "transcript segment with long clean_text and low filler ratio",
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return _take_until_target(candidates, target_total_seconds, max_total_seconds, min_total_seconds)


def _select_from_silence_detection(
    video_path: Path,
    *,
    video_duration: float | None,
    target_total_seconds: int,
    min_total_seconds: int,
    max_total_seconds: int,
    segment_min_seconds: float,
    segment_max_seconds: float,
) -> list[dict[str, Any]]:
    intervals = _detect_non_silent_intervals(video_path, video_duration)
    candidates = []
    for start, end in intervals:
        duration = end - start
        if duration < segment_min_seconds:
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + segment_max_seconds)
            if chunk_end - cursor >= segment_min_seconds:
                candidates.append({
                    "start": cursor,
                    "end": chunk_end,
                    "duration": chunk_end - cursor,
                    "text": "",
                    "score": chunk_end - cursor,
                    "reason": "non-silent audio interval detected by ffmpeg",
                })
            cursor = chunk_end
    return _take_until_target(candidates, target_total_seconds, max_total_seconds, min_total_seconds)


def _take_until_target(
    candidates: list[dict[str, Any]],
    target_total_seconds: int,
    max_total_seconds: int,
    min_total_seconds: int,
) -> list[dict[str, Any]]:
    selected = []
    total = 0.0
    for item in candidates:
        if total >= target_total_seconds:
            break
        duration = float(item.get("duration", 0) or 0)
        if total + duration > max_total_seconds:
            continue
        selected.append(item)
        total += duration
    if total < min_total_seconds and len(selected) < len(candidates):
        for item in candidates:
            if item in selected:
                continue
            duration = float(item.get("duration", 0) or 0)
            if total + duration > max_total_seconds:
                continue
            selected.append(item)
            total += duration
            if total >= min_total_seconds:
                break
    return sorted(selected, key=lambda item: float(item.get("start", 0) or 0))


def _detect_non_silent_intervals(video_path: Path, video_duration: float | None) -> list[tuple[float, float]]:
    cmd = _cmd([
        "ffmpeg", "-hide_banner", "-i", str(video_path),
        "-af", "silencedetect=noise=-35dB:d=0.7",
        "-f", "null", "-",
    ])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    silence_starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    silence_ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    duration = float(video_duration or 0)
    if duration <= 0:
        duration = _probe_duration(video_path) or 0
    if duration <= 0:
        return []

    intervals = []
    cursor = 0.0
    for idx, silence_start in enumerate(silence_starts):
        silence_end = silence_ends[idx] if idx < len(silence_ends) else duration
        if silence_start > cursor:
            intervals.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        intervals.append((cursor, duration))
    return intervals


def _extract_audio_range(video_path: Path, start: float, end: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(float(end) - float(start), 0.1)
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output_path),
    ])


def _merge_wav_files(sample_paths: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(sample_paths) == 1:
        shutil.copyfile(sample_paths[0], output_path)
        return
    list_file = output_path.parent / "_voice_sample_concat.txt"
    lines = []
    for path in sample_paths:
        abs_path = str(Path(path).resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{abs_path}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")
    try:
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path),
        ])
    finally:
        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass


def _probe_duration(path: Path) -> float | None:
    try:
        proc = subprocess.run(
            _cmd([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode == 0:
            return float((proc.stdout or "0").strip() or 0)
    except Exception:
        return None
    return None


def _filler_ratio(text: str) -> float:
    if not text:
        return 1.0
    filler_count = sum(text.count(pattern) * len(pattern) for pattern in FILLER_PATTERNS)
    return min(1.0, filler_count / max(len(text), 1))


def _write_manifest(output_dir: Path, result: dict) -> None:
    manifest = {
        "source_video_path": result.get("debug", {}).get("source_video_path"),
        "selected_segments": result.get("debug", {}).get("selected_segments") or [],
        "merged_sample_path": result.get("merged_sample_path"),
        "total_duration": result.get("total_duration", 0),
        "success": result.get("success"),
        "error": result.get("error"),
        "debug": result.get("debug") or {},
    }
    (output_dir / "voice_sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
