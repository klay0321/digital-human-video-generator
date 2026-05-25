import re
import subprocess
from pathlib import Path

from services.ffmpeg_utils import _cmd
from services.layout_engine import (
    LayoutConfig, probe_video_size, compute_layout,
    make_ass_subtitle_for_clip, build_ffmpeg_filter_complex,
    _FONT_PARAM, _escape_ffmpeg_text,
)


def _run(cmd):
    cmd = _cmd(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stderr


def compose_final_video(
    main_video,
    avatar_image_path=None,
    voice_audio=None,
    subtitle_srt=None,
    title="",
    output_path=None,
    talking_head_video_path=None,
    talking_head_full_video_path=None,
    digital_human_window_style="direct",
    return_debug=False,
    layout_config=None,
    # Legacy kwargs (kept for backwards compat with old callers).
    avatar_scale=0.14,
    avatar_margin_right=24,
    avatar_margin_bottom=100,
    subtitle_size=20,
    subtitle_margin_bottom=40,
    subtitle_max_width_ratio=0.55,
    title_size=28,
):
    """Compose one final clip video.

    All positions come from layout_engine.compute_layout — there are NO
    hardcoded overlay coordinates in this module. On subtitle failure we
    retry without subtitle but keep the avatar overlay and the audio track.
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

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

    # Single source of truth for layout (shared with preview).
    try:
        w, h = probe_video_size(main_video)
    except Exception:
        w, h = 1920, 1080  # last-resort fallback only when ffprobe itself fails
    layout = compute_layout(w, h, layout_config)

    # Build ASS via the shared generator.
    ass_path = None
    if subtitle_srt and Path(subtitle_srt).exists():
        try:
            ass_path = str(Path(output_path).parent / f"_{Path(output_path).stem}.ass")
            srt_texts = _extract_srt_texts(subtitle_srt)
            full_text = "。".join(srt_texts) if srt_texts else ""
            duration = _srt_duration(subtitle_srt)
            if full_text:
                make_ass_subtitle_for_clip(full_text, ass_path, duration, layout, layout_config)
            else:
                ass_path = None
        except Exception as e:
            print(f"[compose] ASS build failed: {e}")
            ass_path = None

    full_head_info = _validate_talking_head_video(talking_head_full_video_path)
    loop_head_info = _validate_talking_head_video(talking_head_video_path)
    use_full_head_video = bool(full_head_info["valid"])
    use_loop_head_video = bool(loop_head_info["valid"]) and not use_full_head_video
    selected_head_path = talking_head_full_video_path if use_full_head_video else talking_head_video_path
    selected_head_info = full_head_info if use_full_head_video else loop_head_info
    use_talking_head_video = use_full_head_video or use_loop_head_video
    duration = _video_duration(main_video)
    use_card_style = digital_human_window_style == "card"
    fc = _build_filter_complex(
        title, ass_path, layout["avatar_width"], layout, layout_config,
        use_talking_head_video=use_talking_head_video,
        use_card_style=use_card_style,
        duration=duration,
    )
    debug = {
        "output_path": output_path,
        "used_talking_head": use_talking_head_video,
        "talking_head_video_path": str(selected_head_path) if selected_head_path else None,
        "talking_head_validation": selected_head_info,
        "talking_head_full_video_path": str(talking_head_full_video_path) if talking_head_full_video_path else None,
        "talking_head_full_validation": full_head_info,
        "talking_head_loop_validation": loop_head_info,
        "used_full_length_avatar_video": use_full_head_video,
        "used_loop_avatar_video": use_loop_head_video,
        "used_static_avatar_fallback": not use_talking_head_video,
        "final_composer_used_window_style": digital_human_window_style,
        "filter_complex": fc,
    }

    try:
        _compose(
            main_video, avatar_image_path, voice_audio, fc, output_path,
            talking_head_video_path=selected_head_path if use_talking_head_video else None,
            loop_talking_head=use_loop_head_video,
        )
    except RuntimeError as e:
        # Retry: drop subtitle, but keep avatar overlay and audio mapping.
        print(f"[compose] subtitle pass failed, retrying without subtitle: {e}")
        fc_nosub = _build_filter_complex(
            title, None, layout["avatar_width"], layout, layout_config,
            use_talking_head_video=use_talking_head_video,
            use_card_style=use_card_style,
            duration=duration,
        )
        debug["filter_complex"] = fc_nosub
        _compose(
            main_video, avatar_image_path, voice_audio, fc_nosub, output_path,
            talking_head_video_path=selected_head_path if use_talking_head_video else None,
            loop_talking_head=use_loop_head_video,
        )

    if return_debug:
        return debug
    return output_path


def _compose(main_video, avatar_image, voice_audio, filter_complex, output_path,
             talking_head_video_path=None, loop_talking_head=False):
    cmd = ["ffmpeg", "-y", "-i", str(main_video)]
    if talking_head_video_path and Path(talking_head_video_path).exists():
        if loop_talking_head:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", str(talking_head_video_path)]
    else:
        if not avatar_image:
            raise RuntimeError("avatar_image_path is required when talking_head_video_path is not valid")
        cmd += ["-loop", "1", "-i", str(avatar_image)]
    if voice_audio and Path(voice_audio).exists():
        cmd += ["-i", str(voice_audio)]
    cmd += ["-filter_complex", filter_complex, "-map", "[v]"]
    if voice_audio and Path(voice_audio).exists():
        cmd += ["-map", "2:a"]
    else:
        cmd += ["-map", "0:a?"]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", "-shortest", output_path]
    _run(cmd)


def _build_filter_complex(title, subtitle_ass_path, avatar_w, layout, config,
                          use_talking_head_video=False, use_card_style=False,
                          duration=None):
    if not use_talking_head_video and not use_card_style:
        return build_ffmpeg_filter_complex(title, subtitle_ass_path, avatar_w, layout, config)

    title_f = _title_filter(title, layout, config)
    parts = [f"[0:v]{title_f}[t]"]

    if subtitle_ass_path and Path(subtitle_ass_path).exists():
        ass_esc = str(subtitle_ass_path).replace("\\", "/").replace(":", "\\:")
        parts.append(f"[t]ass='{ass_esc}'[s]")
        prev = "s"
    else:
        prev = "t"

    trim = f"trim=duration={max(float(duration or 0), 0.1):.3f}," if duration else ""
    if use_card_style:
        parts.extend(_card_overlay_filters(prev, avatar_w, layout, use_talking_head_video, trim, duration))
    else:
        ax = layout["avatar_x"]
        ay = layout["avatar_y"]
        parts.append(
            f"[1:v]{trim}setpts=PTS-STARTPTS,"
            f"scale={avatar_w}:{avatar_w}:force_original_aspect_ratio=increase,"
            f"crop={avatar_w}:{avatar_w},setsar=1[digital]"
        )
        parts.append(_framed_direct_overlay(prev, ax, ay, avatar_w, duration))
    return ";".join(parts)


def _card_overlay_filters(prev_label, avatar_w, layout, input_is_video, trim, duration):
    video_w = max(48, avatar_w - 28)
    video_h = max(48, int(video_w * 0.82))
    header_h = max(24, int(avatar_w * 0.12))
    footer_h = max(18, int(avatar_w * 0.09))
    pad = max(8, int(avatar_w * 0.04))
    card_w = avatar_w
    card_h = header_h + video_h + footer_h + pad * 3
    margin_right = max(0, layout["video_width"] - (layout["avatar_x"] + layout["avatar_width"]))
    margin_bottom = max(0, layout["video_height"] - (layout["avatar_y"] + layout["avatar_width"]))
    card_x = max(0, layout["video_width"] - card_w - margin_right)
    card_y = max(0, layout["video_height"] - card_h - margin_bottom)
    video_x = card_x + pad + (card_w - video_w - pad * 2) // 2
    video_y = card_y + header_h + pad
    shadow_x = card_x + max(4, pad // 2)
    shadow_y = card_y + max(5, pad // 2)
    border_t = 2
    card_duration = max(float(duration or 0.1), 0.1)
    top_label = _escape_ffmpeg_text("AI讲解中")
    bottom_label = _escape_ffmpeg_text("数字讲解员")
    font_small = max(12, int(avatar_w * 0.06))
    font_tiny = max(10, int(avatar_w * 0.045))

    parts = []
    parts.append(
        f"color=c=black@0.28:s={card_w}x{card_h}:d={card_duration:.3f},format=rgba,"
        f"{_rounded_alpha_filter(max(10, pad * 2), 72)}[dh_shadow]"
    )
    parts.append(
        f"[{prev_label}][dh_shadow]overlay=x={shadow_x}:y={shadow_y}:shortest=1[dh_shadowed]"
    )
    parts.append(
        f"color=c=0x07131f@0.78:s={card_w}x{card_h}:d={card_duration:.3f},format=rgba,"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x8bb8ff@0.45:t={border_t},"
        f"drawtext={_FONT_PARAM}:text='{top_label}':fontsize={font_small}:fontcolor=white@0.94:x={pad}:y={max(5, pad // 2)},"
        f"drawtext={_FONT_PARAM}:text='{bottom_label}':fontsize={font_tiny}:fontcolor=white@0.70:x={pad}:y={card_h - footer_h},"
        f"{_rounded_alpha_filter(max(10, pad * 2), 200)}[dh_card]"
    )
    parts.append(
        f"[dh_shadowed][dh_card]overlay=x={card_x}:y={card_y}:shortest=1[dh_card_base]"
    )
    digital_prefix = f"[1:v]{trim}setpts=PTS-STARTPTS" if input_is_video else "[1:v]setpts=PTS-STARTPTS"
    parts.append(
        f"{digital_prefix},scale={video_w}:{video_h}:force_original_aspect_ratio=increase,"
        f"crop={video_w}:{video_h},setsar=1[digital]"
    )
    parts.append(
        f"[dh_card_base][digital]overlay=x={video_x}:y={video_y}:shortest=1[v]"
    )
    return parts


def _rounded_alpha_filter(radius, alpha):
    radius = max(1, int(radius))
    alpha = max(0, min(255, int(alpha)))
    r2 = radius * radius
    return (
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        f"a='if(lte(pow(max(abs(X-W/2)-(W/2-{radius})\\,0)\\,2)+"
        f"pow(max(abs(Y-H/2)-(H/2-{radius})\\,0)\\,2)\\,{r2})\\,{alpha}\\,0)'"
    )


def _framed_direct_overlay(prev_label, ax, ay, avatar_w, duration):
    frame_pad = max(4, int(avatar_w * 0.025))
    frame_w = avatar_w + frame_pad * 2
    frame_h = avatar_w + frame_pad * 2
    shadow_x = ax + max(3, frame_pad)
    shadow_y = ay + max(4, frame_pad)
    frame_x = max(0, ax - frame_pad)
    frame_y = max(0, ay - frame_pad)
    frame_duration = max(float(duration or 0.1), 0.1)
    return (
        f"color=c=black@0.22:s={frame_w}x{frame_h}:d={frame_duration:.3f},format=rgba[dh_direct_shadow];"
        f"[{prev_label}][dh_direct_shadow]overlay=x={shadow_x}:y={shadow_y}:shortest=1[dh_direct_base];"
        f"color=c=0x07131f@0.34:s={frame_w}x{frame_h}:d={frame_duration:.3f},format=rgba,"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.22:t=2[dh_direct_frame];"
        f"[dh_direct_base][dh_direct_frame]overlay=x={frame_x}:y={frame_y}:shortest=1[dh_direct_framed];"
        f"[dh_direct_framed][digital]overlay=x={ax}:y={ay}:shortest=1[v]"
    )


def _title_filter(title, layout, config):
    # Reuse layout_engine indirectly for static overlays; keep this small clone
    # here so dynamic video overlays can still use the same title placement.
    from services.layout_engine import build_drawtext_title_filter
    return build_drawtext_title_filter(title, config, layout)


def _video_duration(path):
    try:
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
            return max(float((result.stdout or "0").strip() or 0), 0.1)
    except Exception:
        pass
    return 0.1


def _validate_talking_head_video(path):
    if not path:
        return {"valid": False, "exists": False, "size": 0, "duration": None, "error": "empty path"}
    p = Path(path)
    if not p.exists():
        return {"valid": False, "exists": False, "size": 0, "duration": None, "error": "missing file"}
    size = p.stat().st_size
    if size <= 0:
        return {"valid": False, "exists": True, "size": size, "duration": None, "error": "empty file"}
    duration = _video_duration(p)
    if duration <= 0.5:
        return {"valid": False, "exists": True, "size": size, "duration": duration, "error": "duration <= 0.5s"}
    return {"valid": True, "exists": True, "size": size, "duration": duration, "error": None}


def _srt_duration(srt_path):
    text = Path(srt_path).read_text(encoding="utf-8")
    times = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->", text)
    if not times:
        return 30.0
    h, m, s, ms = times[-1]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _extract_srt_texts(srt_path):
    text = Path(srt_path).read_text(encoding="utf-8")
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or re.match(r"^\d+$", line) or re.match(r"\d{2}:\d{2}:\d{2}", line):
            continue
        lines.append(line)
    return lines
