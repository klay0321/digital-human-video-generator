import json
import subprocess
from pathlib import Path

from services.ffmpeg_utils import _cmd
from services.layout_engine import (
    LayoutConfig, probe_video_size, compute_layout,
    make_ass_subtitle_for_clip, build_ffmpeg_filter_complex,
)


# A deliberately long default so users immediately see whether the layout
# survives realistic subtitle lengths.
_DEFAULT_PREVIEW_TEXT = (
    "这是一段较长的字幕预览，用来检验在真实长度下字幕是否会进入右下角数字人区域。"
    "如果排版正确，这段字幕应当被严格限制在左下角的安全区，不会和头像发生重叠。"
)


def _resolve_subtitle_text(subtitle_text, clips_json_path, char_limit=80):
    """Pick the most realistic subtitle text for preview.

    Priority:
      1. Caller-supplied subtitle_text (if non-empty).
      2. clip_001.voice_script (first char_limit chars) from clips.json.
      3. Built-in deliberately long default.
    """
    if subtitle_text and str(subtitle_text).strip():
        return str(subtitle_text).strip()[:char_limit + 40], "caller"

    if clips_json_path:
        try:
            cp = Path(clips_json_path)
            if cp.exists():
                data = json.loads(cp.read_text(encoding="utf-8"))
                clips = data.get("clips") or []
                if clips:
                    first = clips[0]
                    script = (
                        first.get("voice_script")
                        or first.get("cleaned_clip_text")
                        or first.get("source_text")
                        or ""
                    ).strip()
                    if script:
                        cid = first.get("clip_id", "clip_001")
                        return script[:char_limit], f"clips.json[{cid}].voice_script"
        except Exception:
            pass

    # Default: cap at char_limit + 40 so it's clearly longer than typical 1 line
    return _DEFAULT_PREVIEW_TEXT[:char_limit + 40], "default"


def create_layout_preview(
    video_path,
    avatar_image_path,
    title,
    output_path,
    layout_config,
    subtitle_text=None,
    clips_json_path=None,
):
    """Render a single preview frame using the EXACT same layout pipeline as
    the final video (same compute_layout, same ASS generator, same
    filter_complex).

    Returns:
      {
        "preview_path": str,
        "subtitle_text_used": str,
        "subtitle_text_source": str,
        "computed_layout": dict,
        "layout_warnings": list[str],
        "ass_path": str | None,
        "filter_complex": str,
      }
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Probe the ORIGINAL video directly (do not re-encode a temp clip — that
    #    can flip W/H for rotated mobile recordings and silently break parity
    #    with the final video).
    w, h = probe_video_size(video_path)

    # 2. Single source of truth for layout.
    layout = compute_layout(w, h, layout_config)

    # 3. Resolve the subtitle text we will actually render.
    sub_text, sub_text_source = _resolve_subtitle_text(subtitle_text, clips_json_path)

    # 4. Build the ASS via the shared generator (NEVER hand-roll positions here).
    tmp_dir = Path(output_path).parent
    ass_path = str(tmp_dir / f"_preview_{Path(output_path).stem}.ass")
    try:
        make_ass_subtitle_for_clip(sub_text, ass_path, 3.0, layout, layout_config)
    except Exception:
        ass_path = None

    # 5. Build the filter graph via the shared builder.
    fc = build_ffmpeg_filter_complex(title, ass_path, layout["avatar_width"], layout, layout_config)

    # 6. Grab a single frame from ~1s into the original video (so the preview
    #    represents settled content, not the cold first frame).
    cmd = _cmd([
        "ffmpeg", "-y",
        "-ss", "1",
        "-i", str(video_path),
        "-i", str(avatar_image_path),
        "-filter_complex", fc,
        "-map", "[v]",
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        # Degrade: drop the subtitle pass, keep title + avatar.
        fc_nosub = build_ffmpeg_filter_complex(title, None, layout["avatar_width"], layout, layout_config)
        cmd2 = _cmd([
            "ffmpeg", "-y",
            "-ss", "1",
            "-i", str(video_path),
            "-i", str(avatar_image_path),
            "-filter_complex", fc_nosub,
            "-map", "[v]",
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ])
        subprocess.run(cmd2, capture_output=True, check=True)

    return {
        "preview_path": output_path,
        "subtitle_text_used": sub_text,
        "subtitle_text_source": sub_text_source,
        "computed_layout": layout,
        "layout_warnings": list(layout.get("warnings", [])),
        "ass_path": ass_path,
        "filter_complex": fc,
    }
