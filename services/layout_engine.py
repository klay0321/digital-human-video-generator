import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from services.ffmpeg_utils import _cmd


@dataclass
class LayoutConfig:
    layout_mode: str = "right_avatar_bottom_center_subtitle"

    avatar_scale: float = 0.14
    avatar_margin_right: int = 24
    avatar_margin_bottom: int = 100

    subtitle_size: int = 28
    subtitle_margin_left: int = 36
    subtitle_margin_bottom: int = 80
    subtitle_max_width_ratio: float = 0.62

    # New visual controls (round 4): centered subtitles + adjustable outline /
    # shadow / box.
    subtitle_align: str = "center"          # "center" | "left" | "right"
    subtitle_outline_width: float = 2.5     # ASS Outline (px)
    subtitle_shadow_depth: float = 1.0      # ASS Shadow (px)
    subtitle_show_box: bool = False         # True → BorderStyle=3 半透明背景
    subtitle_box_opacity: int = 140         # 0..255, 越大越不透明

    title_size: int = 28
    title_margin_left: int = 32
    title_margin_top: int = 28

    safe_gap: int = 32

    # Hard wrap policy: never let a single Dialogue exceed this many lines.
    max_lines_per_dialogue: int = 2


# Windows Chinese font (degrades gracefully if missing).
_FONT_FILE = "C:/Windows/Fonts/msyh.ttc"
_FONT_ESCAPED = _FONT_FILE.replace("\\", "/").replace(":", "\\:")
_FONT_PARAM = f"fontfile='{_FONT_ESCAPED}'" if Path(_FONT_FILE).exists() else ""


def probe_video_size(video_path):
    """Return (width, height) of the FIRST video stream of the ORIGINAL file."""
    cmd = _cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video_path),
    ])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    info = json.loads(r.stdout)
    return info["streams"][0]["width"], info["streams"][0]["height"]


def compute_layout(width, height, config):
    """Compute pixel positions for avatar / subtitle / title.

    Subtitle layout policy (round-4):
      * Anchor is **bottom-center** by default (Alignment=2 in ASS).
      * Horizontal width is constrained so the subtitle box never enters the
        avatar safe-region (subtitle_right_limit = avatar_x - safe_gap).
      * When the avatar pushes the available area, the subtitle box stays
        centered inside the available area; we accept a small visual offset
        to the left rather than overlapping the avatar.
      * The function returns `subtitle_align`, `subtitle_anchor_x` and
        `subtitle_center_x` so the renderer can emit the correct ASS \\pos
        without duplicating the math.
    """
    warnings = []

    # 1. Avatar at right-bottom (assume ~square avatar of width avatar_w).
    avatar_w = max(40, int(width * config.avatar_scale))
    avatar_x = width - avatar_w - config.avatar_margin_right
    avatar_y = height - avatar_w - config.avatar_margin_bottom

    # 2. Available horizontal range for the subtitle, excluding the avatar.
    subtitle_right_limit = avatar_x - config.safe_gap
    avail_left = config.subtitle_margin_left
    avail_right = subtitle_right_limit
    avail_width = max(0, avail_right - avail_left)

    max_w_by_ratio = int(width * config.subtitle_max_width_ratio)
    if avail_width < 100:
        warnings.append(
            f"avatar overlaps subtitle area: avail_right={avail_right}, "
            f"avail_left={avail_left}; reduce avatar size or margins."
        )
        sub_w = max(100, avail_width)
    else:
        sub_w = min(max_w_by_ratio, avail_width)
        if max_w_by_ratio > avail_width:
            warnings.append(
                f"subtitle_width auto-shrunk from {max_w_by_ratio} to {sub_w} "
                f"to keep clear of avatar (right_limit={subtitle_right_limit})."
            )

    align = (config.subtitle_align or "center").lower()
    if align not in {"center", "left", "right"}:
        align = "center"

    if align == "left":
        sub_x = avail_left
    elif align == "right":
        sub_x = max(avail_left, avail_right - sub_w)
    else:  # center: keep the subtitle box centered inside the available area.
        sub_x = max(avail_left, avail_left + (avail_width - sub_w) // 2)

    # Hard guarantee.
    if sub_x + sub_w > avail_right and avail_width >= 100:
        sub_x = max(avail_left, avail_right - sub_w)

    if sub_w < 200:
        warnings.append(
            f"subtitle width is narrow ({sub_w}px); long lines will wrap frequently."
        )

    subtitle_center_x = sub_x + sub_w // 2
    if align == "left":
        subtitle_anchor_x = sub_x
    elif align == "right":
        subtitle_anchor_x = sub_x + sub_w
    else:
        subtitle_anchor_x = subtitle_center_x

    # 3. Title (top-left).
    title_x = config.title_margin_left
    title_y = config.title_margin_top

    # 4. Approximate subtitle block top (for diagnostic display only).
    fs = config.subtitle_size
    margin_b = config.subtitle_margin_bottom
    subtitle_block_top = height - margin_b - fs * max(1, config.max_lines_per_dialogue)

    return {
        "video_width": width,
        "video_height": height,
        "avatar_width": avatar_w,
        "avatar_x": avatar_x,
        "avatar_y": avatar_y,
        "subtitle_x": sub_x,
        "subtitle_y": subtitle_block_top,
        "subtitle_width": sub_w,
        "subtitle_right_limit": subtitle_right_limit,
        "subtitle_align": align,
        "subtitle_center_x": subtitle_center_x,
        "subtitle_anchor_x": subtitle_anchor_x,
        "subtitle_margin_bottom": margin_b,
        "subtitle_font_size": fs,
        "subtitle_max_lines": max(1, config.max_lines_per_dialogue),
        "subtitle_outline_width": float(getattr(config, "subtitle_outline_width", 2.5) or 2.5),
        "subtitle_shadow_depth": float(getattr(config, "subtitle_shadow_depth", 1.0) or 1.0),
        "subtitle_show_box": bool(getattr(config, "subtitle_show_box", False)),
        "subtitle_box_opacity": int(getattr(config, "subtitle_box_opacity", 140) or 0),
        "title_x": title_x,
        "title_y": title_y,
        "warnings": warnings,
    }


def estimate_subtitle_layout_report(layout):
    """Return a compact subtitle_layout block suitable for report.json."""
    if not isinstance(layout, dict):
        return {}
    align = layout.get("subtitle_align", "center")
    return {
        "position": "bottom_center_safe" if align == "center" else (
            "bottom_left_safe" if align == "left" else "bottom_right_safe"
        ),
        "align": align,
        "max_lines": int(layout.get("subtitle_max_lines", 2)),
        "font_size": int(layout.get("subtitle_font_size", 28)),
        "outline_width": float(layout.get("subtitle_outline_width", 2.5)),
        "shadow_depth": float(layout.get("subtitle_shadow_depth", 1.0)),
        "show_box": bool(layout.get("subtitle_show_box", False)),
        "box_opacity": int(layout.get("subtitle_box_opacity", 140)),
        "margin_bottom": int(layout.get("subtitle_margin_bottom", 80)),
        "avoid_digital_human_window": True,
        "max_chars_per_line": int(
            max(8, layout.get("subtitle_width", 0) // max(1, layout.get("subtitle_font_size", 28)))
        ),
        "computed_box": {
            "x": int(layout.get("subtitle_x", 0)),
            "y": int(layout.get("subtitle_y", 0)),
            "width": int(layout.get("subtitle_width", 0)),
            "right_limit": int(layout.get("subtitle_right_limit", 0)),
            "anchor_x": int(layout.get("subtitle_anchor_x", 0)),
            "center_x": int(layout.get("subtitle_center_x", 0)),
        },
    }


def estimate_subtitle_font_size(width):
    """Suggest a default subtitle font size based on the resolution."""
    try:
        w = int(width or 0)
    except Exception:
        w = 0
    if w >= 1920:
        return 30
    if w >= 1280:
        return 24
    if w >= 720:
        return 20
    if w >= 480:
        return 16
    return 18


def _escape_ffmpeg_text(text):
    """Escape special chars for FFmpeg drawtext."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


def build_drawtext_title_filter(title, config, layout):
    """Generate the title drawtext filter chain segment."""
    esc = _escape_ffmpeg_text(title)
    x = layout["title_x"]
    y = layout["title_y"]
    fs = config.title_size
    return (
        f"drawtext={_FONT_PARAM}:text='{esc}'"
        f":fontsize={fs}:fontcolor=white:borderw=2:bordercolor=black"
        f":x={x}:y={y}"
    )


def _wrap_sentence_to_lines(sentence, max_chars):
    """Hard-wrap a sentence into chunks of <= max_chars (CJK-friendly)."""
    if max_chars < 1:
        max_chars = 8
    lines = []
    s = sentence
    while len(s) > max_chars:
        lines.append(s[:max_chars])
        s = s[max_chars:]
    if s:
        lines.append(s)
    return lines


def _ass_time(seconds):
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _alignment_code(align):
    return {"left": 1, "center": 2, "right": 3}.get((align or "center").lower(), 2)


def _ass_back_color(opacity_0_255):
    """Encode a black backdrop with the requested opacity into ASS BackColour.

    ASS colour is &HAABBGGRR where AA is *transparency* (0=opaque, 255=fully
    transparent). The caller passes "opacity" (255 = fully opaque) so we flip.
    """
    opacity = max(0, min(255, int(opacity_0_255 or 0)))
    alpha = 255 - opacity
    return f"&H{alpha:02X}000000"


def make_ass_subtitle_for_clip(text, output_path, duration, layout, config):
    """Generate an ASS subtitle file anchored to the bottom-CENTER safe area.

    Strategy (must match preview EXACTLY):
      * Hard-wrap the text in Python — never rely on ASS auto-wrap when using
        \\pos (\\pos disables natural line breaks).
      * One Chinese char is assumed to be ~ subtitle_size px wide, so
        max_chars_per_line = max(8, subtitle_width // subtitle_size).
      * Each Dialogue holds at most config.max_lines_per_dialogue lines
        (joined with \\N). If the text needs more lines, we split into
        multiple Dialogues across time.
      * Alignment / anchor follow layout["subtitle_align"] from compute_layout.
        Default is bottom-center → ASS Alignment=2 with \\pos(center_x, bottom).
      * Outline / shadow / background opacity are configurable.
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fs = max(1, config.subtitle_size)
    sub_x = layout["subtitle_x"]
    sub_w = layout["subtitle_width"]
    vid_w = layout["video_width"]
    vid_h = layout["video_height"]
    margin_b = config.subtitle_margin_bottom
    max_lines = max(1, getattr(config, "max_lines_per_dialogue", 2))

    align = layout.get("subtitle_align") or getattr(config, "subtitle_align", "center")
    alignment_code = _alignment_code(align)
    anchor_x = layout.get("subtitle_anchor_x") or layout.get("subtitle_center_x") or (sub_x + sub_w // 2)

    outline = float(getattr(config, "subtitle_outline_width", 2.5) or 2.5)
    shadow = float(getattr(config, "subtitle_shadow_depth", 1.0) or 0.0)
    show_box = bool(getattr(config, "subtitle_show_box", False))
    box_opacity = int(getattr(config, "subtitle_box_opacity", 140) or 0)
    border_style = 3 if show_box else 1
    back_colour = _ass_back_color(box_opacity if show_box else 128)

    # CJK-conservative wrap (the audit found the old fs/2 estimate caused overflow).
    max_chars_per_line = max(8, sub_w // fs)

    # 1. Sentence-level split.
    parts = re.split(r"[。！？\n]+", text or "")
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        sentences = [(text or "").strip() or "..."]

    # 2. Per-sentence hard wrap.
    all_lines = []
    for sent in sentences:
        all_lines.extend(_wrap_sentence_to_lines(sent, max_chars_per_line))
    if not all_lines:
        all_lines = ["..."]

    # 3. Pack lines into Dialogues of at most max_lines lines each.
    dialogues = []
    for i in range(0, len(all_lines), max_lines):
        block = all_lines[i:i + max_lines]
        dialogues.append("\\N".join(block))

    # 4. Time slicing.
    n = len(dialogues)
    safe_duration = duration if duration and duration > 0 else max(1.0, n * 2.0)
    per = safe_duration / n

    # 5. Anchor coordinates.
    pos_y = vid_h - margin_b

    header = f"""[Script Info]
Title: Subtitle
ScriptType: v4.00+
PlayResX: {vid_w}
PlayResY: {vid_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,{fs},&H00FFFFFF,&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline:.2f},{shadow:.2f},{alignment_code},{int(sub_x)},{max(0, vid_w - (sub_x + sub_w))},{margin_b},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for i, block in enumerate(dialogues):
        start = i * per
        end = (i + 1) * per
        styled = f"{{\\pos({int(anchor_x)},{int(pos_y)})}}{block}"
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{styled}"
        )

    content = header + "\n".join(events) + "\n"
    Path(output_path).write_text(content, encoding="utf-8-sig")
    return output_path


def build_ffmpeg_filter_complex(title, subtitle_ass_path, avatar_w, layout, config):
    """Build the full filter_complex chain.

    Order is fixed: title (drawtext) -> subtitle (ass) -> avatar overlay (top).
    All coordinates come from the layout dict — no hardcoded positions.
    """
    title_f = build_drawtext_title_filter(title, config, layout)

    ax = layout["avatar_x"]
    ay = layout["avatar_y"]

    parts = [f"[0:v]{title_f}[t]"]

    if subtitle_ass_path and Path(subtitle_ass_path).exists():
        ass_esc = str(subtitle_ass_path).replace("\\", "/").replace(":", "\\:")
        parts.append(f"[t]ass='{ass_esc}'[s]")
        prev = "s"
    else:
        prev = "t"

    # Avatar last -> stays on top, never hidden behind subtitle.
    parts.append(f"[1:v]scale={avatar_w}:-1[a]")
    parts.append(f"[{prev}][a]overlay=x={ax}:y={ay}[v]")

    return ";".join(parts)


def to_dict(config):
    return asdict(config)
