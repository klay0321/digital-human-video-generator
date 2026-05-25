import re
from pathlib import Path


def format_seconds(seconds: float) -> str:
    """格式化秒数为 HH:MM:SS。"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def seconds_to_srt_time(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_sentences(text: str) -> list[str]:
    """按中文标点和换行拆句。"""
    parts = re.split(r"[。！？\n]+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]


def make_simple_srt(text: str, output_path: str, duration: float) -> str:
    """根据文本生成简单 SRT 字幕文件。

    每句按比例分配时间，输出 UTF-8 编码。
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sentences = _split_sentences(text)
    n = len(sentences)
    per_sentence = duration / n if n > 0 else duration

    lines: list[str] = []
    for i, sent in enumerate(sentences):
        start = i * per_sentence
        end = (i + 1) * per_sentence
        lines.append(str(i + 1))
        lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        lines.append(sent)
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def create_subtitles_from_tts_script(
    voice_script: str,
    subtitle_lines: list[str],
    audio_duration: float,
    output_srt_path: str,
    max_chars_per_line: int = 20,
) -> str:
    """Create SRT subtitles from the same text used by TTS.

    The subtitle timeline is distributed across the actual TTS audio duration,
    not the source transcript timestamps. This keeps cloned/cleaned narration
    aligned with the generated voice_script.
    """
    output_srt_path = str(Path(output_srt_path))
    Path(output_srt_path).parent.mkdir(parents=True, exist_ok=True)

    script = " ".join(str(voice_script or "").split()).strip()
    lines = [str(line or "").strip() for line in (subtitle_lines or []) if str(line or "").strip()]
    if not lines:
        lines = split_text_for_subtitles(script, max_chars_per_line=max_chars_per_line)
    else:
        # Do not trust LLM-returned lines when they do not come from the script.
        if _normalise_text("".join(lines)) not in _normalise_text(script):
            lines = split_text_for_subtitles(script, max_chars_per_line=max_chars_per_line)

    if not lines:
        lines = ["..."]

    duration = max(float(audio_duration or 0), 0.1)
    weights = [max(len(_normalise_text(line)), 1) for line in lines]
    total_weight = max(sum(weights), 1)
    min_line_seconds = 1.2

    raw_durations = [duration * (weight / total_weight) for weight in weights]
    adjusted = [max(min_line_seconds, item) for item in raw_durations]
    adjusted_total = sum(adjusted)
    if adjusted_total > duration and adjusted_total > 0:
        scale = duration / adjusted_total
        adjusted = [max(0.3, item * scale) for item in adjusted]

    # Make the total end exactly match the TTS duration.
    if adjusted:
        adjusted[-1] += duration - sum(adjusted)
        adjusted[-1] = max(0.3, adjusted[-1])

    srt_lines: list[str] = []
    cursor = 0.0
    for idx, (line, line_duration) in enumerate(zip(lines, adjusted), 1):
        start = cursor
        end = duration if idx == len(lines) else min(duration, cursor + line_duration)
        srt_lines.append(str(idx))
        srt_lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        srt_lines.append(line)
        srt_lines.append("")
        cursor = end

    Path(output_srt_path).write_text("\n".join(srt_lines), encoding="utf-8")
    return output_srt_path


def split_text_for_subtitles(text: str, max_chars_per_line: int = 20) -> list[str]:
    """Split Chinese narration into readable subtitle lines."""
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return []
    parts = re.split(r"([。！？!?；;，,、])", clean)
    chunks: list[str] = []
    current = ""
    for idx in range(0, len(parts), 2):
        piece = parts[idx]
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""
        token = (piece + punct).strip()
        if not token:
            continue
        if len(current) + len(token) <= max_chars_per_line:
            current += token
        else:
            if current:
                chunks.append(current)
            while len(token) > max_chars_per_line:
                chunks.append(token[:max_chars_per_line])
                token = token[max_chars_per_line:]
            current = token
    if current:
        chunks.append(current)
    return chunks


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _ass_time(seconds: float) -> str:
    """将秒数转换为 ASS 时间格式 H:MM:SS.cc"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def make_ass_subtitle(
    text: str,
    output_path: str,
    duration: float,
    font_size: int = 22,
    margin_right: int = 200,
) -> str:
    """生成 ASS 字幕文件，支持右侧安全边距（不遮挡右下角头像）。"""
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sentences = _split_sentences(text)
    n = len(sentences)
    per_sentence = duration / n if n > 0 else duration

    header = f"""[Script Info]
Title: Digital Human Subtitle
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,{margin_right},32,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for i, sent in enumerate(sentences):
        start = i * per_sentence
        end = (i + 1) * per_sentence
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{sent}"
        )

    content = header + "\n".join(events) + "\n"
    Path(output_path).write_text(content, encoding="utf-8-sig")
    return output_path
