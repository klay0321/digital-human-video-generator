import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def _get_api_key() -> str:
    key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("请在 .env 中配置 ELEVENLABS_API_KEY")
    return key


def words_to_segments(words: list[dict], max_chars: int = 80, max_gap: float = 0.8) -> list[dict]:
    """将 word 级时间戳合并为 segment 级。

    合并规则：
    - 累积文本超过 max_chars 时切段
    - 相邻 word 间隔超过 max_gap 秒时切段
    """
    if not words:
        return []

    segments: list[dict] = []
    current_text = ""
    seg_start = words[0].get("start", 0.0)
    seg_end = seg_start

    for w in words:
        w_start = w.get("start", 0.0)
        w_end = w.get("end", w_start)
        w_text = w.get("text", "")

        # 判断是否需要切段
        if current_text and (
            len(current_text) + len(w_text) > max_chars
            or (w_start - seg_end) > max_gap
        ):
            segments.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": current_text.strip(),
            })
            current_text = ""
            seg_start = w_start

        current_text += w_text
        seg_end = w_end

    # 最后一段
    if current_text.strip():
        segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": current_text.strip(),
        })

    return segments


def transcribe_with_elevenlabs(audio_or_video_path: str, language_code: str = "zh") -> dict:
    """调用 ElevenLabs STT API 转写音频/视频文件。

    返回: {"full_text": str, "segments": list[dict]}
    """
    api_key = _get_api_key()
    file_path = Path(audio_or_video_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {audio_or_video_path}")

    url = "https://api.elevenlabs.io/v1/speech-to-text"

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        data = {
            "model_id": "scribe_v2",
            "language_code": language_code,
            "timestamps_granularity": "word",
            "diarize": "false",
            "tag_audio_events": "false",
        }
        headers = {"xi-api-key": api_key}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)

    if resp.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs STT 失败 (HTTP {resp.status_code}): {resp.text[:300]}"
        )

    result = resp.json()
    full_text = result.get("text", "")
    words_raw = result.get("words", [])

    # 将 API 返回的 word 格式统一为 {start, end, text}
    words = []
    for w in words_raw:
        words.append({
            "start": w.get("start", 0.0),
            "end": w.get("end", 0.0),
            "text": w.get("text", ""),
        })

    segments = words_to_segments(words)

    return {
        "full_text": full_text,
        "segments": segments,
    }
