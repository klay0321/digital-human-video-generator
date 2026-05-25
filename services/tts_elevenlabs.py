import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from services.voice_style import get_voice_style_preset, DEFAULT_VOICE_STYLE

load_dotenv()


def generate_tts(text, output_path, voice_id=None,
                 voice_settings=None, voice_style=None,
                 model_id="eleven_multilingual_v2"):
    """Call ElevenLabs TTS to render `text` into MP3.

    voice_settings:
        Explicit dict overrides everything. Keys may include
        stability / similarity_boost / style / speed / use_speaker_boost.
    voice_style:
        Named preset (see services.voice_style). Used only if voice_settings
        is None.
    If neither is given, the default preset (讲解风格) is used.

    Returns:
        {
          "success":          bool,
          "audio_path":       str | None,
          "error":            str | None,
          "voice_style":      str,             # which preset name was used
          "voice_settings":   dict,            # the settings actually sent
          "text_length":      int,
          "model_id":         str,
        }
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    style_name = voice_style or DEFAULT_VOICE_STYLE
    if voice_settings is None:
        voice_settings = get_voice_style_preset(style_name)
    else:
        voice_settings = dict(voice_settings)

    # Build the JSON-serialisable summary used by the report; default this
    # before any early return so failure paths still tell us what was tried.
    report_settings = dict(voice_settings)

    api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        return {
            "success": False, "audio_path": None,
            "error": "缺少 ELEVENLABS_API_KEY",
            "voice_style": style_name, "voice_settings": report_settings,
            "text_length": len(text or ""), "model_id": model_id,
        }

    vid = (voice_id or "").strip()
    if not vid:
        vid = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
    if not vid:
        return {
            "success": False, "audio_path": None,
            "error": "缺少 voice_id",
            "voice_style": style_name, "voice_settings": report_settings,
            "text_length": len(text or ""), "model_id": model_id,
        }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text or "",
        "model_id": model_id,
        "voice_settings": voice_settings,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return {
                "success": False, "audio_path": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "voice_style": style_name, "voice_settings": report_settings,
                "text_length": len(text or ""), "model_id": model_id,
            }
        Path(output_path).write_bytes(resp.content)
        return {
            "success": True, "audio_path": output_path, "error": None,
            "voice_style": style_name, "voice_settings": report_settings,
            "text_length": len(text or ""), "model_id": model_id,
        }
    except Exception as e:
        return {
            "success": False, "audio_path": None, "error": str(e),
            "voice_style": style_name, "voice_settings": report_settings,
            "text_length": len(text or ""), "model_id": model_id,
        }
