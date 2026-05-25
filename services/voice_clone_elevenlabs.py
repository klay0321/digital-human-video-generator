import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def get_mime_type(path):
    """Return ElevenLabs-friendly MIME type based on extension."""
    ext = Path(path).suffix.lower()
    mapping = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
    }
    return mapping.get(ext, "application/octet-stream")


def clone_voice_with_elevenlabs(
    sample_paths,
    voice_name,
    remove_background_noise: bool = False,
    description: Optional[str] = None,
):
    """Call ElevenLabs POST /v1/voices/add to create a cloned voice.

    Always returns a dict with the same shape (success or failure).
    """
    debug = {
        "sample_paths": [str(p) for p in (sample_paths or [])],
        "file_sizes": {},
        "first_field_name": None,
        "first_status_code": None,
        "first_response_text": None,
        "retry_field_name": None,
        "retry_status_code": None,
        "retry_response_text": None,
        "content_type_header": None,
    }

    api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        return _fail("missing ELEVENLABS_API_KEY", debug)

    valid_paths = []
    for p in sample_paths or []:
        if not p:
            continue
        pp = Path(p)
        if not pp.exists():
            continue
        try:
            size = pp.stat().st_size
        except Exception:
            continue
        debug["file_sizes"][pp.name] = size
        if size > 0:
            valid_paths.append(pp)

    if not valid_paths:
        return _fail("no valid sample files (missing or empty)", debug)

    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": api_key}

    data = {"name": voice_name or "demo_voice_clone"}
    if remove_background_noise:
        data["remove_background_noise"] = "true"
    if description:
        data["description"] = description

    first = _post_multipart(url, headers, data, valid_paths, "files[]")
    debug["first_field_name"] = "files[]"
    debug["first_status_code"] = first["status_code"]
    debug["first_response_text"] = first["response_text"]
    debug["content_type_header"] = first["content_type_header"]

    if first["ok"] and first["voice_id"]:
        return {
            "success": True,
            "voice_id": first["voice_id"],
            "requires_verification": (first["raw"] or {}).get("requires_verification", False),
            "error": None,
            "raw": first["raw"],
            "debug": debug,
        }

    second = _post_multipart(url, headers, data, valid_paths, "files")
    debug["retry_field_name"] = "files"
    debug["retry_status_code"] = second["status_code"]
    debug["retry_response_text"] = second["response_text"]
    if second["content_type_header"]:
        debug["content_type_header"] = second["content_type_header"]

    if second["ok"] and second["voice_id"]:
        return {
            "success": True,
            "voice_id": second["voice_id"],
            "requires_verification": (second["raw"] or {}).get("requires_verification", False),
            "error": None,
            "raw": second["raw"],
            "debug": debug,
        }

    parts = []
    if first["status_code"] is not None:
        parts.append("[files[]] HTTP " + str(first["status_code"]) + ": " + (first["response_text"] or "")[:300])
    elif first["exception"]:
        parts.append("[files[]] exception: " + first["exception"])
    if second["status_code"] is not None:
        parts.append("[files] HTTP " + str(second["status_code"]) + ": " + (second["response_text"] or "")[:300])
    elif second["exception"]:
        parts.append("[files] exception: " + second["exception"])

    return _fail(" | ".join(parts) or "ElevenLabs clone failed (unknown)", debug)


def _post_multipart(url, headers, data, valid_paths, field_name):
    """Run one multipart POST. Files are freshly opened every call."""
    result = {
        "ok": False,
        "status_code": None,
        "response_text": None,
        "content_type_header": None,
        "raw": None,
        "voice_id": None,
        "exception": None,
    }

    opened = []
    try:
        files = []
        for p in valid_paths:
            f = open(p, "rb")
            opened.append(f)
            files.append((field_name, (p.name, f, get_mime_type(p))))

        try:
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=300)
        except Exception as e:
            result["exception"] = "requests.post error: " + str(e)
            return result

        result["status_code"] = resp.status_code
        try:
            result["response_text"] = (resp.text or "")[:1000]
        except Exception:
            result["response_text"] = None

        try:
            req_headers = getattr(resp.request, "headers", {}) or {}
            result["content_type_header"] = req_headers.get("Content-Type") or req_headers.get("content-type")
        except Exception:
            pass

        if resp.status_code >= 400:
            return result

        try:
            raw = resp.json()
        except Exception as e:
            result["exception"] = "JSON parse error: " + str(e)
            return result

        result["raw"] = raw
        vid = raw.get("voice_id") if isinstance(raw, dict) else None
        if vid:
            result["voice_id"] = vid
            result["ok"] = True
        return result

    except Exception as e:
        result["exception"] = str(e)
        return result
    finally:
        for f in opened:
            try:
                f.close()
            except Exception:
                pass


def _fail(error, debug):
    return {
        "success": False,
        "voice_id": None,
        "requires_verification": None,
        "error": error,
        "raw": None,
        "debug": debug,
    }
