"""Probe Seedance video+audio lip-sync only with public media URLs.

Usage:
    python scripts/test_seedance_lipsync_with_url.py

This script intentionally does not use local data URLs for reference_video.
The current Ark API response requires reference_video to be a public web URL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(path: Path) -> bool:
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return True


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.seedance_client import DEFAULT_SEEDANCE_MODEL, SeedanceClient, SeedanceClientError  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "seedance_lipsync_url_test"
RESULT_PATH = OUTPUT_DIR / "result.json"


def write_result(payload):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    video_url = (os.getenv("SEEDANCE_REFERENCE_VIDEO_URL") or "").strip()
    audio_url = (os.getenv("SEEDANCE_REFERENCE_AUDIO_URL") or "").strip()
    if not video_url or not audio_url:
        message = "video_audio_lipsync 需要公网 URL，当前未配置，跳过测试。"
        result = {
            "supported": False,
            "skipped": True,
            "error": message,
            "required_env": ["SEEDANCE_REFERENCE_VIDEO_URL", "SEEDANCE_REFERENCE_AUDIO_URL"],
        }
        write_result(result)
        print(message)
        print(f"[result] wrote {RESULT_PATH}")
        return 0

    try:
        client = SeedanceClient(
            api_key=os.getenv("SEEDANCE_API_KEY", ""),
            api_base=os.getenv("SEEDANCE_API_BASE", "https://ark.cn-beijing.volces.com/api/v3"),
            model=os.getenv("SEEDANCE_MODEL", DEFAULT_SEEDANCE_MODEL),
            endpoint_generate=os.getenv("SEEDANCE_ENDPOINT_GENERATE", "contents/generations/tasks"),
            endpoint_status=os.getenv("SEEDANCE_ENDPOINT_STATUS", "contents/generations/tasks/{task_id}"),
            endpoint_upload=os.getenv("SEEDANCE_ENDPOINT_UPLOAD", ""),
        )
        create_response = client.create_task(
            "video_audio_lipsync",
            uploaded_assets={
                "video": {"url": video_url},
                "audio": {"url": audio_url},
            },
        )
        task_id = client.extract_task_id(create_response)
        if not task_id:
            raise RuntimeError("Seedance response did not include task id")
        final_response = client.poll_task(task_id)
        output_path = client.download_result(final_response, OUTPUT_DIR / "video_audio_lipsync.mp4")
        result = {
            "supported": bool(output_path or client.extract_output_video_url(final_response)),
            "task_id": task_id,
            "output_video": output_path or client.extract_output_video_url(final_response),
            "error": None,
            "response": final_response,
        }
        write_result(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except SeedanceClientError as error:
        result = {
            "supported": False,
            "task_id": None,
            "output_video": None,
            "error": error.message,
            "status_code": error.status_code,
            "response_text": error.response_text,
            "request_payload": error.request_payload,
            "endpoint": error.endpoint,
        }
        write_result(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    except Exception as error:
        result = {"supported": False, "error": str(error)}
        write_result(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
