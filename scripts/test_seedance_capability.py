"""Probe Seedance 2.0 API capabilities without touching the main pipeline.

Usage:
    python scripts/test_seedance_capability.py

The project currently has no official Seedance API docs checked in. Because of
that, the client deliberately refuses to invent a task payload schema. This
script still validates env/materials/endpoints, runs every probe that can be
run, and records complete failure details in outputs/seedance_capability_test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on local env
    def load_dotenv(path: Path) -> bool:
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return True


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.seedance_client import (  # noqa: E402
    SeedanceClient,
    SeedanceClientError,
    SeedanceConfigurationError,
)


PROMPT = "A professional Chinese presenter explaining AI tools in a clean studio."
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "seedance_capability_test"
RESULT_PATH = OUTPUT_DIR / "result.json"


def empty_result() -> Dict[str, Any]:
    return {
        "supported": False,
        "task_id": None,
        "output_video": None,
        "error": None,
        "status_code": None,
        "response_text": None,
        "request_payload": None,
        "endpoint": None,
    }


def error_result(message: str, **extra: Any) -> Dict[str, Any]:
    result = empty_result()
    result["error"] = message
    result.update(extra)
    return result


def client_error_result(error: SeedanceClientError) -> Dict[str, Any]:
    result = empty_result()
    result["error"] = error.message
    result["status_code"] = error.status_code
    result["response_text"] = error.response_text
    result["request_payload"] = error.request_payload
    result["endpoint"] = error.endpoint
    return result


def write_results(results: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def find_materials() -> Dict[str, Optional[Path]]:
    test_dir = PROJECT_ROOT / "test"
    avatar = None
    for name in ("avatar.jpg", "avatar.png"):
        candidate = test_dir / name
        if candidate.exists():
            avatar = candidate
            break

    # Compatibility fallback for the current repo: keep the requested names as
    # the primary contract, but allow the probe to run when a single test image
    # already exists under another name.
    if avatar is None:
        for pattern in ("*.jpg", "*.png", "*.jpeg"):
            matches = sorted(test_dir.glob(pattern))
            if matches:
                avatar = matches[0]
                break

    audio = test_dir / "test_01.m4a"
    short_video = test_dir / "short_video.mp4"
    if not short_video.exists():
        video_matches = sorted(test_dir.glob("*.mp4"))
        short_video = video_matches[0] if video_matches else short_video

    return {
        "avatar": avatar,
        "audio": audio if audio.exists() else None,
        "short_video": short_video if short_video.exists() else None,
    }


def print_materials(materials: Dict[str, Optional[Path]]) -> None:
    for name, path in materials.items():
        if path:
            print(f"[material] {name}: {path} ({path.stat().st_size} bytes)")
        else:
            print(f"[material] {name}: missing")


def upload_required_assets(
    client: SeedanceClient,
    assets: Dict[str, Optional[Path]],
) -> Dict[str, Any]:
    uploaded: Dict[str, Any] = {}
    for purpose, path in assets.items():
        if path is None:
            raise SeedanceConfigurationError(
                f"Missing required test material for {purpose}",
                request_payload={"purpose": purpose},
            )
        uploaded[purpose] = client.upload_file(path, purpose=purpose)
    return uploaded


def run_probe(
    client: SeedanceClient,
    capability: str,
    *,
    prompt: Optional[str] = None,
    assets: Optional[Dict[str, Optional[Path]]] = None,
) -> Dict[str, Any]:
    try:
        uploaded_assets = upload_required_assets(client, assets or {}) if assets else {}
        create_response = client.create_task(
            capability,
            prompt=prompt,
            uploaded_assets=uploaded_assets,
        )
        task_id = client.extract_task_id(create_response)
        if not task_id:
            return error_result(
                "Seedance create_task response did not include a task id",
                request_payload={"capability": capability, "prompt": prompt},
                response_text=json.dumps(create_response, ensure_ascii=False),
                endpoint=client.endpoint_generate,
            )

        final_response = client.poll_task(task_id)
        local_video = client.download_result(
            final_response,
            OUTPUT_DIR / f"{capability}.mp4",
        )
        output_video = local_video or client.extract_output_video_url(final_response)
        if not output_video:
            return error_result(
                "Seedance task finished without an output video URL",
                task_id=task_id,
                response_text=json.dumps(final_response, ensure_ascii=False),
                endpoint=client.endpoint_status,
            )
        return {
            **empty_result(),
            "supported": True,
            "task_id": task_id,
            "output_video": output_video,
            "response_text": json.dumps(final_response, ensure_ascii=False),
            "endpoint": client.endpoint_generate,
        }
    except SeedanceClientError as error:
        return client_error_result(error)
    except Exception as error:  # Keep the probe resilient across all modes.
        return error_result(str(error), request_payload={"capability": capability, "prompt": prompt})


def build_client_from_env() -> SeedanceClient:
    return SeedanceClient(
        api_key=os.getenv("SEEDANCE_API_KEY", ""),
        api_base=os.getenv("SEEDANCE_API_BASE", ""),
        model=os.getenv("SEEDANCE_MODEL", "doubao-seedance-1-0-pro-fast-251015"),
        endpoint_generate=os.getenv("SEEDANCE_ENDPOINT_GENERATE", ""),
        endpoint_status=os.getenv("SEEDANCE_ENDPOINT_STATUS", ""),
        endpoint_upload=os.getenv("SEEDANCE_ENDPOINT_UPLOAD", ""),
    )


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "text_to_video": empty_result(),
        "image_to_video": empty_result(),
        "image_audio_to_talking_head": empty_result(),
        "video_audio_lipsync": empty_result(),
    }

    api_key_present = bool((os.getenv("SEEDANCE_API_KEY") or "").strip())
    api_base_present = bool((os.getenv("SEEDANCE_API_BASE") or "").strip())
    print(f"[env] SEEDANCE_API_KEY configured: {api_key_present}")
    print(f"[env] SEEDANCE_API_BASE configured: {api_base_present}")
    print(f"[env] SEEDANCE_MODEL: {(os.getenv('SEEDANCE_MODEL') or 'doubao-seedance-1-0-pro-fast-251015').strip()}")
    print(f"[env] SEEDANCE_ENDPOINT_GENERATE configured: {bool((os.getenv('SEEDANCE_ENDPOINT_GENERATE') or '').strip())}")
    print(f"[env] SEEDANCE_ENDPOINT_STATUS configured: {bool((os.getenv('SEEDANCE_ENDPOINT_STATUS') or '').strip())}")
    print(f"[env] SEEDANCE_ENDPOINT_UPLOAD configured: {bool((os.getenv('SEEDANCE_ENDPOINT_UPLOAD') or '').strip())}")

    materials = find_materials()
    print_materials(materials)

    missing_env = []
    if not api_key_present:
        missing_env.append("SEEDANCE_API_KEY")
    if not api_base_present:
        missing_env.append("SEEDANCE_API_BASE")
    if missing_env:
        error = SeedanceConfigurationError(
            "Missing required Seedance env vars: " + ", ".join(missing_env),
            request_payload={"missing_env": missing_env},
        )
        for key in results:
            results[key] = client_error_result(error)
        write_results(results)
        print(f"[result] Seedance config is incomplete. Wrote {RESULT_PATH}")
        return 1

    try:
        client = build_client_from_env()
    except SeedanceClientError as error:
        for key in results:
            results[key] = client_error_result(error)
        write_results(results)
        print(f"[result] Seedance client setup failed. Wrote {RESULT_PATH}")
        return 1

    probes = [
        (
            "text_to_video",
            {"prompt": PROMPT},
        ),
        (
            "image_to_video",
            {"prompt": PROMPT, "assets": {"avatar": materials["avatar"]}},
        ),
        (
            "image_audio_to_talking_head",
            {"assets": {"avatar": materials["avatar"], "audio": materials["audio"]}},
        ),
        (
            "video_audio_lipsync",
            {"assets": {"video": materials["short_video"], "audio": materials["audio"]}},
        ),
    ]

    for capability, kwargs in probes:
        print(f"[run] {capability}")
        results[capability] = run_probe(client, capability, **kwargs)
        write_results(results)
        if results[capability]["supported"]:
            print(f"[ok] {capability}: supported")
        else:
            print(f"[fail] {capability}: {results[capability]['error']}")

    print(f"[result] wrote {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
