"""Probe Seedance text-to-video model ids.

Dry run:
    python scripts/probe_seedance_models.py

Submit real tasks:
    python scripts/probe_seedance_models.py --submit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from services.seedance_client import SEEDANCE_MODELS, SeedanceClient, SeedanceClientError  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "seedance_model_probe"
RESULT_PATH = OUTPUT_DIR / "result.json"

CANDIDATE_MODELS = [
    SEEDANCE_MODELS["t2v_2_0"],
    SEEDANCE_MODELS["t2v_2_0_fast"],
    SEEDANCE_MODELS["avatar_fast"],
]

PROMPT = (
    "A clean professional presenter in a small studio, natural blinking, "
    "slight head movement, stable camera."
)


def build_client(model: str) -> SeedanceClient:
    return SeedanceClient(
        api_key=os.getenv("SEEDANCE_API_KEY", ""),
        api_base=os.getenv("SEEDANCE_API_BASE", "https://ark.cn-beijing.volces.com"),
        model=model,
        endpoint_generate=os.getenv("SEEDANCE_ENDPOINT_GENERATE", "contents/generations/tasks"),
        endpoint_status=os.getenv("SEEDANCE_ENDPOINT_STATUS", "contents/generations/tasks/{task_id}"),
        endpoint_upload=os.getenv("SEEDANCE_ENDPOINT_UPLOAD", ""),
        max_poll_seconds=30,
    )


def result_from_error(model: str, error: SeedanceClientError, payload: dict) -> dict:
    detail = error.to_result_error()
    return {
        "model": model,
        "submitted": True,
        "supported": False,
        "stage": detail.get("stage") or "create_task",
        "status_code": detail.get("status_code"),
        "error_code": detail.get("error_code"),
        "error_message": detail.get("error_message"),
        "response_text": detail.get("response_text"),
        "response_json": detail.get("response_json"),
        "task_id": None,
        "payload_sanitized": detail.get("request_payload_sanitized") or payload,
        "endpoint": detail.get("endpoint"),
    }


def probe_model(model: str, submit: bool) -> dict:
    client = build_client(model)
    payload = client.build_task_payload(
        "text_to_video",
        prompt=PROMPT,
        uploaded_assets={},
        duration=4,
        resolution="720p",
        ratio="16:9",
    )
    payload_sanitized = client.sanitize_payload(payload)
    if not submit:
        return {
            "model": model,
            "submitted": False,
            "supported": None,
            "stage": "dry_run",
            "status_code": None,
            "error_code": None,
            "error_message": None,
            "response_text": None,
            "task_id": None,
            "payload_sanitized": payload_sanitized,
            "endpoint": client._url(client.endpoint_generate),
        }

    try:
        response = client.create_task(
            "text_to_video",
            prompt=PROMPT,
            uploaded_assets={},
            duration=4,
            resolution="720p",
            ratio="16:9",
        )
        task_id = client.extract_task_id(response)
        poll_once = None
        if task_id:
            try:
                poll_once = client._get_json(
                    client._status_endpoint(client.endpoint_status, task_id),
                    request_payload={"task_id": task_id},
                    stage="poll_task",
                )
            except SeedanceClientError as poll_error:
                poll_once = {"error": poll_error.to_result_error()}
        return {
            "model": model,
            "submitted": True,
            "supported": bool(task_id),
            "stage": "create_task",
            "status_code": 200,
            "error_code": None,
            "error_message": None,
            "response_text": None,
            "response_json": response,
            "task_id": task_id,
            "poll_once": poll_once,
            "payload_sanitized": payload_sanitized,
            "endpoint": client._url(client.endpoint_generate),
        }
    except SeedanceClientError as error:
        return result_from_error(model, error, payload_sanitized)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="actually create Seedance probe tasks")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.submit:
        print("该脚本会向 Seedance 创建最小测试任务，可能产生费用。继续请加 --submit。")

    results = [probe_model(model, args.submit) for model in CANDIDATE_MODELS]
    first_client = build_client(CANDIDATE_MODELS[0])
    payload = {
        "base_url": first_client.api_base,
        "create_endpoint": first_client._url(first_client.endpoint_generate),
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "submitted": args.submit,
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
