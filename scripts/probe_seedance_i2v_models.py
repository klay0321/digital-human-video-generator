"""Probe Seedance image-to-video model ids.

Dry run:
    python scripts/probe_seedance_i2v_models.py

Submit real tasks:
    python scripts/probe_seedance_i2v_models.py --submit
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "seedance_i2v_model_probe"
RESULT_PATH = OUTPUT_DIR / "result.json"

CANDIDATE_MODELS = [
    SEEDANCE_MODELS["i2v_2_0"],
    SEEDANCE_MODELS["i2v_2_0_fast"],
    SEEDANCE_MODELS["avatar_fast"],
]

PROMPT = (
    "基于参考头像生成真实中文讲解人像，正面对镜头，自然眨眼，轻微点头，"
    "轻微口播式嘴部动作，背景简洁，画面稳定。"
)


def find_local_avatar() -> Path | None:
    test_dir = PROJECT_ROOT / "test"
    for pattern in ("*.jpg", "*.png", "*.jpeg"):
        matches = sorted(test_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


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


def uploaded_asset_for_probe(client: SeedanceClient):
    public_url = (os.getenv("SEEDANCE_TEST_IMAGE_URL") or "").strip()
    if public_url:
        return {
            "asset": {"url": public_url, "source": "public_url"},
            "public": True,
            "warning": None,
        }
    avatar = find_local_avatar()
    if not avatar:
        return {"asset": None, "public": False, "warning": "No jpg/png found under test/."}
    if not client.endpoint_upload:
        warning = "当前 image_to_video 需要公网 image_url，请配置 SEEDANCE_TEST_IMAGE_URL。未配置上传端点，本次将用 data URL 探测并记录接口真实错误。"
    else:
        warning = None
    return {
        "asset": client.upload_file(avatar, purpose="avatar"),
        "public": False,
        "warning": warning,
    }


def result_from_error(model: str, error: SeedanceClientError, payload: dict, public: bool) -> dict:
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
        "whether_image_url_was_public_url": public,
        "task_id": None,
        "payload_sanitized": detail.get("request_payload_sanitized") or payload,
        "endpoint": detail.get("endpoint"),
    }


def probe_model(model: str, submit: bool) -> dict:
    client = build_client(model)
    asset_info = uploaded_asset_for_probe(client)
    if not asset_info["asset"]:
        return {
            "model": model,
            "submitted": False,
            "supported": False,
            "stage": "prepare_asset",
            "status_code": None,
            "error_code": "MissingTestImage",
            "error_message": asset_info["warning"],
            "response_text": None,
            "whether_image_url_was_public_url": False,
            "task_id": None,
            "payload_sanitized": None,
            "endpoint": client._url(client.endpoint_generate),
        }
    payload = client.build_task_payload(
        "image_to_video",
        prompt=PROMPT,
        uploaded_assets={"avatar": asset_info["asset"]},
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
            "error_message": asset_info["warning"],
            "response_text": None,
            "whether_image_url_was_public_url": asset_info["public"],
            "task_id": None,
            "payload_sanitized": payload_sanitized,
            "endpoint": client._url(client.endpoint_generate),
        }
    try:
        response = client.create_task(
            "image_to_video",
            prompt=PROMPT,
            uploaded_assets={"avatar": asset_info["asset"]},
            duration=4,
            resolution="720p",
            ratio="16:9",
        )
        task_id = client.extract_task_id(response)
        return {
            "model": model,
            "submitted": True,
            "supported": bool(task_id),
            "stage": "create_task",
            "status_code": 200,
            "error_code": None,
            "error_message": asset_info["warning"],
            "response_text": None,
            "response_json": response,
            "whether_image_url_was_public_url": asset_info["public"],
            "task_id": task_id,
            "payload_sanitized": payload_sanitized,
            "endpoint": client._url(client.endpoint_generate),
        }
    except SeedanceClientError as error:
        result = result_from_error(model, error, payload_sanitized, asset_info["public"])
        if asset_info["warning"]:
            result["asset_warning"] = asset_info["warning"]
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="actually create Seedance i2v probe tasks")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.submit:
        print("该脚本会向 Seedance 创建最小测试任务，可能产生费用。继续请加 --submit。")
    if not (os.getenv("SEEDANCE_TEST_IMAGE_URL") or "").strip() and not (os.getenv("SEEDANCE_ENDPOINT_UPLOAD") or "").strip():
        print("当前 image_to_video 需要公网 image_url，请配置 SEEDANCE_TEST_IMAGE_URL。")

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
