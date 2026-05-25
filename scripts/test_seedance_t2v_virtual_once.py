"""Generate one Seedance 2.0 text-to-video virtual presenter clip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from services.digital_human.seedance_t2v_provider import (  # noqa: E402
    DEFAULT_T2V_MODEL,
    SeedanceTextToVideoVirtualPresenterProvider,
)


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / "seedance_t2v_virtual_once"
    provider = SeedanceTextToVideoVirtualPresenterProvider(
        model=DEFAULT_T2V_MODEL,
        cache_root=output_dir / "cache",
        force_regenerate=True,
    )
    result = provider.create(
        voice_style="沉稳专业风格",
        voice_script=(
            "这是一个 Seedance 2.0 文生视频虚拟讲解人测试。"
            "请生成适合放在录屏右下角小窗的中文讲解人像。"
        ),
        target_duration=4,
        output_dir=str(output_dir),
        clip_id="seedance_t2v_virtual_once",
        model=DEFAULT_T2V_MODEL,
    )
    debug = result.debug or {}
    seedance_error = debug.get("seedance_error") or {}
    output_path = result.talking_head_video_path
    validation = provider.validate_video(output_path) if output_path else {
        "valid": False,
        "duration": None,
    }
    payload = {
        "success": bool(result.success),
        "model": debug.get("seedance_model_used") or debug.get("seedance_model_requested") or DEFAULT_T2V_MODEL,
        "output_path": output_path,
        "duration": validation.get("duration"),
        "error_code": seedance_error.get("error_code"),
        "error_message": seedance_error.get("error_message") or result.error,
        "response_text": seedance_error.get("response_text"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
