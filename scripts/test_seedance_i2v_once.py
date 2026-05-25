"""Generate one Seedance image-to-video dynamic avatar loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from services.digital_human.seedance_i2v_provider import SeedanceImageToVideoProvider  # noqa: E402
from services.seedance_client import get_seedance_env_status  # noqa: E402


def find_avatar() -> Path | None:
    test_dir = PROJECT_ROOT / "test"
    for pattern in ("*.jpg", "*.png", "*.jpeg"):
        matches = sorted(test_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> int:
    avatar = find_avatar()
    if not avatar:
        result = {"success": False, "error": "No jpg/png avatar found under test/"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    env_status = get_seedance_env_status()
    provider = SeedanceImageToVideoProvider(force_regenerate=True)
    result = provider.create(
        avatar_image_path=str(avatar),
        style="讲解风格",
        voice_script="Seedance dynamic avatar smoke test.",
        clip_id="seedance_i2v_once",
        output_dir=str(PROJECT_ROOT / "outputs"),
    )
    debug = result.debug or {}
    output_path = result.talking_head_video_path
    validation = provider.validate_video(output_path) if output_path else {
        "valid": False,
        "size": 0,
        "duration": None,
    }
    payload = {
        "success": result.success,
        "output_path": output_path,
        "duration": validation.get("duration"),
        "size": validation.get("size"),
        "cache_hit": debug.get("seedance_cache_hit", debug.get("cache_hit")),
        "selected_key_source": env_status.get("selected_key_source"),
        "prompt": debug.get("seedance_prompt", debug.get("prompt")),
        "error": result.error,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
