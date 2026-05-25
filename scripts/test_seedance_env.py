"""Check Seedance/Ark environment variables without printing secrets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from services.seedance_client import get_seedance_env_status  # noqa: E402


def main() -> int:
    status = get_seedance_env_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status.get("selected_key_source"):
        return 1
    if not status.get("api_base"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
