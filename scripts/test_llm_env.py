import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from services.llm_client import get_llm_env_status


def main():
    status = get_llm_env_status()
    print(json.dumps({
        "has_llm_base_url": status["has_llm_base_url"],
        "has_llm_api_key": status["has_llm_api_key"],
        "has_ark_api_key": status["has_ark_api_key"],
        "has_volcengine_api_key": status["has_volcengine_api_key"],
        "selected_key_source": status["selected_key_source"],
        "key_length": status["key_length"],
        "llm_model": status["llm_model"],
        "llm_model_looks_like_seedance": status["llm_model_looks_like_seedance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
