import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from services.llm_client import call_llm_json


def main():
    prompt = """
请把下面这句话清洗成更适合口播的表达，并只返回 JSON：

输入：嗯这个工具其实就是帮我们自动生成短视频。

返回格式：
{
  "cleaned_text": "..."
}
""".strip()
    result = call_llm_json(prompt, stage="llm_clean")
    parsed = result.get("result") if isinstance(result.get("result"), dict) else {}
    output = {
        "success": bool(result.get("success")),
        "status_code": result.get("status_code"),
        "selected_key_source": result.get("selected_key_source"),
        "model": result.get("model"),
        "cleaned_text": parsed.get("cleaned_text"),
        "error_message": result.get("error_message") or result.get("error"),
        "response_text": result.get("response_text"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
