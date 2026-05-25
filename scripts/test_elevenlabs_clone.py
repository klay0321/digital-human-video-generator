"""ElevenLabs 声音克隆最小测试脚本。

用法：
    python scripts/test_elevenlabs_clone.py

行为：
    1. 从 .env 读取 ELEVENLABS_API_KEY
    2. 使用 test/test_01.m4a 作为唯一声音样本
    3. 调用 services.voice_clone_elevenlabs.clone_voice_with_elevenlabs
    4. 打印完整 JSON 结果
    5. 不调用 TTS
    6. 不调用视频生成
"""

import json
import os
import sys
from pathlib import Path

# 让脚本可以独立运行：把项目根加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from services.voice_clone_elevenlabs import clone_voice_with_elevenlabs


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    print(f"[env] ELEVENLABS_API_KEY 是否已配置: {bool(api_key)}")
    if api_key:
        # 不要打印明文 key，只显示长度
        print(f"[env] API key 长度: {len(api_key)}")

    sample_path = PROJECT_ROOT / "test" / "test_01.m4a"
    print(f"[sample] 路径: {sample_path}")
    print(f"[sample] 是否存在: {sample_path.exists()}")
    if sample_path.exists():
        try:
            print(f"[sample] 大小: {sample_path.stat().st_size} bytes")
        except Exception as e:
            print(f"[sample] 读取大小失败: {e}")

    if not sample_path.exists():
        print("[ERROR] 测试样本不存在，请确认 test/test_01.m4a 存在")
        return 1

    print("\n[run] 调用 clone_voice_with_elevenlabs ...")
    try:
        result = clone_voice_with_elevenlabs(
            sample_paths=[str(sample_path)],
            voice_name="test_clone_demo",
            description="ElevenLabs clone smoke test",
        )
    except Exception as e:
        print(f"[FATAL] clone_voice_with_elevenlabs 抛出异常: {e}")
        return 2

    print("\n[result] 完整返回 JSON：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("success"):
        print(f"\n[OK] 克隆成功，voice_id = {result.get('voice_id')}")
        return 0
    else:
        print(f"\n[FAIL] 克隆失败：{result.get('error')}")
        debug = result.get("debug") or {}
        print("[debug] first_field_name =", debug.get("first_field_name"))
        print("[debug] first_status_code =", debug.get("first_status_code"))
        print("[debug] retry_field_name =", debug.get("retry_field_name"))
        print("[debug] retry_status_code =", debug.get("retry_status_code"))
        print("[debug] file_sizes =", debug.get("file_sizes"))
        print("[debug] content_type_header =", debug.get("content_type_header"))
        return 3


if __name__ == "__main__":
    sys.exit(main())
