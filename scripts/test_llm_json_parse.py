"""extract_json_object 鲁棒性测试。

不调用 LLM、不依赖网络。stub openai 后直接对 services.llm_client.extract_json_object
喂各种"可能的 LLM 真实输出"看能否提出 JSON。

Usage:
    python -m compileall .
    python scripts/test_llm_json_parse.py
"""
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub openai so the import is offline-safe.
_fake_openai = types.ModuleType("openai")
class _FakeOpenAI:
    def __init__(self, *a, **k):
        pass
_fake_openai.OpenAI = _FakeOpenAI
sys.modules.setdefault("openai", _fake_openai)

from services.llm_client import extract_json_object  # noqa: E402

CASES = [
    # 1. Pure JSON.
    (
        "pure JSON",
        '{"cleaned_text": "abc", "clips": [{"clip_id": "clip_001"}]}',
        True,
    ),
    # 2. Markdown ```json fence.
    (
        "fenced ```json",
        '```json\n{"cleaned_text": "a", "clips": []}\n```',
        True,
    ),
    # 3. Plain ``` fence (no lang).
    (
        "fenced ``` (no lang)",
        '```\n{"cleaned_text": "b", "clips": [1, 2]}\n```',
        True,
    ),
    # 4. Prefix + suffix text.
    (
        "prefix + JSON + suffix",
        '好的，下面是结果：\n{"cleaned_text": "ok", "clips": []}\n希望对你有帮助。',
        True,
    ),
    # 5. Trailing comma after the object — invalid JSON, but the first '{...}'
    #    walks to a balanced close before that.
    (
        "balanced { ... } inside garbage",
        'noise{"clips": []}more noise',
        True,
    ),
    # 6. Nested braces (the regex-naive approach would have failed here).
    (
        "nested braces",
        '{"cleaned_text": "x", "clips": [{"clip_id": "c1", "meta": {"a": 1, "b": {"c": 2}}}]}',
        True,
    ),
    # 7. JSON inside two fenced blocks, first one empty.
    (
        "second fence has the real JSON",
        '```\n\n```\n```json\n{"cleaned_text": "y", "clips": []}\n```',
        True,
    ),
    # 8. Quoted braces inside a string field shouldn't break the balancer.
    (
        "quoted brace in string",
        '{"cleaned_text": "a{b}c", "clips": []}',
        True,
    ),
    # 9. Total garbage.
    (
        "garbage",
        "hello world, nothing here",
        False,
    ),
    # 10. Empty string.
    ("empty", "", False),
    # 11. None.
    ("None input", None, False),
    # 12. A bare JSON array, not an object.
    (
        "JSON array (not object)",
        '[1, 2, 3]',
        False,
    ),
    # 13. Object preceded by an array (we must skip the array and find the obj).
    (
        "array then object",
        '[1, 2] {"cleaned_text": "ok", "clips": []}',
        True,
    ),
    # 14. Prompt echo — extract_json_object should still parse it; LEAK
    #     detection happens elsewhere.
    (
        "valid JSON containing leak text",
        '{"cleaned_text": "你是一个录屏讲解短视频策划助手", "clips": []}',
        True,
    ),
]


def main():
    failed = 0
    for name, inp, expect_ok in CASES:
        got = extract_json_object(inp)
        ok = isinstance(got, dict)
        passed = (ok == expect_ok)
        tag = "[OK]  " if passed else "[FAIL]"
        print(f"{tag} {name:<40}  expect={expect_ok}  got={ok}")
        if not passed:
            print(f"        input: {inp!r}")
            print(f"        parsed: {got!r}")
            failed += 1
    if failed:
        print(f"\n{failed} case(s) failed.")
        sys.exit(1)
    print("\n=== extract_json_object 全部 14 个用例通过 ===")


if __name__ == "__main__":
    main()
