"""Minimal smoke test for prompt-leak immunity in the plan pipeline.

Verifies:
  1. looks_like_prompt_leak flags prompt-like text but leaves normal STT text alone.
  2. build_fallback_clips_from_segments produces clips whose source_text,
     cleaned_clip_text and voice_script contain ONLY transcript content —
     never any prompt-template marker.
  3. Length-based fallback (no real timestamps) also produces leak-free clips.
  4. validate_plan_result accepts a clean fallback plan and rejects a leaky one.

Runs WITHOUT real LLM or STT. The openai SDK is stubbed so pipeline can be
imported in offline test environments.

Usage:
    python -m compileall .
    python scripts/test_generate_plan_no_prompt_leak.py
"""
import json
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub openai for offline import.
_fake_openai = types.ModuleType("openai")
class _FakeOpenAI:
    def __init__(self, *a, **k):
        pass
_fake_openai.OpenAI = _FakeOpenAI
sys.modules.setdefault("openai", _fake_openai)

from pipeline import (  # noqa: E402
    looks_like_prompt_leak,
    build_fallback_clips_from_segments,
    validate_plan_result,
)

PROMPT_FRAGMENT_FROM_TEMPLATE = (
    "你是一个录屏讲解短视频策划助手——请输出 JSON。\n"
    "字段包括 source_segment_indexes、voice_script、cleaned_text。\n"
    "不要凭空编造时间戳。请返回合法 JSON。"
)

TRANSCRIPT_FULL_TEXT = (
    "大家好，今天我们演示如何用 AI 自动剪辑录屏视频。"
    "第一步上传视频，第二步自动转写，第三步生成短视频。"
)

TRANSCRIPT_SEGMENTS = [
    {"start": 0, "end": 10, "text": "大家好，今天我们演示如何用 AI 自动剪辑录屏视频。"},
    {"start": 10, "end": 20, "text": "第一步上传视频，第二步自动转写，第三步生成短视频。"},
]


def check(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    print(f"[OK] {msg}")


def main():
    # 1. looks_like_prompt_leak basic behaviour.
    check(looks_like_prompt_leak(PROMPT_FRAGMENT_FROM_TEMPLATE),
          "looks_like_prompt_leak 正确识别 prompt 片段")
    check(not looks_like_prompt_leak(TRANSCRIPT_FULL_TEXT),
          "looks_like_prompt_leak 不误伤正常 STT 文本")
    check(not looks_like_prompt_leak(""),
          "looks_like_prompt_leak 对空字符串返回 False")
    check(not looks_like_prompt_leak(None),
          "looks_like_prompt_leak 对 None 返回 False")

    # 2. Fallback with real timestamps.
    clips_ts = build_fallback_clips_from_segments(TRANSCRIPT_SEGMENTS, TRANSCRIPT_FULL_TEXT)
    check(len(clips_ts) >= 1, "有时间戳时至少生成 1 段")
    for c in clips_ts:
        check("source_segment_indexes" not in c["source_text"],
              f"{c['clip_id']} source_text 不含 source_segment_indexes 字样")
        check("你是一个录屏讲解短视频策划助手" not in c["voice_script"],
              f"{c['clip_id']} voice_script 不含 prompt 开头")
        check("{{FULL_TEXT}}" not in c["cleaned_clip_text"],
              f"{c['clip_id']} cleaned_clip_text 不含 {{FULL_TEXT}} 占位符")
        check("输出 JSON" not in c["voice_script"],
              f"{c['clip_id']} voice_script 不含 输出 JSON 字样")
        check(not looks_like_prompt_leak(c["source_text"]),
              f"{c['clip_id']} source_text 通过 leak 检查")
        check(not looks_like_prompt_leak(c["voice_script"]),
              f"{c['clip_id']} voice_script 通过 leak 检查")
        check(c["end"] > c["start"], f"{c['clip_id']} end > start")

    # 3. Length-based fallback (no real timestamps).
    clips_len = build_fallback_clips_from_segments(
        [{"start": 0, "end": 0, "text": TRANSCRIPT_FULL_TEXT}],
        TRANSCRIPT_FULL_TEXT,
    )
    check(len(clips_len) >= 1, "无真实时间戳时至少生成 1 段")
    for c in clips_len:
        check(c.get("estimated_timestamps") is True,
              f"{c['clip_id']} 标记 estimated_timestamps=True")
        check(not looks_like_prompt_leak(c["source_text"]),
              f"{c['clip_id']} (估算) source_text 通过 leak 检查")
        check(not looks_like_prompt_leak(c["voice_script"]),
              f"{c['clip_id']} (估算) voice_script 通过 leak 检查")
        check(c["end"] > c["start"],
              f"{c['clip_id']} (估算) end > start")

    # 4. validate_plan_result accepts a clean plan.
    clean_plan = {
        "transcript_full_text": TRANSCRIPT_FULL_TEXT,
        "full_text": TRANSCRIPT_FULL_TEXT,
        "cleaned_text": TRANSCRIPT_FULL_TEXT,
        "clips": clips_ts,
    }
    validate_plan_result(clean_plan)
    print("[OK] validate_plan_result 接受合规 plan")

    # 5. validate_plan_result rejects a leaky cleaned_text.
    leaky_plan = dict(clean_plan)
    leaky_plan["cleaned_text"] = PROMPT_FRAGMENT_FROM_TEMPLATE
    try:
        validate_plan_result(leaky_plan)
        print("[FAIL] validate_plan_result 应拒绝 leaky cleaned_text")
        sys.exit(1)
    except ValueError as e:
        check("检测到提示词模板" in str(e),
              "validate_plan_result 用规范错误信息拒绝 leaky cleaned_text")

    # 6. validate_plan_result rejects a leaky source_text.
    poisoned = json.loads(json.dumps(clips_ts))
    poisoned[0]["source_text"] = PROMPT_FRAGMENT_FROM_TEMPLATE
    leaky_plan2 = dict(clean_plan)
    leaky_plan2["clips"] = poisoned
    try:
        validate_plan_result(leaky_plan2)
        print("[FAIL] validate_plan_result 应拒绝 leaky source_text")
        sys.exit(1)
    except ValueError:
        print("[OK] validate_plan_result 拒绝 leaky clip.source_text")

    # 7. Empty transcript rejected.
    try:
        validate_plan_result({"transcript_full_text": "", "clips": clips_ts})
        print("[FAIL] validate_plan_result 应拒绝空 transcript")
        sys.exit(1)
    except ValueError:
        print("[OK] validate_plan_result 拒绝空 transcript")

    print("\n=== 全部 prompt-leak 测试通过 ===")


if __name__ == "__main__":
    main()
