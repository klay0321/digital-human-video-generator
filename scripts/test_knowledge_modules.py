"""End-to-end smoke test for knowledge-module clip planning (offline).

No real LLM / STT / FFmpeg required. The openai SDK is stubbed and
call_llm_json is monkey-patched to simulate LLM failure so we exercise the
deterministic fallback path.

Asserts:
  1. clip_plan_mode="知识点模块切片" runs without raising
  2. project_dir contains cleaned_segments.json, knowledge_modules.json,
     selected_render_plan.json, clips.json, modules/*.md
  3. Each clip has `fragments`; MCP module has multiple non-contiguous frags
  4. fragment_id is globally unique
  5. build_selected_render_plan picks only the chosen fragments — the gap
     between two MCP fragments is never included
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub openai
_fake_openai = types.ModuleType("openai")
class _FakeOpenAI:
    def __init__(self, *a, **k): pass
_fake_openai.OpenAI = _FakeOpenAI
sys.modules.setdefault("openai", _fake_openai)

import pipeline
from services import knowledge_planner as kp

# Force LLM to fail so we exercise the deterministic fallback
def _failing_call(prompt):
    return {"result": None, "raw_response": "", "error": "simulated LLM failure", "model": "fake"}
pipeline.call_llm_json = _failing_call


def check(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    print(f"[OK]  {msg}")


def main():
    # Simulate STT segments (MCP — Agent — Demo — MCP again, with fillers)
    fake_segments = [
        {"start": 0.0,   "end": 12.0,  "text": "嗯大家好，今天我们讲一下 MCP 是什么。MCP 是一种协议。"},
        {"start": 12.0,  "end": 25.0,  "text": "MCP 主要用来连接工具调用，让 Agent 能用外部工具。"},
        {"start": 25.0,  "end": 40.0,  "text": "好那个我们再聊 Agent。Agent 是一种自主决策的程序。"},
        {"start": 40.0,  "end": 55.0,  "text": "Agent 的工作流包括感知、推理、执行三步。"},
        {"start": 55.0,  "end": 70.0,  "text": "演示一下，我们打开终端运行 demo 代码。"},
        {"start": 70.0,  "end": 82.0,  "text": "Demo 的运行结果就是这样，可以看到 Agent 调用了 MCP 工具。"},
        {"start": 82.0,  "end": 95.0,  "text": "回到 MCP 总结一下，MCP 让 Agent 有了标准的工具接口。"},
        {"start": 95.0,  "end": 105.0, "text": "好就这样，谢谢大家。"},
    ]
    transcript = "".join(s["text"] for s in fake_segments)

    # Inject a fake STT response so input_mode="auto" returns multi-segments
    # without actually running ffmpeg / hitting ElevenLabs.
    def fake_get_transcript(input_mode, video_path, manual_text=None, transcript_file_text=None):
        return {
            "source": "fake_stt",
            "language": "zh",
            "full_text": transcript,
            "segments": fake_segments,
        }
    pipeline.get_transcript_from_source = fake_get_transcript

    # Read prompt templates from the real prompts dir
    prompts_dir = PROJECT_ROOT / "prompts"
    timeline_prompt = (prompts_dir / "clean_and_plan.md").read_text(encoding="utf-8")

    fake_video = b"\x00" * 1024
    fake_avatar = b"\x00" * 1024

    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        Path("outputs").mkdir()
        Path("prompts").mkdir()
        # Mirror prompts so pipeline can read them via prompts_dir="prompts"
        for name in ("clean_and_plan.md", "clean_segments.md", "plan_knowledge_modules.md"):
            (Path("prompts") / name).write_text(
                (prompts_dir / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        result = pipeline.generate_clips_plan(
            video_bytes=fake_video,
            video_name="test.mp4",
            avatar_bytes=fake_avatar,
            avatar_name="avatar.jpg",
            input_mode="auto",
            prompt_template=timeline_prompt,
            clip_plan_mode="知识点模块切片",
            voice_style="讲解风格",
            prompts_dir="prompts",
        )

        check(not result["errors"], f"no fatal errors  (errors={result['errors']})")
        check(result["clip_plan_mode"] == "知识点模块切片", "clip_plan_mode 正确")
        check(result["used_fallback_clips"] is True, "标记为 fallback（LLM 已被强制失败）")

        proj_dir = Path(result["project_dir"])
        check((proj_dir / "cleaned_segments.json").exists(), "cleaned_segments.json 已落盘")
        check((proj_dir / "knowledge_modules.json").exists(), "knowledge_modules.json 已落盘")
        check((proj_dir / "selected_render_plan.json").exists(), "selected_render_plan.json 已落盘（默认全选）")
        check((proj_dir / "clips.json").exists(), "clips.json 已落盘")
        modules_dir = proj_dir / "modules"
        check(modules_dir.exists() and any(modules_dir.glob("module_*.md")),
              "modules/*.md 已落盘")

        # Validate clips structure
        clips = result["clips"]
        check(len(clips) >= 1, "至少 1 个 clip")
        for c in clips:
            check("fragments" in c, f"{c['clip_id']} 含 fragments 字段")
            check(c["render_mode"] == "concat_fragments",
                  f"{c['clip_id']} render_mode=concat_fragments")

        # fragment_id 全局唯一
        all_fids = [f["fragment_id"]
                    for c in clips for f in (c.get("fragments") or [])]
        check(len(all_fids) == len(set(all_fids)), "fragment_id 全局唯一")

        # MCP module 应包含 >=2 个不连续 fragments
        mcp_clip = next((c for c in clips if c.get("topic_key") == "mcp"), None)
        check(mcp_clip is not None, "存在 MCP clip")
        mcp_frags = mcp_clip.get("fragments") or []
        check(len(mcp_frags) >= 2, "MCP 含 >=2 个 fragment")
        gap = mcp_frags[1]["start"] - mcp_frags[0]["end"]
        check(gap > 0, f"MCP 两段间不连续 (gap={gap}s)")

        # 关键：clip.start/end 是 advisory；不能依赖它做单次裁剪
        span = mcp_clip["end"] - mcp_clip["start"]
        total = sum(f["end"] - f["start"] for f in mcp_frags)
        check(span > total, f"span ({span}s) > sum_of_fragments ({total}s) —— "
                            "证明渲染必须按 fragment 分别裁剪，不能拉成连续大区间")

        # build_selected_render_plan：只选 MCP，确认中间 Agent/Demo 不会出现
        modules = result["knowledge_modules"]
        plan = kp.build_selected_render_plan(
            selected_module_ids=[mcp_clip["module_id"]],
            selected_fragment_ids=[],
            knowledge_modules=modules,
            fragment_order="source_time",
            voice_style="讲解风格",
        )
        picked = plan["render_jobs"][0]["fragments"]
        check(len(picked) == len(mcp_frags),
              f"渲染计划只挑 MCP 的 {len(mcp_frags)} 个 fragment")
        for f in picked:
            for c in clips:
                if c.get("topic_key") != "mcp":
                    for other in c.get("fragments") or []:
                        check(f["fragment_id"] != other["fragment_id"],
                              f"{f['fragment_id']} 不在 {c['topic_key']} 模块中")

        # debug 文件齐全
        dbg = proj_dir / "debug"
        for name in ("debug_transcript_full_text.txt",
                     "debug_transcript_segments.json",
                     "debug_clean_segments_prompt.md",
                     "debug_clean_segments_response.txt",
                     "debug_knowledge_modules_prompt.md",
                     "debug_knowledge_modules_response.txt",
                     "debug_pipeline_warnings.json"):
            check((dbg / name).exists(), f"debug/{name} 已落盘")

        # 没有 prompt 漏到任何用户可见字段
        for f in all_fids:
            pass
        for c in clips:
            for k in ("source_text", "cleaned_clip_text", "voice_script",
                      "title", "hook", "summary"):
                v = (c.get(k) or "")
                check(not pipeline.looks_like_prompt_leak(v),
                      f"{c['clip_id']}.{k} 无 prompt 泄漏")
            for f in c.get("fragments") or []:
                for k in ("source_text", "cleaned_text"):
                    v = (f.get(k) or "")
                    check(not pipeline.looks_like_prompt_leak(v),
                          f"{f['fragment_id']}.{k} 无 prompt 泄漏")

    print("\n=== knowledge_modules end-to-end fallback 全部通过 ===")


if __name__ == "__main__":
    main()
