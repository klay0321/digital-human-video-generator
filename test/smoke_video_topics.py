"""Smoke test for the new cap + video_topics pipeline.

This script does NOT touch real STT / LLM / Seedance / ElevenLabs.
It only exercises cap_global_knowledge_points and build_video_topics_from_plan
on a synthetic plan to confirm the new functions behave as designed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.knowledge_planner import (
    cap_global_knowledge_points,
    build_video_topics_from_plan,
)


def build_synthetic_plan():
    points = []
    for i in range(25):
        kp_id = f"kp_{i+1:03d}"
        importance = (i % 5) + 1
        points.append({
            "kp_id": kp_id,
            "kp_title": f"测试知识点 {i+1}",
            "kp_type": "concept" if i % 2 == 0 else "operation",
            "kp_summary": f"摘要 {i+1}",
            "voice_script": "这是一个测试用的朗读稿。" * 3,
            "subtitle_lines": ["这是一个测试用的朗读稿", "内容只是为了测试"],
            "fragments": [{
                "fragment_id": f"frag_{i+1:03d}",
                "start": i * 30.0,
                "end": i * 30.0 + 25.0,
                "clean_text": "测试片段",
            }],
            "scores": {
                "importance": importance,
                "clip_value": 0.5 + (i % 3) * 0.1,
            },
            "dependencies": {"requires": [], "can_stand_alone": True},
        })

    # 3 super-short kp (5s) should be filtered out by min_duration_seconds=15.
    for i in range(3):
        kp_id = f"kp_short_{i+1:03d}"
        points.append({
            "kp_id": kp_id,
            "kp_title": f"超短知识点 {i+1}",
            "kp_type": "transition",
            "voice_script": "过场。",
            "subtitle_lines": ["过场"],
            "fragments": [{
                "fragment_id": f"short_{i+1:03d}",
                "start": 1000 + i * 10,
                "end": 1005 + i * 10,
                "clean_text": "过场",
            }],
            "scores": {"importance": 5, "clip_value": 0.9},
            "dependencies": {"requires": []},
        })

    return {
        "knowledge_points": points,
        "big_hooks": [
            {
                "hook_id": "hook_001",
                "hook_title": "为什么 X 真正重要",
                "hook_type": "boss_report",
                "opening_hook": "opening",
                "hook_summary": "summary",
                "recommended_kp_ids": ["kp_001", "kp_002", "kp_003"],
                "optional_kp_ids": [],
                "estimated_duration": 90,
                "why_it_works": "reason",
            },
            {
                "hook_id": "hook_002",
                "hook_title": "从概念到操作 X",
                "hook_type": "tutorial",
                "opening_hook": "opening",
                "hook_summary": "summary",
                "recommended_kp_ids": ["kp_004", "kp_005", "kp_006", "kp_007"],
                "optional_kp_ids": [],
                "estimated_duration": 120,
            },
        ],
        "assembly_paths": [],
        "source_summary": {"main_topic": "test topic", "content_type": "mixed"},
    }


def main():
    plan = build_synthetic_plan()
    cap_global_knowledge_points(plan)
    build_video_topics_from_plan(plan)

    print("total_points =", len(plan["knowledge_points"]))
    print("user_visible_kp_ids =", len(plan["user_visible_kp_ids"]))
    print("advanced_kp_ids    =", len(plan["advanced_kp_ids"]))
    print("kp_cap_summary     =", json.dumps(plan["kp_cap_summary"], ensure_ascii=False))
    print("video_topics_count =", plan["video_topics_count"])
    for t in plan["video_topics"]:
        line = (
            f"  - {t['topic_id']}  "
            f"{t['topic_title'][:24]}  "
            f"type={t['video_type']}  "
            f"kps={len(t['recommended_kp_ids'])}  "
            f"dur={t['estimated_duration']}s  "
            f"auto={t.get('is_auto_supplemented', False)}"
        )
        print(line)
    last_short = plan["knowledge_points"][-1]
    print("short kp visibility:", last_short.get("visibility"), "reason=", last_short.get("advanced_reason"))


if __name__ == "__main__":
    main()
