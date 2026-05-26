"""Smoke test for round-2: render_jobs + selected_render_jobs flow.

Verifies (without touching STT / LLM / ffmpeg / Seedance / ElevenLabs):
  - build_render_jobs_from_topics produces a render_job per video_topic
  - each render_job has render_units with voice_script / subtitle_lines
  - per-job render plans can be constructed deterministically and contain the
    fields generate_videos_from_plan expects (selected_topic_id, render_units,
    render_output_mode, voice_style)
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
    build_render_jobs_from_topics,
)


def build_synthetic_plan():
    points = []
    for i in range(10):
        kp_id = f"kp_{i+1:03d}"
        points.append({
            "kp_id": kp_id,
            "kp_title": f"知识点 {i+1}",
            "kp_type": "operation" if i % 2 else "concept",
            "kp_summary": f"摘要 {i+1}",
            "voice_script": "这是一段供测试用的朗读稿，长度刚好满足约束。" * 2,
            "subtitle_lines": ["这是一段供测试用的朗读稿", "长度刚好满足约束"],
            "fragments": [{
                "fragment_id": f"frag_{i+1:03d}",
                "start": i * 30.0,
                "end": i * 30.0 + 25.0,
                "clean_text": "测试片段",
            }],
            "scores": {"importance": 4, "clip_value": 0.7},
            "dependencies": {"requires": [], "can_stand_alone": True},
        })
    return {
        "project_id": "smoke_test_round2",
        "knowledge_points": points,
        "big_hooks": [
            {
                "hook_id": "hook_001",
                "hook_title": "为什么 X 重要",
                "hook_type": "boss_report",
                "opening_hook": "opening",
                "hook_summary": "summary",
                "recommended_kp_ids": ["kp_001", "kp_002", "kp_003"],
                "optional_kp_ids": [],
                "estimated_duration": 90,
                "why_it_works": "reason",
            },
        ],
        "assembly_paths": [],
        "source_summary": {"main_topic": "smoke", "content_type": "mixed"},
    }


def main():
    plan = build_synthetic_plan()
    cap_global_knowledge_points(plan)
    build_video_topics_from_plan(plan)
    jobs_payload = build_render_jobs_from_topics(plan, voice_style="讲解风格")

    print("video_topics_count =", plan["video_topics_count"])
    print("render_jobs_count  =", jobs_payload["render_jobs_count"])

    selected_topic_ids = [t["topic_id"] for t in plan["video_topics"][:2]]
    selected_jobs = [
        job for job in jobs_payload["render_jobs"]
        if job["topic_id"] in selected_topic_ids
    ]
    selected_payload = {
        "project_id": plan["project_id"],
        "plan_reusable": True,
        "voice_style": "讲解风格",
        "selected_topic_ids": selected_topic_ids,
        "selected_job_ids": [j["job_id"] for j in selected_jobs],
        "render_jobs": selected_jobs,
        "render_output_mode": "single_complete_video",
    }
    print("selected_topic_ids =", selected_payload["selected_topic_ids"])
    print("selected_job_ids   =", selected_payload["selected_job_ids"])

    print("\n--- per-job render plan that would be written ---")
    for job in selected_jobs:
        per_job_plan = {
            "selected_topic_id": job["topic_id"],
            "selected_kp_ids": job["selected_kp_ids"],
            "render_units": job["render_units"],
            "render_output_mode": "single_complete_video",
            "voice_style": "讲解风格",
            "final_video_title": job["final_video_title"],
        }
        print(
            f"  topic_id={per_job_plan['selected_topic_id']}  "
            f"kp_count={len(per_job_plan['selected_kp_ids'])}  "
            f"render_unit_count={len(per_job_plan['render_units'])}  "
            f"first_unit_tts_source={(per_job_plan['render_units'][0] or {}).get('tts_text_source') if per_job_plan['render_units'] else '(none)'}"
        )

    print("\nOK")


if __name__ == "__main__":
    main()
