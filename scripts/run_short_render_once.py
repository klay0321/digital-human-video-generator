import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import generate_videos_from_plan
from services.digital_human import DIGITAL_HUMAN_MODE_STATIC
from services.layout_engine import LayoutConfig


def main():
    load_dotenv(Path(".env"), override=True)
    result = generate_videos_from_plan(
        project_dir="outputs/20260521_204130_c36403",
        video_name="demo.mp4",
        avatar_name="avatar.jpg",
        audio_mode="使用已有 ElevenLabs voice_id 替换原声",
        layout_config=LayoutConfig(),
        voice_sample_paths=[],
        voice_name="short_test_voice",
        remove_background_noise=False,
        voice_consent=True,
        voice_style="专业讲解",
        render_plan_path="outputs/20260521_204130_c36403/selected_render_plan.short_test.json",
        digital_human_provider=DIGITAL_HUMAN_MODE_STATIC,
        force_regenerate_seedance_avatar=False,
        seedance_quality_mode="fast",
        digital_human_window_style="card",
        digital_human_video_mode="preview_loop",
        fallback_experimental_i2v_to_fast=False,
    )
    print(json.dumps({
        "errors": result.get("errors"),
        "warnings": result.get("warnings"),
        "audio_mode": result.get("audio_mode"),
        "tts_status": result.get("tts_status"),
        "clips_count": len(result.get("clips") or []),
        "clips": [
            {
                "clip_id": c.get("clip_id"),
                "kp_id": c.get("kp_id"),
                "final_video": c.get("final_video"),
                "voice_audio": c.get("voice_audio"),
                "srt": c.get("srt"),
                "tts_text_source": c.get("tts_text_source"),
                "subtitle_text_source": c.get("subtitle_text_source"),
                "subtitle_uses_same_text_as_tts": c.get("subtitle_uses_same_text_as_tts"),
                "subtitle_duration_source": c.get("subtitle_duration_source"),
            }
            for c in result.get("clips", [])
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
