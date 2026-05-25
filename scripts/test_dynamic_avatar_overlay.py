"""Minimal test for dynamic avatar video overlay.

Usage:
    python scripts/test_dynamic_avatar_overlay.py

It finds the newest outputs/seedance_cache/*/dynamic_avatar_loop.mp4 and
overlays it onto a test mp4, writing outputs/debug_dynamic_avatar_overlay.mp4.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.video_composer import compose_final_video  # noqa: E402


def newest_dynamic_avatar():
    cache_root = PROJECT_ROOT / "outputs" / "seedance_cache"
    candidates = list(cache_root.glob("*/dynamic_avatar_loop.mp4"))
    candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def test_video():
    preferred = PROJECT_ROOT / "test" / "short_video.mp4"
    if preferred.exists():
        return preferred
    matches = sorted((PROJECT_ROOT / "test").glob("*.mp4"))
    return matches[0] if matches else None


def fallback_avatar():
    for pattern in ("*.jpg", "*.png", "*.jpeg"):
        matches = sorted((PROJECT_ROOT / "test").glob(pattern))
        if matches:
            return matches[0]
    return None


def main():
    dynamic_avatar = newest_dynamic_avatar()
    if not dynamic_avatar:
        print("No dynamic avatar cache found under outputs/seedance_cache.")
        return 1
    video = test_video()
    if not video:
        print("No test mp4 found under test/.")
        return 1
    avatar = fallback_avatar()

    output = PROJECT_ROOT / "outputs" / "debug_dynamic_avatar_overlay.mp4"
    result = compose_final_video(
        main_video=str(video),
        avatar_image_path=str(avatar) if avatar else None,
        voice_audio=None,
        subtitle_srt=None,
        title="dynamic avatar overlay test",
        output_path=str(output),
        talking_head_video_path=str(dynamic_avatar),
        digital_human_window_style="card",
        return_debug=True,
    )
    print(f"dynamic_avatar={dynamic_avatar}")
    print(f"source_video={video}")
    print(f"output={output}")
    print(f"used_talking_head={result.get('used_talking_head')}")
    print(f"window_style={result.get('final_composer_used_window_style')}")
    print(f"filter_complex={result.get('filter_complex')}")
    return 0 if result.get("used_talking_head") and output.exists() else 2


if __name__ == "__main__":
    raise SystemExit(main())
