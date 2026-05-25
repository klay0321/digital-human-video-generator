import os
import shutil
import subprocess
from pathlib import Path

_FFMPEG_DIR = None


def _find_ffmpeg_dir():
    if shutil.which("ffmpeg") is not None:
        return None
    local = os.environ.get("LOCALAPPDATA", "")
    packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if packages.exists():
        for d in packages.glob("Gyan.FFmpeg*"):
            for exe in d.rglob("ffmpeg.exe"):
                return str(exe.parent)
    return None


def _ensure_ffmpeg():
    global _FFMPEG_DIR
    if _FFMPEG_DIR is None:
        _FFMPEG_DIR = _find_ffmpeg_dir()
    if shutil.which("ffmpeg") is None and shutil.which("ffprobe") is None and _FFMPEG_DIR is None:
        raise RuntimeError(
            "未找到 FFmpeg。请安装：https://ffmpeg.org/download.html，"
            "或 winget install Gyan.FFmpeg，安装后重启终端。"
        )


_FFMPEG_TOOLS = {"ffmpeg", "ffprobe", "ffplay"}


def _cmd(cmd):
    _ensure_ffmpeg()
    if _FFMPEG_DIR and cmd[0] in _FFMPEG_TOOLS:
        return [str(Path(_FFMPEG_DIR) / f"{cmd[0]}.exe")] + cmd[1:]
    return cmd


def run_cmd(cmd):
    cmd = _cmd(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 命令失败 (exit {result.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}"
        )


def extract_audio(video_path, audio_path):
    audio_path = str(Path(audio_path))
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    run_cmd(cmd)
    return audio_path


def cut_video(input_path, start, end, output_path):
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = float(end) - float(start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    run_cmd(cmd)
    return output_path


def concat_videos(video_paths, output_path):
    """Concatenate multiple mp4 files using FFmpeg concat demuxer.

    Strategy: try `-c copy` first (fast & lossless when codecs match), fall
    back to re-encoding on failure. Single-file input is just copied.
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not video_paths:
        raise ValueError("concat_videos: video_paths is empty")

    valid = [str(Path(p)) for p in video_paths if Path(p).exists()]
    if not valid:
        raise FileNotFoundError(f"concat_videos: none of the inputs exist: {video_paths}")

    if len(valid) == 1:
        shutil.copyfile(valid[0], output_path)
        return output_path

    list_file = Path(output_path).parent / f"_concat_{Path(output_path).stem}.txt"
    lines = []
    for p in valid:
        abs_p = str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{abs_p}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")

    cmd_copy = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", output_path,
    ]
    try:
        run_cmd(cmd_copy)
    except RuntimeError:
        cmd_reenc = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-c:a", "aac",
            "-movflags", "+faststart", output_path,
        ]
        run_cmd(cmd_reenc)
    finally:
        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass

    return output_path
