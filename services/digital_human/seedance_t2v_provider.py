import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from services.digital_human.base import DigitalHumanProvider, DigitalHumanProviderResult
from services.ffmpeg_utils import _cmd
from services.seedance_client import (
    SEEDANCE_MODELS,
    SeedanceClient,
    SeedanceClientError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

DEFAULT_T2V_MODEL = SEEDANCE_MODELS["t2v_2_0"]
FAST_T2V_MODEL = SEEDANCE_MODELS["t2v_2_0_fast"]
DEFAULT_DURATION_SECONDS = 5.0
MIN_T2V_DURATION_SECONDS = 4.0

NEGATIVE_PROMPT = (
    "不要多人，不要大幅移动，不要离开画面，不要转身，不要全身走动，"
    "不要卡通化，不要夸张演戏感，不要画面跳变，不要复杂背景。"
)

STYLE_PROMPT_SUFFIX = {
    "讲解风格": "像中文技术教程讲解，表达清晰，节奏自然。",
    "开会风格": "像商务汇报，表情沉稳，动作克制。",
    "带货风格": "轻微热情，有感染力，但不要夸张。",
    "沉稳专业风格": "正式、专业、动作克制，适合企业讲解。",
    "轻松口播风格": "亲切自然，像短视频口播分享。",
}


class SeedanceTextToVideoVirtualPresenterProvider(DigitalHumanProvider):
    provider_name = "seedance_text_to_video_virtual_presenter"

    def __init__(
        self,
        *,
        model=DEFAULT_T2V_MODEL,
        cache_root=None,
        client=None,
        force_regenerate=False,
    ):
        self.model = (model or DEFAULT_T2V_MODEL).strip()
        self.cache_root = Path(cache_root or Path("outputs") / "seedance_t2v_cache")
        self.client = client
        self.force_regenerate = force_regenerate

    def create(
        self,
        *,
        avatar_image_path=None,
        style=None,
        voice_style=None,
        voice_script,
        clip_id,
        output_dir,
        target_duration=None,
        model=None,
        **kwargs,
    ):
        selected_model = (model or self.model or DEFAULT_T2V_MODEL).strip()
        effective_style = voice_style or style or "讲解风格"
        duration = self._normalize_duration(target_duration)
        prompt = self.build_prompt(effective_style, voice_script, duration)
        cache_key = self.build_cache_key(selected_model, prompt, duration)
        cache_dir = self.cache_root / cache_key
        output_path = cache_dir / "virtual_presenter.mp4"
        metadata_path = cache_dir / "metadata.json"

        debug = {
            "clip_id": clip_id,
            "style": effective_style,
            "seedance_model_requested": selected_model,
            "seedance_requested_model": selected_model,
            "requested_model_candidates": [selected_model],
            "seedance_model_candidates": [selected_model],
            "seedance_model_used": selected_model,
            "seedance_used_model": selected_model,
            "seedance_generation_type": "text_to_video",
            "identity_preserved_from_avatar": False,
            "is_lip_sync": False,
            "uses_audio_driven_lipsync": False,
            "target_duration": duration,
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "prompt": prompt,
            "seedance_prompt": prompt,
            "seedance_negative_prompt": NEGATIVE_PROMPT,
            "force_regenerate": self.force_regenerate,
            "generation_mode": "text_to_video",
            "avatar_image_ignored": bool(avatar_image_path),
            "reason": "virtual text_to_video presenter does not preserve uploaded avatar identity",
        }

        try:
            if not self.force_regenerate and output_path.exists():
                validation = self.validate_video(output_path)
                debug["cached_validation"] = validation
                if validation["valid"]:
                    debug.update({
                        "cache_hit": True,
                        "seedance_cache_hit": True,
                        "talking_head_video_path": str(output_path),
                        "talking_head_video_exists": True,
                        "talking_head_video_size": validation["size"],
                        "talking_head_video_duration": validation["duration"],
                        "generated_duration": validation["duration"],
                    })
                    return self._success(str(output_path), debug)
                debug["cache_invalid_reason"] = validation["error"]

            debug["cache_hit"] = False
            debug["seedance_cache_hit"] = False
            cache_dir.mkdir(parents=True, exist_ok=True)
            result_path, generation_debug = self._generate_one(
                prompt=prompt,
                model=selected_model,
                output_path=output_path,
                target_duration=duration,
            )
            debug.update(generation_debug)

            validation = self.validate_video(result_path)
            debug["output_validation"] = validation
            if not validation["valid"]:
                return DigitalHumanProviderResult(
                    talking_head_video_path=None,
                    success=False,
                    error="Seedance text_to_video output video invalid or missing",
                    debug=debug,
                    provider=self.provider_name,
                    is_lip_sync=False,
                    uses_audio_driven_lipsync=False,
                )

            debug.update({
                "talking_head_video_path": str(result_path),
                "talking_head_video_exists": True,
                "talking_head_video_size": validation["size"],
                "talking_head_video_duration": validation["duration"],
                "generated_duration": validation["duration"],
            })
            metadata_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
            return self._success(str(result_path), debug)
        except Exception as error:
            debug["error_type"] = error.__class__.__name__
            if isinstance(error, SeedanceClientError):
                debug["seedance_error"] = error.to_result_error()
            return DigitalHumanProviderResult(
                talking_head_video_path=None,
                success=False,
                error=str(error),
                debug=debug,
                provider=self.provider_name,
                is_lip_sync=False,
                uses_audio_driven_lipsync=False,
            )

    def build_prompt(self, voice_style, voice_script, target_duration):
        summary = self._summarize_voice_script(voice_script)
        style_suffix = STYLE_PROMPT_SUFFIX.get(voice_style) or STYLE_PROMPT_SUFFIX["讲解风格"]
        return (
            "生成一个真实中文讲解人像视频。虚拟讲解人正面对镜头，半身或胸像构图，"
            "自然眨眼，轻微口播动作，轻微点头，头部和肩部只有很轻微的自然移动。"
            "背景简洁干净，画面稳定清晰，适合放在录屏视频右下角小窗。"
            f"{style_suffix}"
            f"讲解内容摘要：{summary}。"
            f"视频目标时长约 {target_duration:.1f} 秒。"
            f"{NEGATIVE_PROMPT}"
        )

    def _generate_one(self, *, prompt, model, output_path, target_duration):
        client = self.client or self._build_client_from_env(model=model)
        if self.client is not None:
            client.model = model
        create_response = client.create_task(
            "text_to_video",
            prompt=prompt,
            uploaded_assets={},
            duration=max(1, int(round(target_duration))),
        )
        task_id = client.extract_task_id(create_response)
        if not task_id:
            raise RuntimeError("Seedance text_to_video did not return task id")
        final_response = client.poll_task(task_id)
        local_path = client.download_result(final_response, output_path)
        if not local_path:
            raise RuntimeError("Seedance text_to_video finished without output video")
        return Path(local_path), {
            "task_id": task_id,
            "create_response": create_response,
            "final_response": final_response,
            "requested_part_duration": target_duration,
        }

    def _build_client_from_env(self, *, model):
        return SeedanceClient(
            api_key=os.getenv("SEEDANCE_API_KEY", ""),
            api_base=os.getenv("SEEDANCE_API_BASE", "https://ark.cn-beijing.volces.com"),
            model=model,
            endpoint_generate=os.getenv("SEEDANCE_ENDPOINT_GENERATE", "contents/generations/tasks"),
            endpoint_status=os.getenv("SEEDANCE_ENDPOINT_STATUS", "contents/generations/tasks/{task_id}"),
            endpoint_upload=os.getenv("SEEDANCE_ENDPOINT_UPLOAD", ""),
            max_poll_seconds=int(os.getenv("SEEDANCE_MAX_POLL_SECONDS", "600") or "600"),
        )

    def _success(self, video_path, debug):
        return DigitalHumanProviderResult(
            talking_head_video_path=video_path,
            success=True,
            error=None,
            debug=debug,
            provider=self.provider_name,
            is_lip_sync=False,
            uses_audio_driven_lipsync=False,
        )

    def build_cache_key(self, model, prompt, target_duration):
        raw = "|".join([model or "", prompt or "", f"{target_duration:.1f}"])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def validate_video(self, video_path):
        path = Path(video_path) if video_path else None
        if not path:
            return {"valid": False, "exists": False, "size": 0, "duration": None,
                    "error": "talking_head_video_path is empty"}
        if not path.exists():
            return {"valid": False, "exists": False, "size": 0, "duration": None,
                    "error": "talking_head_video_path does not exist"}
        size = path.stat().st_size
        if size <= 0:
            return {"valid": False, "exists": True, "size": size, "duration": None,
                    "error": "talking_head_video_path is empty"}
        duration = self._probe_duration(path)
        if duration is None or duration <= 0.5:
            return {"valid": False, "exists": True, "size": size, "duration": duration,
                    "error": "ffprobe duration is missing or <= 0.5s"}
        return {"valid": True, "exists": True, "size": size, "duration": duration, "error": None}

    def _probe_duration(self, video_path):
        try:
            import subprocess
            result = subprocess.run(
                _cmd([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ]),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode != 0:
                return None
            return float((result.stdout or "0").strip() or 0)
        except Exception:
            return None

    def _normalize_duration(self, target_duration):
        try:
            value = float(target_duration or DEFAULT_DURATION_SECONDS)
        except Exception:
            value = DEFAULT_DURATION_SECONDS
        return max(MIN_T2V_DURATION_SECONDS, min(value, 120.0))

    def _summarize_voice_script(self, voice_script):
        text = " ".join(str(voice_script or "").split())
        if not text:
            return "中文讲解人正在进行简洁专业的课程说明"
        return text[:180]
