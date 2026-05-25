import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from services.digital_human.base import DigitalHumanProvider, DigitalHumanProviderResult
from services.ffmpeg_utils import _cmd
from services.seedance_client import (
    DEFAULT_SEEDANCE_MODEL,
    SEEDANCE_PRIVACY_USER_MESSAGE,
    SEEDANCE_MODELS,
    SeedanceClient,
    SeedanceClientError,
    is_seedance_privacy_error_code,
    seedance_error_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

REMOVED_SEEDANCE_MODEL_IDS = {f"seedance-{version}" for version in ("2.0", "2.0-fast")}
AVATAR_FAST_MODEL = SEEDANCE_MODELS["avatar_fast"]
AVATAR_FAST_REASON = "2.0 i2v with real person image triggers privacy protection"


def _configured_model(env_name, default_model):
    value = (os.getenv(env_name) or "").strip()
    if not value or value in REMOVED_SEEDANCE_MODEL_IDS:
        return default_model
    return value


def _configured_model_candidates(env_name, default_candidates):
    raw = (os.getenv(env_name) or "").strip()
    candidates = [item.strip() for item in raw.split(",") if item.strip()] if raw else list(default_candidates)
    candidates = [item for item in candidates if item not in REMOVED_SEEDANCE_MODEL_IDS]
    return candidates or list(default_candidates)


FAST_SEEDANCE_MODEL = _configured_model(
    "SEEDANCE_FAST_MODEL",
    _configured_model("SEEDANCE_MODEL", DEFAULT_SEEDANCE_MODEL),
)
HIGH_QUALITY_SEEDANCE_MODEL = _configured_model("SEEDANCE_HIGH_QUALITY_MODEL", SEEDANCE_MODELS["i2v_2_0"])

HIGH_QUALITY_MODEL_CANDIDATES = _configured_model_candidates(
    "SEEDANCE_HIGH_QUALITY_MODEL_CANDIDATES",
    [SEEDANCE_MODELS["i2v_2_0"], SEEDANCE_MODELS["i2v_2_0_fast"], SEEDANCE_MODELS["avatar_fast"]],
)
FAST_MODEL_CANDIDATES = _configured_model_candidates(
    "SEEDANCE_FAST_MODEL_CANDIDATES",
    [AVATAR_FAST_MODEL],
)

PREVIEW_DURATION_SECONDS = 5.0
CHUNK_SECONDS = 10.0

BASE_DYNAMIC_AVATAR_PROMPT = (
    "基于参考头像生成一个真实中文讲解人像视频。人物正面对镜头，保持参考头像的人物身份、脸型、五官特征和整体气质。"
    "表情自然，有轻微表情变化、自然眨眼、轻微点头、轻微口播式嘴部开合动作；头部和肩部有非常轻微的自然移动，"
    "像正在讲解内容。构图稳定，半身或胸像，适合放在录屏视频右下角小窗，背景简洁，画面清晰稳定。"
)

NEGATIVE_PROMPT = (
    "不要大幅移动，不要强烈手势，不要离开画面，不要转身，不要全身走动，不要多人，不要换脸，"
    "不要画面跳变，不要卡通化，不要静止不动，不要夸张演戏感。"
)

PROMPT_SUFFIX_BY_STYLE = {
    "讲解风格": "像技术教程讲解，表达清晰，节奏自然。",
    "开会风格": "像商务汇报，表情沉稳，动作克制。",
    "带货风格": "更有感染力，轻微热情，但不要夸张。",
    "沉稳专业风格": "更正式，动作克制，表情专业，适合企业汇报。",
    "轻松口播风格": "更亲切自然，像短视频口播分享。",
}


class SeedanceImageToVideoProvider(DigitalHumanProvider):
    provider_name = "seedance_image_to_video"

    def __init__(
        self,
        cache_root=None,
        client=None,
        force_regenerate=False,
        quality_mode="fast",
        model_candidates=None,
        reason=AVATAR_FAST_REASON,
    ):
        self.cache_root = Path(cache_root or Path("outputs") / "seedance_cache")
        self.client = client
        self.force_regenerate = force_regenerate
        self.quality_mode = quality_mode or "fast"
        self.model_candidates = list(model_candidates) if model_candidates else None
        self.reason = reason or AVATAR_FAST_REASON

    def create(
        self,
        *,
        avatar_image_path,
        style,
        voice_script,
        clip_id,
        output_dir,
        target_duration=None,
        avatar_video_mode="preview_loop",
        quality_mode=None,
    ):
        avatar = Path(avatar_image_path)
        requested_quality_mode = quality_mode or self.quality_mode or "fast"
        requested_model_candidates = self._model_candidates_for_quality(requested_quality_mode)
        requested_model = requested_model_candidates[0]
        effective_mode = avatar_video_mode or "preview_loop"
        target_duration = self._normalize_target_duration(target_duration, effective_mode)
        duration_bucket = self._duration_bucket(target_duration, effective_mode)
        prompt = self.build_prompt(style, target_duration=target_duration, mode=effective_mode)
        cache_key = self.build_cache_key(
            avatar,
            style,
            requested_model,
            prompt,
            duration_bucket,
            requested_quality_mode,
        )
        cache_dir = self.cache_root / cache_key
        output_name = "dynamic_avatar_loop.mp4" if effective_mode == "preview_loop" else "digital_human_full.mp4"
        cached_video = cache_dir / output_name
        metadata_path = cache_dir / "metadata.json"

        debug = {
            "clip_id": clip_id,
            "style": style,
            "quality_mode": requested_quality_mode,
            "avatar_video_mode": effective_mode,
            "seedance_model_requested": requested_model,
            "seedance_requested_model": requested_model,
            "requested_model_candidates": requested_model_candidates,
            "seedance_model_candidates": requested_model_candidates,
            "seedance_model_used": None,
            "seedance_used_model": None,
            "seedance_model_fallback_reason": None,
            "seedance_generation_type": "image_to_video",
            "identity_preserved_from_avatar": True,
            "reason": self.reason,
            "target_duration": target_duration,
            "duration_bucket": duration_bucket,
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "prompt": prompt,
            "seedance_prompt": prompt,
            "seedance_negative_prompt": NEGATIVE_PROMPT,
            "force_regenerate": self.force_regenerate,
            "is_lip_sync": False,
            "uses_audio_driven_lipsync": False,
            "generation_mode": effective_mode,
            "chunk_count": 0,
        }

        try:
            if not self.force_regenerate and cached_video.exists():
                validation = self.validate_video(cached_video)
                debug["cached_validation"] = validation
                if validation["valid"]:
                    debug.update({
                        "cache_hit": True,
                        "seedance_cache_hit": True,
                        "talking_head_video_path": str(cached_video),
                        "talking_head_full_video_path": str(cached_video) if effective_mode != "preview_loop" else None,
                        "talking_head_video_exists": True,
                        "talking_head_video_size": validation["size"],
                        "talking_head_video_duration": validation["duration"],
                        "generated_duration": validation["duration"],
                        "seedance_model_used": self._cached_model_used(metadata_path, requested_model),
                        "seedance_used_model": self._cached_model_used(metadata_path, requested_model),
                        "generation_mode": self._cached_generation_mode(metadata_path, effective_mode),
                        "chunk_count": self._cached_chunk_count(metadata_path),
                    })
                    return self._success(str(cached_video), debug)
                debug["cache_invalid_reason"] = validation["error"]

            debug["cache_hit"] = False
            debug["seedance_cache_hit"] = False
            debug["cache_bypassed"] = bool(self.force_regenerate)
            cache_dir.mkdir(parents=True, exist_ok=True)

            result_path, model_used, fallback_reason, generation_debug = self._generate_with_model_fallback(
                avatar=avatar,
                prompt=prompt,
                cache_dir=cache_dir,
                output_path=cached_video,
                target_duration=target_duration,
                requested_model=requested_model,
                model_candidates=requested_model_candidates,
                requested_quality_mode=requested_quality_mode,
                effective_mode=effective_mode,
            )
            debug.update(generation_debug)
            debug["seedance_model_used"] = model_used
            debug["seedance_used_model"] = model_used
            debug["seedance_model_fallback_reason"] = fallback_reason
            debug["fallback_reason"] = fallback_reason
            privacy_error = self._first_privacy_error(debug)
            if privacy_error:
                debug.update(self._privacy_debug_fields(privacy_error))

            validation = self.validate_video(result_path)
            debug["output_validation"] = validation
            if not validation["valid"]:
                return DigitalHumanProviderResult(
                    talking_head_video_path=None,
                    success=False,
                    error="Seedance output video invalid or missing",
                    debug=debug,
                    provider=self.provider_name,
                    is_lip_sync=False,
                    uses_audio_driven_lipsync=False,
                )

            debug.update({
                "talking_head_video_path": str(result_path),
                "talking_head_full_video_path": str(result_path) if effective_mode != "preview_loop" else None,
                "talking_head_video_exists": True,
                "talking_head_video_size": validation["size"],
                "talking_head_video_duration": validation["duration"],
                "generated_duration": validation["duration"],
            })
            metadata_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
            return self._success(str(result_path), debug)
        except Exception as error:
            debug["error_type"] = error.__class__.__name__
            debug["seedance_attempt_errors"] = getattr(self, "_last_attempt_errors", [])
            if isinstance(error, SeedanceClientError):
                debug["seedance_error"] = error.to_result_error()
            elif debug["seedance_attempt_errors"]:
                debug["seedance_error"] = debug["seedance_attempt_errors"][-1].get("error")
            privacy_error = self._first_privacy_error(debug)
            if privacy_error:
                debug.update(self._privacy_debug_fields(privacy_error))
                result_error = SEEDANCE_PRIVACY_USER_MESSAGE
            else:
                result_error = str(error)
            return DigitalHumanProviderResult(
                talking_head_video_path=None,
                success=False,
                error=result_error,
                debug=debug,
                provider=self.provider_name,
                is_lip_sync=False,
                uses_audio_driven_lipsync=False,
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

    def _generate_with_model_fallback(
        self,
        *,
        avatar,
        prompt,
        cache_dir,
        output_path,
        target_duration,
        requested_model,
        model_candidates,
        requested_quality_mode,
        effective_mode,
    ):
        last_error = None
        attempt_errors = []
        self._last_attempt_errors = attempt_errors
        for index, model in enumerate(model_candidates):
            try:
                path, generation_debug = self._generate_for_mode(
                    avatar=avatar,
                    prompt=prompt,
                    cache_dir=cache_dir,
                    output_path=output_path,
                    target_duration=target_duration,
                    model=model,
                    effective_mode=effective_mode,
                )
                fallback_reason = None
                if index > 0:
                    fallback_reason = self._fallback_reason(requested_model, model, attempt_errors)
                generation_debug["seedance_attempt_errors"] = attempt_errors
                return path, model, fallback_reason, generation_debug
            except Exception as error:
                last_error = str(error)
                attempt_errors.append(self._error_to_debug(error, model))
                self._last_attempt_errors = attempt_errors
                if index >= len(model_candidates) - 1:
                    raise
        raise RuntimeError(last_error or "Seedance generation failed")

    def _generate_for_mode(self, *, avatar, prompt, cache_dir, output_path, target_duration, model, effective_mode):
        if effective_mode == "preview_loop":
            part_debug = self._generate_one(
                avatar=avatar,
                prompt=prompt,
                model=model,
                output_path=output_path,
                target_duration=PREVIEW_DURATION_SECONDS,
                part_index=1,
            )
            part_debug.update({"generation_mode": "preview_loop", "chunk_count": 1})
            return output_path, part_debug

        direct_debug = self._generate_one(
            avatar=avatar,
            prompt=prompt,
            model=model,
            output_path=output_path,
            target_duration=target_duration,
            part_index=1,
        )
        direct_validation = self.validate_video(output_path)
        direct_duration = direct_validation.get("duration") or 0
        if direct_validation["valid"] and direct_duration >= max(target_duration - 1.0, target_duration * 0.85):
            direct_debug.update({"generation_mode": "full_length", "chunk_count": 1})
            return output_path, direct_debug

        remaining_after_direct = max(target_duration - direct_duration, 0)
        chunk_count = max(2, 1 + int(math.ceil(remaining_after_direct / CHUNK_SECONDS)))
        chunk_paths = []
        chunk_debug = {
            "generation_mode": "chunked_concat",
            "chunk_count": chunk_count,
            "direct_attempt_duration": direct_duration,
            "chunks": [],
        }
        first_part = cache_dir / "clip_001_dh_part01.mp4"
        output_path.replace(first_part)
        chunk_paths.append(first_part)
        chunk_debug["chunks"].append({"part": 1, "path": str(first_part), "duration": direct_duration})

        for part_index in range(2, chunk_count + 1):
            part_path = cache_dir / f"clip_001_dh_part{part_index:02d}.mp4"
            remaining = max(target_duration - sum((self.validate_video(p).get("duration") or 0) for p in chunk_paths), 1.0)
            part_duration = min(CHUNK_SECONDS, remaining)
            part_prompt = f"{prompt} 这是同一位讲解人像的第 {part_index} 段，动作保持自然连续，避免突然换脸或大幅变化。"
            part_result = self._generate_one(
                avatar=avatar,
                prompt=part_prompt,
                model=model,
                output_path=part_path,
                target_duration=part_duration,
                part_index=part_index,
            )
            validation = self.validate_video(part_path)
            chunk_paths.append(part_path)
            chunk_debug["chunks"].append({
                "part": part_index,
                "path": str(part_path),
                "duration": validation.get("duration"),
                "task_id": part_result.get("task_id"),
            })

        self._concat_videos(chunk_paths, output_path)
        return output_path, chunk_debug

    def _generate_one(self, *, avatar, prompt, model, output_path, target_duration, part_index):
        client = self.client or self._build_client_from_env(model=model)
        if self.client is not None:
            client.model = model

        duration_prompt = (
            f"{prompt} 视频目标时长约 {target_duration:.1f} 秒，动作自然连贯，避免循环感。{NEGATIVE_PROMPT}"
        )
        uploaded = {"avatar": client.upload_file(avatar, purpose="avatar")}
        create_response = client.create_task(
            "image_to_video",
            prompt=duration_prompt,
            uploaded_assets=uploaded,
            duration=max(1, int(round(target_duration))),
        )
        task_id = client.extract_task_id(create_response)
        if not task_id:
            raise RuntimeError("Seedance image_to_video did not return task id")
        final_response = client.poll_task(task_id)
        local_path = client.download_result(final_response, output_path)
        if not local_path:
            raise RuntimeError("Seedance image_to_video finished without output video")
        return {
            "part_index": part_index,
            "task_id": task_id,
            "create_response": create_response,
            "final_response": final_response,
            "requested_part_duration": target_duration,
        }

    def _concat_videos(self, video_paths, output_path):
        list_path = Path(output_path).with_suffix(".concat.txt")
        list_text = "".join(f"file '{Path(path).resolve().as_posix()}'\n" for path in video_paths)
        list_path.write_text(list_text, encoding="utf-8")
        result = subprocess.run(
            _cmd([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c:v", "libx264",
                "-an",
                "-movflags", "+faststart",
                str(output_path),
            ]),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Seedance chunk concat failed: {result.stderr}")

    def build_prompt(self, style, *, target_duration=None, mode="preview_loop"):
        suffix = PROMPT_SUFFIX_BY_STYLE.get(style) or PROMPT_SUFFIX_BY_STYLE["讲解风格"]
        duration_text = ""
        if target_duration:
            duration_text = f" 生成接近 {target_duration:.1f} 秒的讲解人像视频。"
        if mode == "preview_loop":
            duration_text += " 这是快速预览短视频，动作要有轻微变化，方便作为短循环预览。"
        else:
            duration_text += " 这是正式导出视频，动作要尽量自然延续，减少短循环重复感。"
        return f"{BASE_DYNAMIC_AVATAR_PROMPT}{duration_text}{suffix}{NEGATIVE_PROMPT}"

    def build_cache_key(self, avatar_path, voice_style, seedance_model, prompt, duration_bucket, quality_mode):
        avatar_hash = hashlib.sha256(Path(avatar_path).read_bytes()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        raw = "|".join([
            avatar_hash,
            voice_style or "",
            seedance_model or "",
            prompt_hash,
            str(duration_bucket),
            quality_mode or "",
        ])
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

    def _model_candidates_for_quality(self, quality_mode):
        if self.model_candidates:
            return list(self.model_candidates)
        if quality_mode == "high_quality":
            candidates = list(HIGH_QUALITY_MODEL_CANDIDATES)
            if HIGH_QUALITY_SEEDANCE_MODEL not in candidates:
                candidates.append(HIGH_QUALITY_SEEDANCE_MODEL)
            for fast_model in FAST_MODEL_CANDIDATES:
                if fast_model not in candidates:
                    candidates.append(fast_model)
            return candidates
        return [AVATAR_FAST_MODEL]

    def _error_to_debug(self, error, model):
        if isinstance(error, SeedanceClientError):
            result_error = error.to_result_error()
            return {
                "model": model,
                "error": result_error,
                "privacy_guard_triggered": bool(result_error.get("privacy_guard_triggered")),
                "seedance_error_code": result_error.get("error_code"),
                "seedance_error_message": result_error.get("error_message"),
                "fallback_reason": result_error.get("display_message") or result_error.get("error_message"),
            }
        return {
            "model": model,
            "error": {
                "success": False,
                "stage": None,
                "model": model,
                "error_message": str(error),
                "message": str(error),
            },
        }

    def _fallback_reason(self, requested_model, fallback_model, attempt_errors):
        first_error = (attempt_errors or [{}])[0].get("error") or {}
        code = first_error.get("error_code")
        message = first_error.get("error_message") or first_error.get("message")
        if is_seedance_privacy_error_code(code):
            return f"{requested_model} 触发真人图片隐私风控；返回至 {fallback_model}：{SEEDANCE_PRIVACY_USER_MESSAGE}"
        status = first_error.get("status_code")
        hints = []
        if code and "notfound" in str(code).lower():
            hints.append("可能是 model id 写错、base_url 不匹配或当前 key 未开通该模型。")
        if status in {401, 403}:
            hints.append("可能是 key 没有该模型权限。")
        if message and any(token in str(message).lower() for token in ("image", "url", "base64")):
            hints.append("可能是 image_to_video 要求公网 image_url，而不是本地 data URL。")
        hint_text = " ".join(hints)
        detail = f"status={status}, code={code}, message={message}"
        if hint_text:
            detail = f"{detail}. {hint_text}"
        return f"{requested_model} 不可用或失败；返回至 {fallback_model}：{detail}"

    def _first_privacy_error(self, debug):
        errors = []
        if isinstance(debug.get("seedance_error"), dict):
            errors.append(debug["seedance_error"])
        for attempt in debug.get("seedance_attempt_errors") or []:
            if isinstance(attempt, dict) and isinstance(attempt.get("error"), dict):
                errors.append(attempt["error"])
        for error in errors:
            if is_seedance_privacy_error_code(error.get("error_code")):
                return error
        return None

    def _privacy_debug_fields(self, error):
        metadata = seedance_error_metadata(error.get("error_code"))
        return {
            "seedance_error_type": metadata.get("seedance_error_type"),
            "seedance_error_code": error.get("error_code"),
            "seedance_error_message": error.get("error_message") or error.get("message"),
            "suggested_action": metadata.get("suggested_action"),
            "privacy_guard_triggered": bool(metadata.get("privacy_guard_triggered")),
            "privacy_guard_message": metadata.get("user_message"),
        }

    def _normalize_target_duration(self, target_duration, mode):
        if mode == "preview_loop":
            return PREVIEW_DURATION_SECONDS
        try:
            value = float(target_duration or PREVIEW_DURATION_SECONDS)
        except Exception:
            value = PREVIEW_DURATION_SECONDS
        return max(PREVIEW_DURATION_SECONDS, min(value, 120.0))

    def _duration_bucket(self, target_duration, mode):
        if mode == "preview_loop":
            return "preview_5s"
        bucket = int(math.ceil(float(target_duration or PREVIEW_DURATION_SECONDS) / 5.0) * 5)
        return f"full_{bucket}s"

    def _cached_model_used(self, metadata_path, default_model):
        data = self._read_metadata(metadata_path)
        return data.get("seedance_model_used") or default_model

    def _cached_generation_mode(self, metadata_path, default_mode):
        data = self._read_metadata(metadata_path)
        return data.get("generation_mode") or default_mode

    def _cached_chunk_count(self, metadata_path):
        data = self._read_metadata(metadata_path)
        return data.get("chunk_count") or 1

    def _read_metadata(self, metadata_path):
        try:
            if Path(metadata_path).exists():
                return json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {}
