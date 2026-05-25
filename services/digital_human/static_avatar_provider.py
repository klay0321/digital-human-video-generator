from services.digital_human.base import DigitalHumanProvider, DigitalHumanProviderResult


class StaticAvatarProvider(DigitalHumanProvider):
    provider_name = "static"

    def create(self, *, avatar_image_path, style, voice_script, clip_id, output_dir, **kwargs):
        return DigitalHumanProviderResult(
            talking_head_video_path=None,
            success=True,
            error=None,
            debug={"avatar_image_path": str(avatar_image_path), "mode": "static_fallback"},
            provider=self.provider_name,
            is_lip_sync=False,
            uses_audio_driven_lipsync=False,
        )
