from services.digital_human.seedance_i2v_provider import SeedanceImageToVideoProvider
from services.digital_human.seedance_t2v_provider import (
    FAST_T2V_MODEL,
    DEFAULT_T2V_MODEL,
    SeedanceTextToVideoVirtualPresenterProvider,
)
from services.digital_human.static_avatar_provider import StaticAvatarProvider
from services.seedance_client import SEEDANCE_MODELS


DIGITAL_HUMAN_MODE_STATIC = "static"
DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST = "seedance_i2v_avatar_fast"
DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0 = "seedance_t2v_virtual_2_0"
DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST = "seedance_t2v_virtual_2_0_fast"
DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL = "seedance_i2v_avatar_2_0_experimental"
DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC_LEGACY = "seedance_i2v"
DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC = DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST
DIGITAL_HUMAN_MODE_LIPSYNC_DISABLED = "lipsync_disabled"


def build_digital_human_provider(mode, *, force_regenerate=False, quality_mode="fast"):
    if mode in {
        DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
        DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC_LEGACY,
    }:
        effective_quality_mode = "fast" if mode == DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST else quality_mode
        return SeedanceImageToVideoProvider(
            force_regenerate=force_regenerate,
            quality_mode=effective_quality_mode,
        )
    if mode == DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL:
        return SeedanceImageToVideoProvider(
            force_regenerate=force_regenerate,
            quality_mode="experimental_2_0_i2v",
            model_candidates=[SEEDANCE_MODELS["i2v_2_0"]],
            reason="Seedance 2.0 image_to_video real-person avatar experimental mode",
        )
    if mode == DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0:
        return SeedanceTextToVideoVirtualPresenterProvider(
            model=DEFAULT_T2V_MODEL,
            force_regenerate=force_regenerate,
        )
    if mode == DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST:
        return SeedanceTextToVideoVirtualPresenterProvider(
            model=FAST_T2V_MODEL,
            force_regenerate=force_regenerate,
        )
    return StaticAvatarProvider()


def create_digital_human(mode, *, avatar_image_path, style, voice_script, clip_id,
                         output_dir, force_regenerate=False, quality_mode="fast",
                         target_duration=None, avatar_video_mode="preview_loop"):
    if mode == DIGITAL_HUMAN_MODE_LIPSYNC_DISABLED:
        return StaticAvatarProvider().create(
            avatar_image_path=avatar_image_path,
            style=style,
            voice_script=voice_script,
            clip_id=clip_id,
            output_dir=output_dir,
        )
    provider = build_digital_human_provider(
        mode,
        force_regenerate=force_regenerate,
        quality_mode=quality_mode,
    )
    if mode in {
        DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0,
        DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST,
    }:
        return provider.create(
            style=style,
            voice_style=style,
            voice_script=voice_script,
            clip_id=clip_id,
            output_dir=output_dir,
            target_duration=target_duration,
        )
    if mode == DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_FAST:
        effective_quality_mode = "fast"
    elif mode == DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL:
        effective_quality_mode = "experimental_2_0_i2v"
    else:
        effective_quality_mode = quality_mode
    return provider.create(
        avatar_image_path=avatar_image_path,
        style=style,
        voice_script=voice_script,
        clip_id=clip_id,
        output_dir=output_dir,
        target_duration=target_duration,
        avatar_video_mode=avatar_video_mode,
        quality_mode=effective_quality_mode,
    )
