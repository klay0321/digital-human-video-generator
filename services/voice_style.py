"""ElevenLabs TTS voice-style presets for different speaking modes.

CRITICAL: `style` should stay moderate. Values >0.5 make the model unstable
on Chinese content. Per-fragment short TTS is preferred over long monologue
TTS — see pipeline._render_clip_with_fragments.
"""

_PRESETS = {
    "讲解风格": {
        "stability": 0.45,
        "similarity_boost": 0.82,
        "style": 0.20,
        "speed": 0.95,
        "use_speaker_boost": True,
    },
    "开会风格": {
        "stability": 0.55,
        "similarity_boost": 0.85,
        "style": 0.10,
        "speed": 0.92,
        "use_speaker_boost": True,
    },
    "带货风格": {
        "stability": 0.35,
        "similarity_boost": 0.75,
        "style": 0.45,
        "speed": 1.06,
        "use_speaker_boost": True,
    },
    "沉稳专业风格": {
        "stability": 0.70,
        "similarity_boost": 0.88,
        "style": 0.05,
        "speed": 0.90,
        "use_speaker_boost": True,
    },
    "轻松口播风格": {
        "stability": 0.42,
        "similarity_boost": 0.80,
        "style": 0.30,
        "speed": 1.00,
        "use_speaker_boost": True,
    },
}

DEFAULT_VOICE_STYLE = "讲解风格"
VOICE_STYLE_NAMES = tuple(_PRESETS.keys())


def get_voice_style_preset(style_name):
    """Return a copy of the voice_settings dict for a named style."""
    if style_name in _PRESETS:
        return dict(_PRESETS[style_name])
    return dict(_PRESETS[DEFAULT_VOICE_STYLE])


def list_voice_styles():
    return list(VOICE_STYLE_NAMES)
