from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DigitalHumanProviderResult:
    talking_head_video_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    provider: str = "unknown"
    is_lip_sync: bool = False
    uses_audio_driven_lipsync: bool = False

    def to_report(self) -> Dict[str, Any]:
        return asdict(self)


class DigitalHumanProvider:
    provider_name = "base"

    def create(self, *, avatar_image_path, style, voice_script, clip_id, output_dir, **kwargs):
        raise NotImplementedError
