from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Configuration options for the Project Ariane video engine."""

    segment_duration: int = 10
    """Default duration in seconds for each timeline segment."""

    target_lufs: float = -14.0
    """Normalization level applied to audio tracks."""

    upscaler: str = "auto"
    """Upscaling method used when a segment repeats an image."""


DEFAULT_CONFIG = EngineConfig()
