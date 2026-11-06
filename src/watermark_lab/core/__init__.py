"""Core contracts and plugin registry for the watermark lab."""

from __future__ import annotations

from watermark_lab.core.contracts import (
    DetectionConfig,
    DetectionResult,
    DetectorPlugin,
    WatermarkerPlugin,
)
from watermark_lab.core.registry import PluginRegistry

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "DetectorPlugin",
    "PluginRegistry",
    "WatermarkerPlugin",
]
