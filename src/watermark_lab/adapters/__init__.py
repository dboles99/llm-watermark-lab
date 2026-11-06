"""Watermark adapters implementing paper-specified methods."""

from __future__ import annotations

from watermark_lab.adapters.kgw import DEFAULT_HASH_KEY, kgw_detect
from watermark_lab.core.registry import PluginRegistry

_REGISTRY = PluginRegistry()


class _KgwDetectorPlugin:
    method_id = "kgw"

    def detect(self, text: str, config):  # type: ignore[no-untyped-def]
        from watermark_lab.run_engine.detect import DetectionService

        return DetectionService().detect(text, config)


_REGISTRY.register_detector(_KgwDetectorPlugin())


def registry() -> PluginRegistry:
    return _REGISTRY


__all__ = [
    "DEFAULT_HASH_KEY",
    "kgw_detect",
    "registry",
]
