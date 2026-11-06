from __future__ import annotations

from typing import Any


class PluginRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, Any] = {}
        self._watermarkers: dict[str, Any] = {}

    def register_detector(self, plugin: Any) -> None:
        self._detectors[plugin.method_id] = plugin

    def register_watermarker(self, plugin: Any) -> None:
        self._watermarkers[plugin.method_id] = plugin

    def detector(self, method_id: str) -> Any:
        return self._detectors[method_id]

    def watermarker(self, method_id: str) -> Any:
        return self._watermarkers[method_id]

    def capabilities(self) -> dict[str, list[str]]:
        return {
            "detectors": sorted(self._detectors),
            "watermarkers": sorted(self._watermarkers),
        }

