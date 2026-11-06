from watermark_lab.core.registry import PluginRegistry


def test_empty_registry_has_explicit_capabilities() -> None:
    registry = PluginRegistry()
    assert registry.capabilities() == {"detectors": [], "watermarkers": []}

