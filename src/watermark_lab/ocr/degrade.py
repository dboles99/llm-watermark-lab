"""Deterministic degradation ladder for print-scan-OCR round trips.

Each degradation is a named transformation with a severity in ``[0, 1]`` and an
explicit seed, so it slots into the project's existing typed-edit-class model:
severity zero must be the identity, and the same (name, severity, seed) triple
must reproduce the same image.

The ladder models what actually happens to a printed page — resampling, JPEG
compression, skew from a misaligned feed, sensor noise, optical blur, paper
texture and show-through, and finally a photograph of a screen. These are not
adversarial perturbations. They are the ordinary degradations of the
print-and-rescan path, which is precisely why the resulting attack preserves
meaning by construction.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DegradeFn = Callable[["Any", float, "Any"], "Any"]


class DegradationError(ValueError):
    """Raised when a degradation is misconfigured."""


@dataclass(frozen=True, slots=True)
class Degradation:
    """One named degradation coordinate."""

    name: str
    severity: float
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "severity": self.severity, "seed": self.seed}


def _validate_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)):
        raise DegradationError("severity must be a real number.")
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise DegradationError("severity must be between 0 and 1 inclusive.")
    return value


def _identity(image: Any, severity: float, rng: Any) -> Any:
    return image


def _jpeg(image: Any, severity: float, rng: Any) -> Any:
    """Re-encode as JPEG. Quality falls from 95 to 10 as severity rises."""

    import numpy as np
    from PIL import Image

    quality = round(95 - 85 * severity)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=max(1, quality))
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def _downsample(image: Any, severity: float, rng: Any) -> Any:
    """Resample down and back up, modelling a lower scan DPI (300 -> 72)."""

    import numpy as np
    from PIL import Image

    scale = 1.0 - 0.76 * severity
    pil = Image.fromarray(image)
    width = max(1, int(pil.width * scale))
    height = max(1, int(pil.height * scale))
    shrunk = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(shrunk.resize(pil.size, Image.Resampling.LANCZOS))


def _rotate(image: Any, severity: float, rng: Any) -> Any:
    """Rotate by up to 5 degrees, modelling a skewed feed."""

    import numpy as np
    from PIL import Image

    angle = (rng.random() * 2.0 - 1.0) * 5.0 * severity
    rotated = Image.fromarray(image).rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255)
    )
    return np.array(rotated)


def _noise(image: Any, severity: float, rng: Any) -> Any:
    """Add Gaussian sensor noise."""

    import numpy as np

    sigma = 45.0 * severity
    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _blur(image: Any, severity: float, rng: Any) -> Any:
    """Apply Gaussian blur, modelling optical defocus."""

    import numpy as np
    from PIL import Image, ImageFilter

    radius = 3.0 * severity
    if radius <= 0:
        return image
    return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))


def _paper(image: Any, severity: float, rng: Any) -> Any:
    """Add paper texture and show-through from the reverse side."""

    import numpy as np

    height, width = image.shape[:2]
    texture = rng.normal(0.0, 12.0 * severity, (height, width, 1))
    tint = np.array([0.0, -6.0 * severity, -14.0 * severity])
    show_through = rng.normal(0.0, 4.0 * severity, (height, width, 1))
    combined = image.astype(np.float64) + texture + tint + show_through
    return np.clip(combined, 0, 255).astype(np.uint8)


def _photo(image: Any, severity: float, rng: Any) -> Any:
    """Model photographing a screen: perspective, uneven lighting, moire, noise."""

    import numpy as np
    from PIL import Image

    height, width = image.shape[:2]
    shifted = _rotate(image, severity * 0.4, rng).astype(np.float64)

    yy, xx = np.mgrid[0:height, 0:width]
    vignette = 1.0 - severity * 0.35 * (
        ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
    )
    shifted *= np.clip(vignette, 0.0, 1.0)[:, :, None]

    moire = 10.0 * severity * np.sin(yy / 2.3) * np.cos(xx / 3.1)
    shifted += moire[:, :, None]
    shifted += rng.normal(0.0, 18.0 * severity, shifted.shape)

    result = np.clip(shifted, 0, 255).astype(np.uint8)
    return np.array(Image.fromarray(result))


DEGRADATIONS: dict[str, DegradeFn] = {
    "clean": _identity,
    "jpeg": _jpeg,
    "downsample": _downsample,
    "rotate": _rotate,
    "noise": _noise,
    "blur": _blur,
    "paper": _paper,
    "photo": _photo,
}

# Ordered roughly by how much they disturb glyph shape.
LADDER: tuple[str, ...] = (
    "clean",
    "jpeg",
    "downsample",
    "rotate",
    "blur",
    "noise",
    "paper",
    "photo",
)

DEFAULT_SEVERITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def apply_degradation(image: Any, degradation: Degradation) -> Any:
    """Apply one degradation deterministically.

    Severity zero is the identity for every degradation, so the ladder always has
    a well-defined null rung.
    """

    import numpy as np

    if degradation.name not in DEGRADATIONS:
        known = ", ".join(sorted(DEGRADATIONS))
        raise DegradationError(f"Unknown degradation {degradation.name!r}. Known: {known}.")
    severity = _validate_severity(degradation.severity)
    if severity == 0.0:
        return image.copy()

    rng = np.random.default_rng(degradation.seed)
    return DEGRADATIONS[degradation.name](image, severity, rng)


def apply_chain(image: Any, degradations: list[Degradation]) -> Any:
    """Apply degradations in order, modelling a multi-stage handling path."""

    result = image
    for degradation in degradations:
        result = apply_degradation(result, degradation)
    return result


def build_ladder(
    *,
    names: tuple[str, ...] = LADDER,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    seed: int = 0,
) -> list[Degradation]:
    """Enumerate a (name, severity) grid as a flat list of coordinates.

    ``clean`` appears once at severity zero rather than once per severity, since
    every rung of it is the identity.
    """

    grid: list[Degradation] = []
    for name in names:
        if name == "clean":
            grid.append(Degradation(name="clean", severity=0.0, seed=seed))
            continue
        for severity in severities:
            grid.append(Degradation(name=name, severity=severity, seed=seed))
    return grid
