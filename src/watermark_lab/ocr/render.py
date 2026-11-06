"""Deterministic text-to-page rendering for OCR round-trip experiments.

Rendering is half of the instrument. Font, size and DPI change the character
error rate as much as the degradation does, so every rendering parameter is
recorded alongside the result and the whole render is reproducible from its
declared settings.

Font selection fails closed for CJK. Rendering Traditional Chinese with a font
that lacks the glyphs produces tofu boxes, which OCR then reads as noise — a
silent path to a fabricated "watermark destroyed" result. A locale that needs
CJK coverage and cannot get it raises instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watermark_lab.ocr.profile import digest_file

# Ordered by preference. Traditional-Chinese-capable faces first for zh-Hant.
_CJK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msjh.ttc",       # Microsoft JhengHei (Traditional, Taiwan)
    "C:/Windows/Fonts/mingliub.ttc",   # MingLiU-ExtB (Traditional)
    "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei (Simplified; last resort)
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_LATIN_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)

_SIMPLIFIED_ONLY_FONTS = ("msyh.ttc", "simsun.ttc", "simhei.ttf")


class RenderError(RuntimeError):
    """Raised when a page cannot be rendered reproducibly."""


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """Reproducible page-rendering parameters."""

    font_path: str
    font_size: int = 28
    dpi: int = 300
    page_width: int = 1654
    margin: int = 96
    line_spacing: float = 1.5
    background: str = "white"
    foreground: str = "black"
    font_sha256: str = ""
    warnings: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "font_path": self.font_path,
            "font_name": Path(self.font_path).name,
            "font_sha256": self.font_sha256,
            "font_size": self.font_size,
            "dpi": self.dpi,
            "page_width": self.page_width,
            "margin": self.margin,
            "line_spacing": self.line_spacing,
            "background": self.background,
            "foreground": self.foreground,
            "warnings": list(self.warnings),
        }


def _needs_cjk(text: str) -> bool:
    """Return whether ``text`` contains CJK ideographs."""

    return any(
        "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff" or "\u3000" <= ch <= "\u303f"
        for ch in text
    )


def select_font(text: str, *, locale: str | None = None) -> tuple[str, tuple[str, ...]]:
    """Choose a font that can actually render ``text``.

    Raises when CJK coverage is required and unavailable, rather than rendering
    replacement boxes that OCR would read as noise.
    """

    wants_cjk = _needs_cjk(text) or (locale or "").lower().startswith(("zh", "ja", "ko"))
    candidates = _CJK_FONT_CANDIDATES if wants_cjk else _LATIN_FONT_CANDIDATES
    warnings: list[str] = []

    for candidate in candidates:
        if Path(candidate).is_file():
            name = Path(candidate).name.lower()
            if wants_cjk and name in _SIMPLIFIED_ONLY_FONTS:
                warnings.append(
                    f"font {name!r} is a Simplified Chinese face; Traditional glyph coverage "
                    "may be incomplete and rendered forms may not match Taiwan usage"
                )
            return candidate, tuple(warnings)

    if wants_cjk:
        raise RenderError(
            "No CJK-capable font was found, so Traditional Chinese cannot be rendered. "
            "Rendering it with a Latin-only font would produce replacement boxes that OCR "
            "reads as noise, fabricating a 'watermark destroyed' result. Install a "
            "Traditional Chinese font (Microsoft JhengHei, MingLiU, or Noto CJK)."
        )
    raise RenderError("No usable font was found for Latin text rendering.")


def build_render_settings(
    text: str,
    *,
    locale: str | None = None,
    font_path: str | None = None,
    font_size: int = 28,
    dpi: int = 300,
    page_width: int = 1654,
    margin: int = 96,
    line_spacing: float = 1.5,
) -> RenderSettings:
    """Resolve and hash a font, returning fully-specified render settings."""

    warnings: tuple[str, ...] = ()
    if font_path is None:
        font_path, warnings = select_font(text, locale=locale)
    elif not Path(font_path).is_file():
        raise RenderError(f"font_path does not exist: {font_path}")

    font_sha256, _size = digest_file(Path(font_path))
    return RenderSettings(
        font_path=font_path,
        font_size=font_size,
        dpi=dpi,
        page_width=page_width,
        margin=margin,
        line_spacing=line_spacing,
        font_sha256=font_sha256,
        warnings=warnings,
    )


def _wrap(text: str, font: Any, max_width: int, draw: Any) -> list[str]:
    """Wrap text to ``max_width`` pixels.

    CJK has no inter-word spaces, so whitespace wrapping alone would emit one
    enormous line. Paragraphs without spaces are wrapped per character.
    """

    def width_of(candidate: str) -> float:
        return float(draw.textlength(candidate, font=font))

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue

        units = paragraph.split(" ") if " " in paragraph else list(paragraph)
        joiner = " " if " " in paragraph else ""
        current = ""
        for unit in units:
            candidate = f"{current}{joiner}{unit}" if current else unit
            if current and width_of(candidate) > max_width:
                lines.append(current)
                current = unit
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def render_text_to_image(text: str, settings: RenderSettings) -> Any:
    """Render ``text`` onto a white page and return an RGB numpy array."""

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(settings.font_path, settings.font_size)
    except OSError as exc:
        raise RenderError(f"Could not load font {settings.font_path}: {exc}") from exc

    probe = Image.new("RGB", (settings.page_width, 10), settings.background)
    probe_draw = ImageDraw.Draw(probe)
    usable_width = settings.page_width - 2 * settings.margin
    if usable_width <= 0:
        raise RenderError("margin is too large for the requested page width.")

    lines = _wrap(text, font, usable_width, probe_draw) or [""]
    line_height = int(settings.font_size * settings.line_spacing)
    page_height = 2 * settings.margin + line_height * len(lines)

    page = Image.new("RGB", (settings.page_width, page_height), settings.background)
    draw = ImageDraw.Draw(page)
    y = settings.margin
    for line in lines:
        draw.text((settings.margin, y), line, font=font, fill=settings.foreground)
        y += line_height

    return np.array(page)


def render_fingerprint(text: str, settings: RenderSettings) -> str:
    """Return a hash binding the source text to its rendering parameters."""

    payload = repr((text, settings.as_dict())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
