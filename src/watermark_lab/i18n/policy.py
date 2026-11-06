"""Locale policy for the product surface.

Traditional Chinese is the only Chinese script variant supported by the lab.
Simplified Chinese locales are rejected instead of being silently negotiated.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LOCALE = "en"
TRADITIONAL_CHINESE_LOCALE = "zh-Hant-TW"
TRADITIONAL_CHINESE_ALIAS = "zh-TW"
SUPPORTED_LOCALES = (DEFAULT_LOCALE, TRADITIONAL_CHINESE_LOCALE)
TRADITIONAL_CHINESE_FALLBACK_CHAIN = (
    TRADITIONAL_CHINESE_LOCALE,
    TRADITIONAL_CHINESE_ALIAS,
    "zh-Hant",
    DEFAULT_LOCALE,
)


class LocalePolicyError(ValueError):
    """Raised when a locale violates the explicit product policy."""


def _locale_key(locale: str) -> str:
    return locale.strip().replace("_", "-").lower()


def _reject_simplified_chinese(locale: str) -> None:
    key = _locale_key(locale)
    if key in {"zh-cn", "zh-hans", "zh-sg"} or key.startswith(
        ("zh-cn-", "zh-hans-", "zh-sg-")
    ):
        raise LocalePolicyError(
            f"Simplified Chinese locale {locale!r} is prohibited; "
            "use 'zh-Hant-TW' (or the 'zh-TW' alias)."
        )


def canonicalize_locale(locale: str | None) -> str:
    """Return a supported canonical locale, explicitly rejecting zh-CN.

    Empty values select English. ``zh-TW`` and common case/underscore variants
    are accepted as aliases for the canonical ``zh-Hant-TW`` locale.
    """

    if locale is None or not locale.strip():
        return DEFAULT_LOCALE
    _reject_simplified_chinese(locale)
    key = _locale_key(locale)
    if key == "en" or key.startswith("en-"):
        return DEFAULT_LOCALE
    if key in {"zh-tw", "zh-hant", "zh-hant-tw"}:
        return TRADITIONAL_CHINESE_LOCALE
    raise LocalePolicyError(
        f"Unsupported locale {locale!r}; choose 'en', 'zh-Hant-TW', or the 'zh-TW' alias."
    )


def fallback_chain(locale: str | None) -> tuple[str, ...]:
    """Return the complete, safe resource lookup chain for ``locale``."""

    canonical = canonicalize_locale(locale)
    if canonical == TRADITIONAL_CHINESE_LOCALE:
        return TRADITIONAL_CHINESE_FALLBACK_CHAIN
    return (DEFAULT_LOCALE,)


def validate_fallbacks(locales: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a caller-supplied resource chain and reject any zh-CN entry."""

    for locale in locales:
        _reject_simplified_chinese(locale)
    return tuple(locales)


@dataclass(frozen=True, slots=True)
class LocalePolicy:
    """Serializable description shown by the CLI and Settings view."""

    canonical_traditional_chinese: str = TRADITIONAL_CHINESE_LOCALE
    accepted_alias: str = TRADITIONAL_CHINESE_ALIAS
    default_locale: str = DEFAULT_LOCALE
    fallback_chain: tuple[str, ...] = TRADITIONAL_CHINESE_FALLBACK_CHAIN
    simplified_chinese_allowed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_traditional_chinese": self.canonical_traditional_chinese,
            "accepted_alias": self.accepted_alias,
            "default_locale": self.default_locale,
            "fallback_chain": list(self.fallback_chain),
            "simplified_chinese_allowed": self.simplified_chinese_allowed,
        }
