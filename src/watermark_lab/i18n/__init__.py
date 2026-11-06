"""Internationalization helpers with a strict Traditional Chinese policy."""

from watermark_lab.i18n.catalog import translate
from watermark_lab.i18n.policy import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    TRADITIONAL_CHINESE_ALIAS,
    TRADITIONAL_CHINESE_FALLBACK_CHAIN,
    TRADITIONAL_CHINESE_LOCALE,
    LocalePolicy,
    LocalePolicyError,
    canonicalize_locale,
    fallback_chain,
    validate_fallbacks,
)

__all__ = [
    "DEFAULT_LOCALE",
    "SUPPORTED_LOCALES",
    "TRADITIONAL_CHINESE_ALIAS",
    "TRADITIONAL_CHINESE_FALLBACK_CHAIN",
    "TRADITIONAL_CHINESE_LOCALE",
    "LocalePolicy",
    "LocalePolicyError",
    "canonicalize_locale",
    "fallback_chain",
    "translate",
    "validate_fallbacks",
]
