"""Small built-in catalog for the initial desktop product shell."""

from __future__ import annotations

from watermark_lab.i18n.policy import canonicalize_locale

_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        "runs": "Runs",
        "artifact_history": "Artifact History",
        "settings": "Settings",
        "refresh": "Refresh",
        "locale": "Locale",
        "history_mode": "Artifact history mode",
    },
    "zh-Hant-TW": {
        "runs": "執行紀錄",
        "artifact_history": "成品歷程",
        "settings": "設定",
        "refresh": "重新整理",
        "locale": "語言",
        "history_mode": "成品歷程模式",
    },
}


def translate(key: str, locale: str | None = None) -> str:
    """Translate a product-shell key, falling back only to English."""

    canonical = canonicalize_locale(locale)
    return _CATALOG.get(canonical, {}).get(key, _CATALOG["en"].get(key, key))
