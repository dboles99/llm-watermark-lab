"""Detect mainland-Chinese vocabulary hiding inside Traditional-Chinese glyphs.

The problem this solves is easy to miss and fatal to a `zh-Hant-TW` corpus claim.

Converting Simplified Chinese to Traditional with a character-level mapping
(OpenCC ``s2t`` / ``s2tw``, or MediaWiki's variant tables) changes the *glyphs*
and nothing else. The result passes every script check: every character is a
valid Traditional form, so a Hant/Hans detector reports Traditional and moves on.
But the *vocabulary* is untouched. The converted text still says 軟件, 網絡, 信息,
視頻 where Taiwan writes 軟體, 網路, 資訊, 影片.

Only the phrase-aware conversions (``s2twp`` / ``tw2sp``) touch vocabulary, and
pipelines that mechanically convert a corpus rarely use them. So the most common
contamination in a "Traditional Chinese" dataset is invisible to the most common
check.

This module screens for that directly, using a curated regional-term lexicon. It
is deliberately dependency-free: OpenCC is not required, so screening can run
inside the offline evidence path and inside a TEE.

The sweep of 2026-08-30 found no published tool doing this as corpus hygiene, so
this is offered as a small contribution rather than a reimplementation.

**Scope and honesty.** A lexicon match is evidence of mainland usage, not proof
of mechanical conversion: a Taiwan author may quote a mainland source, and some
terms are genuinely used in both regions. The output is a graded signal with the
matched evidence attached, never a verdict. Absence of matches is much weaker
evidence than presence, because a short text may simply contain no discriminating
term — ``screen`` reports that case explicitly rather than returning "clean".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["taiwan-consistent", "mainland-indicators", "mixed", "insufficient-evidence"]

# Curated (mainland term, Taiwan term, gloss) triples. Chosen so that each
# mainland form is *not* standard Taiwan usage, and each pair survives a
# character-level s2t conversion unchanged — which is precisely what makes them
# discriminating.
REGIONAL_TERM_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("軟件", "軟體", "software"),
    ("硬件", "硬體", "hardware"),
    ("網絡", "網路", "network"),
    ("信息", "資訊", "information"),
    ("視頻", "影片", "video"),
    ("音頻", "音訊", "audio"),
    ("數據庫", "資料庫", "database"),
    ("互聯網", "網際網路", "internet"),
    ("默認", "預設", "default"),
    ("內存", "記憶體", "memory"),
    ("鼠標", "滑鼠", "mouse"),
    ("屏幕", "螢幕", "screen"),
    ("打印機", "印表機", "printer"),
    ("激光", "雷射", "laser"),
    ("菜單", "選單", "menu"),
    ("文件夾", "資料夾", "folder"),
    ("服務器", "伺服器", "server"),
    ("端口", "連接埠", "port"),
    ("編程", "程式設計", "programming"),
    ("項目", "專案", "project"),
    ("質量", "品質", "quality"),
    ("水平", "水準", "level"),
    ("渠道", "管道", "channel"),
    ("優化", "最佳化", "optimisation"),
    ("加載", "載入", "load"),
    ("緩存", "快取", "cache"),
    ("集成", "整合", "integration"),
    ("調試", "除錯", "debugging"),
    ("估計", "預估", "estimate"),
    ("智能", "智慧", "intelligent"),
    ("移動電話", "行動電話", "mobile phone"),
    ("計算機", "電腦", "computer"),
    ("信號", "訊號", "signal"),
    ("列表", "清單", "list"),
    ("進程", "行程", "process"),
    ("線程", "執行緒", "thread"),
    ("字符", "字元", "character"),
    ("矢量", "向量", "vector"),
    ("郵件", "郵件", "mail (shared; not discriminating)"),
)

# Characters that exist only in Simplified Chinese. Their presence means the text
# was never converted at all, which is a different and more obvious failure.
SIMPLIFIED_ONLY_CHARACTERS = frozenset(
    "书东车间问长应门这来时说话对开关国实发学习点无产两个们"
    "从会条业务员经济认识汉语让讲义乐买卖头须风飞马鸟鱼见贝"
)

_MAINLAND_TERMS = tuple(
    mainland for mainland, taiwan, gloss in REGIONAL_TERM_PAIRS if "shared" not in gloss
)
_TAIWAN_TERMS = tuple(
    taiwan for mainland, taiwan, gloss in REGIONAL_TERM_PAIRS if "shared" not in gloss
)
_GLOSS_BY_MAINLAND = {
    mainland: gloss for mainland, taiwan, gloss in REGIONAL_TERM_PAIRS if "shared" not in gloss
}
_TAIWAN_BY_MAINLAND = {
    mainland: taiwan for mainland, taiwan, gloss in REGIONAL_TERM_PAIRS if "shared" not in gloss
}

_CJK_PATTERN = re.compile(r"[㐀-鿿豈-﫿]")


@dataclass(frozen=True, slots=True)
class TermHit:
    """One matched regional term with its Taiwan counterpart."""

    term: str
    count: int
    taiwan_equivalent: str
    gloss: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "count": self.count,
            "taiwan_equivalent": self.taiwan_equivalent,
            "gloss": self.gloss,
        }


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """A graded contamination signal with the evidence attached."""

    verdict: Verdict
    cjk_characters: int
    mainland_hits: tuple[TermHit, ...]
    taiwan_hits: tuple[TermHit, ...]
    simplified_characters: tuple[str, ...]
    note: str
    warnings: tuple[str, ...] = field(default=())

    @property
    def mainland_total(self) -> int:
        return sum(hit.count for hit in self.mainland_hits)

    @property
    def taiwan_total(self) -> int:
        return sum(hit.count for hit in self.taiwan_hits)

    @property
    def discriminating_total(self) -> int:
        return self.mainland_total + self.taiwan_total

    @property
    def mainland_ratio(self) -> float | None:
        """Share of discriminating terms that are mainland forms.

        ``None`` when no discriminating term occurs, which is a different state
        from "zero mainland usage" and must not be rendered as 0.0.
        """

        if self.discriminating_total == 0:
            return None
        return self.mainland_total / self.discriminating_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "cjk_characters": self.cjk_characters,
            "mainland_hits": [hit.as_dict() for hit in self.mainland_hits],
            "taiwan_hits": [hit.as_dict() for hit in self.taiwan_hits],
            "mainland_total": self.mainland_total,
            "taiwan_total": self.taiwan_total,
            "mainland_ratio": self.mainland_ratio,
            "simplified_characters": list(self.simplified_characters),
            "note": self.note,
            "warnings": list(self.warnings),
        }


def _count_terms(text: str, terms: tuple[str, ...]) -> list[TermHit]:
    hits: list[TermHit] = []
    for term in terms:
        count = text.count(term)
        if count:
            hits.append(
                TermHit(
                    term=term,
                    count=count,
                    taiwan_equivalent=_TAIWAN_BY_MAINLAND.get(term, term),
                    gloss=_GLOSS_BY_MAINLAND.get(term, ""),
                )
            )
    return sorted(hits, key=lambda hit: (-hit.count, hit.term))


def screen(text: str, *, minimum_cjk: int = 50) -> ScreeningResult:
    """Screen text for mainland vocabulary carried in Traditional glyphs.

    ``minimum_cjk`` is the length below which the result is reported as
    ``insufficient-evidence`` regardless of what was found, because a short
    passage may simply contain no discriminating term.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalised = unicodedata.normalize("NFC", text)
    cjk_characters = len(_CJK_PATTERN.findall(normalised))
    simplified = tuple(sorted({ch for ch in normalised if ch in SIMPLIFIED_ONLY_CHARACTERS}))
    mainland_hits = tuple(_count_terms(normalised, _MAINLAND_TERMS))
    taiwan_hits = tuple(_count_terms(normalised, _TAIWAN_TERMS))

    warnings: list[str] = []
    if simplified:
        warnings.append(
            "Simplified-only characters are present, so this text was not converted to "
            "Traditional at all. This is a script failure, not a vocabulary one."
        )

    mainland_total = sum(hit.count for hit in mainland_hits)
    taiwan_total = sum(hit.count for hit in taiwan_hits)

    if cjk_characters < minimum_cjk or (mainland_total == 0 and taiwan_total == 0):
        verdict: Verdict = "insufficient-evidence"
        note = (
            "No discriminating regional term occurred, or the passage is too short. "
            "This is NOT evidence of Taiwan origin: absence of a signal is much weaker "
            "than its presence."
        )
    elif mainland_total and taiwan_total:
        verdict = "mixed"
        note = (
            "Both mainland and Taiwan regional forms occur. This may indicate partial "
            "conversion, quoted mainland sources, or mixed authorship. Inspect the hits."
        )
    elif mainland_total:
        verdict = "mainland-indicators"
        note = (
            "Mainland vocabulary occurs in Traditional glyphs, which is the signature of "
            "character-level conversion (s2t / s2tw) from a Simplified source. A "
            "script-level Hant check would pass this text."
        )
    else:
        verdict = "taiwan-consistent"
        note = (
            "Only Taiwan regional forms occurred among discriminating terms. Consistent "
            "with Taiwan origin; not proof of it."
        )

    return ScreeningResult(
        verdict=verdict,
        cjk_characters=cjk_characters,
        mainland_hits=mainland_hits,
        taiwan_hits=taiwan_hits,
        simplified_characters=simplified,
        note=note,
        warnings=tuple(warnings),
    )


def screen_corpus(
    documents: dict[str, str], *, minimum_cjk: int = 50
) -> dict[str, Any]:
    """Screen a corpus and summarise which documents to exclude or review.

    Documents are reported by verdict so a curator can act on them, rather than
    receiving a single aggregate score that hides which items are contaminated.
    """

    if not documents:
        raise ValueError("documents must not be empty.")

    results = {doc_id: screen(text, minimum_cjk=minimum_cjk) for doc_id, text in documents.items()}
    by_verdict: dict[str, list[str]] = {}
    for doc_id, result in results.items():
        by_verdict.setdefault(result.verdict, []).append(doc_id)

    contaminated = sorted(by_verdict.get("mainland-indicators", []) + by_verdict.get("mixed", []))
    return {
        "document_count": len(documents),
        "by_verdict": {key: sorted(value) for key, value in sorted(by_verdict.items())},
        "recommended_exclusions": contaminated,
        "results": {doc_id: result.as_dict() for doc_id, result in results.items()},
        "note": (
            "A character-level Hant/Hans check passes mechanically converted text, because "
            "conversion changes glyphs and not vocabulary. Documents flagged here would be "
            "invisible to a script-level screen."
        ),
    }


__all__ = [
    "REGIONAL_TERM_PAIRS",
    "SIMPLIFIED_ONLY_CHARACTERS",
    "ScreeningResult",
    "TermHit",
    "Verdict",
    "screen",
    "screen_corpus",
]
