"""Translingua-TW: signed translation handoff with Taiwan-language invariants.

Binds a payload to semantic units and a declared translation operation.  The
source verifier produces an authenticated handoff; the target-language
verifier tests both provenance continuity and Taiwan terminology/factual
constraints.

Scientific status: skeleton.  A real implementation requires:
- T06 Taiwan corpus and native-review panel;
- locale-aware payload mapping (not token-level survival);
- factual, numeric, and citation invariants; and
- a preregistered translation operation registry.
This may become a provenance protocol rather than a purely statistical
watermark; that is acceptable if evaluated honestly.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal


class TranslinguaTWError(ValueError):
    """Raised when translation handoff construction or verification fails closed."""


@dataclass(frozen=True, slots=True)
class TranslationHandoff:
    """An authenticated source-to-target translation handoff."""

    handoff_id: str
    source_locale: str
    target_locale: str
    source_content_sha256: str
    target_content_sha256: str
    payload_ref: str
    translation_operation: str
    invariants: tuple[str, ...]
    signature: str
    signer: str
    timestamp: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaiwanTerminologyInvariant:
    """A protected Taiwan-language term that must not be converted to zh-CN."""

    term_id: str
    zh_hant_tw_form: str
    forbidden_zh_cn_form: str
    description: str


_TAIWAN_INVARIANTS = (
    TaiwanTerminologyInvariant(
        term_id="software",
        zh_hant_tw_form="軟體",
        forbidden_zh_cn_form="软件",
        description="Taiwan uses 軟體; Simplified 软件 is a zh-CN form.",
    ),
    TaiwanTerminologyInvariant(
        term_id="information",
        zh_hant_tw_form="資訊",
        forbidden_zh_cn_form="信息",
        description="Taiwan uses 資訊 for information; 信息 is a zh-CN form.",
    ),
    TaiwanTerminologyInvariant(
        term_id="network",
        zh_hant_tw_form="網路",
        forbidden_zh_cn_form="网络",
        description="Taiwan uses 網路; 网络 is a zh-CN form.",
    ),
    TaiwanTerminologyInvariant(
        term_id="video",
        zh_hant_tw_form="影片",
        forbidden_zh_cn_form="视频",
        description="Taiwan uses 影片; 视频 is a zh-CN form.",
    ),
    TaiwanTerminologyInvariant(
        term_id="computer",
        zh_hant_tw_form="電腦",
        forbidden_zh_cn_form="计算机",
        description="Taiwan uses 電腦; 计算机 is a zh-CN form in this context.",
    ),
)


@dataclass(frozen=True, slots=True)
class VerifiedHandoff:
    """Result of verifying a translation handoff."""

    scientific_status: Literal["skeleton-uncalibrated"]
    handoff_valid: bool
    provenance_continuous: bool
    taiwan_terminology_ok: bool
    violated_invariants: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "scientific_status": self.scientific_status,
            "handoff_valid": self.handoff_valid,
            "provenance_continuous": self.provenance_continuous,
            "taiwan_terminology_ok": self.taiwan_terminology_ok,
            "violated_invariants": list(self.violated_invariants),
            "warnings": list(self.warnings),
        }


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sign(payload: str, signing_key: str) -> str:
    return hmac.new(
        signing_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


class TranslinguaTWHandoffService:
    """Create and verify signed translation handoffs.

    The source verifier signs a handoff binding source content, target content,
    payload reference, translation operation, and invariants.  The target
    verifier checks signature integrity, provenance continuity (payload
    reference survives), and Taiwan terminology invariants (no zh-CN
    conversion).
    """

    def __init__(
        self,
        *,
        signing_key: str = "development-key-do-not-use-in-production",
        taiwan_invariants: Sequence[TaiwanTerminologyInvariant] = _TAIWAN_INVARIANTS,
    ) -> None:
        if not signing_key:
            raise TranslinguaTWError("A signing key is required for handoff integrity.")
        if not taiwan_invariants:
            raise TranslinguaTWError("At least one Taiwan terminology invariant is required.")
        self._signing_key = signing_key
        self._invariants = tuple(taiwan_invariants)

    @property
    def taiwan_invariants(self) -> tuple[TaiwanTerminologyInvariant, ...]:
        return self._invariants

    def create_handoff(
        self,
        *,
        handoff_id: str,
        source_locale: str,
        target_locale: str,
        source_content: str,
        target_content: str,
        payload_ref: str,
        translation_operation: str,
        invariants: Sequence[str],
        signer: str,
        timestamp: str,
    ) -> TranslationHandoff:
        if source_locale == target_locale:
            raise TranslinguaTWError("source and target locales must differ")
        self._validate_locale(source_locale)
        self._validate_locale(target_locale)
        if not translation_operation.strip():
            raise TranslinguaTWError("translation_operation must be nonempty")
        if not payload_ref.strip():
            raise TranslinguaTWError("payload_ref must be nonempty")
        signing_payload = self._signing_payload(
            handoff_id=handoff_id,
            source_hash=_content_hash(source_content),
            target_hash=_content_hash(target_content),
            payload_ref=payload_ref,
            translation_operation=translation_operation,
            invariants=invariants,
        )
        return TranslationHandoff(
            handoff_id=handoff_id,
            source_locale=source_locale,
            target_locale=target_locale,
            source_content_sha256=_content_hash(source_content),
            target_content_sha256=_content_hash(target_content),
            payload_ref=payload_ref,
            translation_operation=translation_operation,
            invariants=tuple(invariants),
            signature=_sign(signing_payload, self._signing_key),
            signer=signer,
            timestamp=timestamp,
        )

    def verify_handoff(
        self,
        handoff: TranslationHandoff,
        *,
        target_content: str,
    ) -> VerifiedHandoff:
        self._validate_locale(handoff.target_locale)
        signing_payload = self._signing_payload(
            handoff_id=handoff.handoff_id,
            source_hash=handoff.source_content_sha256,
            target_hash=handoff.target_content_sha256,
            payload_ref=handoff.payload_ref,
            translation_operation=handoff.translation_operation,
            invariants=handoff.invariants,
        )
        expected_sig = _sign(signing_payload, self._signing_key)
        handoff_valid = hmac.compare_digest(handoff.signature, expected_sig)
        content_matches = (
            _content_hash(target_content) == handoff.target_content_sha256
        )
        provenance_continuous = handoff_valid and content_matches and bool(handoff.payload_ref)
        violated_terms = self._check_taiwan_terminology(target_content)
        taiwan_ok = len(violated_terms) == 0

        warnings: list[str] = [
            "Skeleton translation handoff: no real payload embedding or cross-lingual detection.",
            "Scientific handoff requires T06 Taiwan corpus and native-review panel.",
        ]
        if not handoff_valid:
            warnings.append("Handoff signature invalid.")
        if not content_matches:
            warnings.append("Target content hash does not match handoff.")
        if not taiwan_ok:
            warnings.append(
                f"Taiwan terminology violation: {', '.join(violated_terms)} "
                "contains zh-CN forms."
            )

        return VerifiedHandoff(
            scientific_status="skeleton-uncalibrated",
            handoff_valid=handoff_valid,
            provenance_continuous=provenance_continuous,
            taiwan_terminology_ok=taiwan_ok,
            violated_invariants=tuple(violated_terms),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _validate_locale(locale: str) -> None:
        from watermark_lab.i18n.policy import LocalePolicyError, canonicalize_locale

        try:
            canonicalize_locale(locale)
        except LocalePolicyError as exc:
            raise TranslinguaTWError(str(exc)) from exc

    @staticmethod
    def _signing_payload(
        *,
        handoff_id: str,
        source_hash: str,
        target_hash: str,
        payload_ref: str,
        translation_operation: str,
        invariants: Sequence[str],
    ) -> str:
        return "|".join(
            [
                handoff_id,
                source_hash,
                target_hash,
                payload_ref,
                translation_operation,
                ",".join(invariants),
            ]
        )

    def _check_taiwan_terminology(self, text: str) -> list[str]:
        """Return term IDs whose forbidden zh-CN forms appear in the text."""

        violated: list[str] = []
        for inv in self._invariants:
            if inv.forbidden_zh_cn_form in text:
                violated.append(inv.term_id)
        return violated


@dataclass(frozen=True, slots=True)
class FactualInvariant:
    """A protected factual element that must be preserved across translation."""

    invariant_id: str
    kind: Literal["number", "citation", "quotation", "date", "url", "name"]
    source_form: str
    target_form: str
    description: str


def check_factual_invariants(
    target_text: str,
    invariants: Sequence[FactualInvariant],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check factual invariants in the target text.

    Returns (violated_ids, intact_ids).  An invariant is violated if its
    ``target_form`` is not found in the text.
    """

    violated: list[str] = []
    intact: list[str] = []
    for inv in invariants:
        if inv.target_form in target_text:
            intact.append(inv.invariant_id)
        else:
            violated.append(inv.invariant_id)
    return tuple(violated), tuple(intact)


def create_factual_handoff(
    service: TranslinguaTWHandoffService,
    *,
    handoff_id: str,
    source_content: str,
    target_content: str,
    payload_ref: str,
    translation_operation: str,
    factual_invariants: Sequence[FactualInvariant],
    signer: str,
    timestamp: str,
) -> tuple[TranslationHandoff, VerifiedHandoff]:
    """Create and verify a translation handoff with factual invariants.

    This extends the basic terminology check with numeric, citation, date,
    URL, and name preservation checks.  The handoff is signed and verified
    in one call for convenience.
    """

    invariant_ids = [inv.invariant_id for inv in factual_invariants]
    handoff = service.create_handoff(
        handoff_id=handoff_id,
        source_locale="en",
        target_locale="zh-Hant-TW",
        source_content=source_content,
        target_content=target_content,
        payload_ref=payload_ref,
        translation_operation=translation_operation,
        invariants=invariant_ids,
        signer=signer,
        timestamp=timestamp,
    )
    verification = service.verify_handoff(handoff, target_content=target_content)
    violated_facts, _intact = check_factual_invariants(target_content, factual_invariants)
    if violated_facts:
        extra_warning = f"Factual invariants violated: {', '.join(violated_facts)}"
        verification = VerifiedHandoff(
            scientific_status=verification.scientific_status,
            handoff_valid=verification.handoff_valid,
            provenance_continuous=verification.provenance_continuous,
            taiwan_terminology_ok=verification.taiwan_terminology_ok,
            violated_invariants=(*verification.violated_invariants, *violated_facts),
            warnings=(*verification.warnings, extra_warning),
        )
    return handoff, verification


__all__ = [
    "FactualInvariant",
    "TaiwanTerminologyInvariant",
    "TranslationHandoff",
    "TranslinguaTWError",
    "TranslinguaTWHandoffService",
    "VerifiedHandoff",
    "check_factual_invariants",
    "create_factual_handoff",
]