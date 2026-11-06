"""Robust-plus-fragile semantic integrity watermark skeleton.

Embeds two authenticated channels:
- a robust provenance channel that survives benign copyediting, formatting,
  and approved translation; and
- a fragile semantic channel tied to protected claims, numbers, citations,
  quotations, or locale-sensitive terminology that intentionally breaks or
  localizes when those invariants change.

The research question is whether transformation-conditioned channel behavior
can distinguish benign editorial drift from consequential semantic tampering.

Scientific status: skeleton/high-risk.  This module provides the data
contracts and a prototype invariant-checking mechanism.  A real implementation
requires:
- a preregistered semantic edit ontology (benign vs material);
- semantic canonicalization that is robust to attack;
- dual-channel embedding that preserves generation quality; and
- T06 calibrated detectors for both channels.
Start with constrained factual templates before open-domain prose.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Literal


class RobustFragileError(ValueError):
    """Raised when dual-channel construction or verification fails closed."""


class InvariantKind(StrEnum):
    NUMBER = "number"
    CITATION = "citation"
    QUOTATION = "quotation"
    LOCALE_TERM = "locale-term"
    FACTUAL_CLAIM = "factual-claim"


@dataclass(frozen=True, slots=True)
class SemanticInvariant:
    """A protected semantic element that the fragile channel monitors."""

    invariant_id: str
    kind: InvariantKind
    pattern: str
    description: str

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class RobustChannelState:
    """State of the robust provenance channel."""

    payload_hash: str
    embedded: bool
    detection_score: float | None
    survives: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FragileChannelState:
    """State of the fragile semantic-integrity channel."""

    invariant_hashes: tuple[str, ...]
    intact: bool
    violated_invariants: tuple[str, ...]
    localization_spans: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "invariant_hashes": list(self.invariant_hashes),
            "intact": self.intact,
            "violated_invariants": list(self.violated_invariants),
            "localization_spans": [list(s) for s in self.localization_spans],
        }


@dataclass(frozen=True, slots=True)
class RobustFragileResult:
    """Combined dual-channel verification result."""

    scientific_status: Literal["skeleton-uncalibrated"]
    robust_channel: RobustChannelState
    fragile_channel: FragileChannelState
    classification: Literal[
        "provenance-intact-semantics-intact",
        "provenance-intact-semantics-violated",
        "provenance-lost-semantics-intact",
        "provenance-lost-semantics-violated",
        "inconclusive",
    ]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "scientific_status": self.scientific_status,
            "robust_channel": self.robust_channel.as_dict(),
            "fragile_channel": self.fragile_channel.as_dict(),
            "classification": self.classification,
            "warnings": list(self.warnings),
        }


class RobustFragileWatermarker:
    """Prototype dual-channel semantic integrity watermarker.

    The robust channel is a payload hash that survives text formatting changes.
    The fragile channel hashes protected semantic invariants (numbers,
    citations, quotations, locale terms) and reports violations with
    localization spans.

    This is a skeleton: it does not embed watermarks into model logits.
    It checks invariant preservation on the text surface, which is the
    first step before a real embedding mechanism is built.
    """

    def __init__(self, invariants: Sequence[SemanticInvariant]) -> None:
        if not invariants:
            raise RobustFragileError("At least one semantic invariant is required.")
        ids = [inv.invariant_id for inv in invariants]
        if len(ids) != len(set(ids)):
            raise RobustFragileError("Invariant IDs must be unique.")
        self._invariants = tuple(invariants)

    @property
    def invariants(self) -> tuple[SemanticInvariant, ...]:
        return self._invariants

    def compute_fragile_state(self, text: str) -> FragileChannelState:
        """Check each invariant against the text surface."""

        invariant_hashes: list[str] = []
        violated: list[str] = []
        spans: list[tuple[int, int]] = []

        for inv in self._invariants:
            inv_hash = hashlib.sha256(
                f"{inv.invariant_id}:{inv.kind.value}:{inv.pattern}".encode()
            ).hexdigest()
            invariant_hashes.append(inv_hash)
            match = re.search(inv.pattern, text)
            if match is None:
                violated.append(inv.invariant_id)
            else:
                spans.append((match.start(), match.end()))

        intact = len(violated) == 0
        return FragileChannelState(
            invariant_hashes=tuple(invariant_hashes),
            intact=intact,
            violated_invariants=tuple(violated),
            localization_spans=tuple(s for s in spans if s != (-1, -1)),
        )

    def verify(
        self,
        *,
        text: str,
        payload: str,
        robust_survives: bool = True,
        robust_score: float | None = None,
    ) -> RobustFragileResult:
        """Classify a document against both channels."""

        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        robust = RobustChannelState(
            payload_hash=payload_hash,
            embedded=True,
            detection_score=robust_score,
            survives=robust_survives,
        )
        fragile = self.compute_fragile_state(text)

        if robust.survives and fragile.intact:
            classification: str = "provenance-intact-semantics-intact"
        elif robust.survives and not fragile.intact:
            classification = "provenance-intact-semantics-violated"
        elif not robust.survives and fragile.intact:
            classification = "provenance-lost-semantics-intact"
        elif not robust.survives and not fragile.intact:
            classification = "provenance-lost-semantics-violated"
        else:
            classification = "inconclusive"

        warnings: list[str] = [
            "Skeleton dual-channel verifier: no real watermark embedding.",
            "Semantic canonicalization is attackable; this is surface pattern matching.",
            "Scientific dual-channel requires T06 calibrated detectors and a preregistered edit ontology.",
        ]
        if not fragile.intact:
            warnings.append(
                f"Fragile channel violated {len(fragile.violated_invariants)} invariant(s): "
                f"{', '.join(fragile.violated_invariants)}"
            )

        return RobustFragileResult(
            scientific_status="skeleton-uncalibrated",
            robust_channel=robust,
            fragile_channel=fragile,
            classification=classification,  # type: ignore[arg-type]
            warnings=tuple(warnings),
        )


class EditClass(StrEnum):
    """Preregistered edit classes for the robust/fragile ontology."""

    BENIGN_PUNCTUATION = "benign-punctuation"
    BENIGN_WHITESPACE = "benign-whitespace"
    BENIGN_FORMATTING = "benign-formatting"
    BENIGN_TYPO_FIX = "benign-typo-fix"
    MATERIAL_NUMBER_CHANGE = "material-number-change"
    MATERIAL_CITATION_CHANGE = "material-citation-change"
    MATERIAL_QUOTATION_CHANGE = "material-quotation-change"
    MATERIAL_LOCALE_TERM_CHANGE = "material-locale-term-change"
    MATERIAL_FACTUAL_CLAIM_CHANGE = "material-factual-claim-change"
    MATERIAL_TRANSLATION = "material-translation"
    MATERIAL_PARAPHRASE = "material-paraphrase"


_BENIGN_CLASSES = frozenset({
    EditClass.BENIGN_PUNCTUATION,
    EditClass.BENIGN_WHITESPACE,
    EditClass.BENIGN_FORMATTING,
    EditClass.BENIGN_TYPO_FIX,
})

_MATERIAL_CLASSES = frozenset({
    EditClass.MATERIAL_NUMBER_CHANGE,
    EditClass.MATERIAL_CITATION_CHANGE,
    EditClass.MATERIAL_QUOTATION_CHANGE,
    EditClass.MATERIAL_LOCALE_TERM_CHANGE,
    EditClass.MATERIAL_FACTUAL_CLAIM_CHANGE,
    EditClass.MATERIAL_TRANSLATION,
    EditClass.MATERIAL_PARAPHRASE,
})


def is_benign_edit(edit_class: EditClass | str) -> bool:
    """Return True if the edit class is in the preregistered benign set."""

    cls = EditClass(edit_class) if isinstance(edit_class, str) else edit_class
    return cls in _BENIGN_CLASSES


def is_material_edit(edit_class: EditClass | str) -> bool:
    """Return True if the edit class is in the preregistered material set."""

    cls = EditClass(edit_class) if isinstance(edit_class, str) else edit_class
    return cls in _MATERIAL_CLASSES


@dataclass(frozen=True, slots=True)
class EditOntology:
    """Preregistered edit ontology mapping edit classes to expected channel behavior.

    Benign edits should preserve both the robust and fragile channels.
    Material edits should preserve the robust channel but break the fragile
    channel (or localize the violation).
    """

    benign_classes: tuple[EditClass, ...] = tuple(sorted(_BENIGN_CLASSES))
    material_classes: tuple[EditClass, ...] = tuple(sorted(_MATERIAL_CLASSES))

    def classify(self, edit_class: EditClass | str) -> Literal["benign", "material", "unknown"]:
        try:
            cls = EditClass(edit_class) if isinstance(edit_class, str) else edit_class
        except ValueError:
            return "unknown"
        if cls in _BENIGN_CLASSES:
            return "benign"
        if cls in _MATERIAL_CLASSES:
            return "material"
        return "unknown"

    def expected_behavior(self, edit_class: EditClass | str) -> dict[str, str]:
        kind = self.classify(edit_class)
        if kind == "benign":
            return {
                "robust_channel": "survives",
                "fragile_channel": "intact",
                "classification": "provenance-intact-semantics-intact",
            }
        if kind == "material":
            return {
                "robust_channel": "survives",
                "fragile_channel": "violated",
                "classification": "provenance-intact-semantics-violated",
            }
        return {
            "robust_channel": "unknown",
            "fragile_channel": "unknown",
            "classification": "inconclusive",
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "benign_classes": [c.value for c in self.benign_classes],
            "material_classes": [c.value for c in self.material_classes],
        }


__all__ = [
    "EditClass",
    "EditOntology",
    "FragileChannelState",
    "InvariantKind",
    "RobustChannelState",
    "RobustFragileError",
    "RobustFragileResult",
    "RobustFragileWatermarker",
    "SemanticInvariant",
    "is_benign_edit",
    "is_material_edit",
]