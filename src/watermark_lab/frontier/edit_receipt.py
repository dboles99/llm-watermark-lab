"""EditReceipt: signed content-addressed receipt DAG with counterfactual certificates.

Represents generation, edit, translation, detection, calibration, and review
events as a directed acyclic graph of content-addressed receipts.  For each
transformation, compute a counterfactual certificate describing which allowed
edit classes and severity regions would retain detection at the preregistered
threshold.

Scientific status: skeleton/prototype.  The certificate covers only the
declared transformation family and must never be presented as universal
robustness.  A scientific certificate requires T06 calibrated detectors and
a preregistered transformation registry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class EditReceiptError(ValueError):
    """Raised when receipt DAG construction or verification fails closed."""


class ReceiptEventKind(StrEnum):
    GENERATION = "generation"
    EDIT = "edit"
    TRANSLATION = "translation"
    DETECTION = "detection"
    CALIBRATION = "calibration"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ReceiptEvent:
    """One immutable event in the lifecycle DAG."""

    event_id: str
    kind: ReceiptEventKind
    content_sha256: str
    parent_event_ids: tuple[str, ...]
    actor: str
    timestamp: str
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "kind": self.kind.value,
        }


@dataclass(frozen=True, slots=True)
class EditReceipt:
    """A signed receipt binding one event to its content and parents."""

    event: ReceiptEvent
    receipt_sha256: str
    signature: str

    def as_dict(self) -> dict[str, object]:
        return {
            "event": self.event.as_dict(),
            "receipt_sha256": self.receipt_sha256,
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class CounterfactualCertificate:
    """Bounded robustness certificate over a declared edit family.

    The certificate states: given a source receipt and a preregistered set of
    allowed edit classes with severity ranges, which combinations are expected
    to retain detection at the configured threshold.  It is computed from
    empirical survival data or a preregistered model, not assumed.

    Scientific status: prototype.  A real certificate requires calibrated
    detector evidence and a frozen transformation registry.
    """

    source_receipt_id: str
    detector_config_hash: str
    threshold: float
    edit_family: tuple[str, ...]
    survival_estimate: float
    survival_lower_bound: float
    survival_upper_bound: float
    confidence_level: float
    sample_size: int
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _receipt_hash(event: ReceiptEvent) -> str:
    return _content_hash(_canonical_json(event.as_dict()))


def _sign(receipt_hash: str, signing_key: str) -> str:
    """HMAC-SHA256 signature for development; not a production KMS."""

    return hmac.new(
        signing_key.encode("utf-8"), receipt_hash.encode("utf-8"), hashlib.sha256
    ).hexdigest()


class EditReceiptDAG:
    """Content-addressed directed acyclic graph of lifecycle receipts.

    The DAG enforces:
    - every event references existing parents;
    - no cycles (a parent must precede its child in insertion order);
    - content hashes are immutable; and
    - every receipt is signed by its actor.
    """

    def __init__(self, *, signing_key: str = "development-key-do-not-use-in-production") -> None:
        if not signing_key:
            raise EditReceiptError("A signing key is required for receipt integrity.")
        self._signing_key = signing_key
        self._receipts: dict[str, EditReceipt] = {}
        self._insertion_order: list[str] = []

    def append(self, event: ReceiptEvent) -> EditReceipt:
        if event.event_id in self._receipts:
            raise EditReceiptError(f"Duplicate event_id: {event.event_id}")
        for parent_id in event.parent_event_ids:
            if parent_id not in self._receipts:
                raise EditReceiptError(f"Unknown parent event: {parent_id}")
            parent_index = self._insertion_order.index(parent_id)
            current_index = len(self._insertion_order)
            if parent_index >= current_index:
                raise EditReceiptError("Receipt DAG violation: parent must precede child")
        receipt_hash = _receipt_hash(event)
        signature = _sign(receipt_hash, self._signing_key)
        receipt = EditReceipt(
            event=event,
            receipt_sha256=receipt_hash,
            signature=signature,
        )
        self._receipts[event.event_id] = receipt
        self._insertion_order.append(event.event_id)
        return receipt

    def get(self, event_id: str) -> EditReceipt:
        try:
            return self._receipts[event_id]
        except KeyError as exc:
            raise EditReceiptError(f"Unknown event: {event_id}") from exc

    def parents(self, event_id: str) -> tuple[EditReceipt, ...]:
        receipt = self.get(event_id)
        return tuple(self._receipts[pid] for pid in receipt.event.parent_event_ids)

    def children(self, event_id: str) -> tuple[EditReceipt, ...]:
        return tuple(
            receipt
            for receipt in self._receipts.values()
            if event_id in receipt.event.parent_event_ids
        )

    def chain(self, event_id: str) -> tuple[EditReceipt, ...]:
        """Return the ordered ancestry from root to the named event."""

        chain: list[EditReceipt] = []
        visited: set[str] = set()
        self._collect_chain(event_id, chain, visited)
        return tuple(chain)

    def _collect_chain(
        self, event_id: str, chain: list[EditReceipt], visited: set[str]
    ) -> None:
        if event_id in visited:
            raise EditReceiptError(f"Cycle detected at {event_id}")
        visited.add(event_id)
        receipt = self.get(event_id)
        for parent_id in receipt.event.parent_event_ids:
            self._collect_chain(parent_id, chain, visited)
        chain.append(receipt)

    def verify(self, receipt: EditReceipt) -> bool:
        expected_hash = _receipt_hash(receipt.event)
        if receipt.receipt_sha256 != expected_hash:
            return False
        expected_sig = _sign(expected_hash, self._signing_key)
        return hmac.compare_digest(receipt.signature, expected_sig)

    def all_receipts(self) -> tuple[EditReceipt, ...]:
        return tuple(self._receipts[eid] for eid in self._insertion_order)

    def root_receipts(self) -> tuple[EditReceipt, ...]:
        return tuple(
            receipt
            for receipt in self._receipts.values()
            if not receipt.event.parent_event_ids
        )

    def leaf_receipts(self) -> tuple[EditReceipt, ...]:
        parent_ids = {
            pid
            for receipt in self._receipts.values()
            for pid in receipt.event.parent_event_ids
        }
        return tuple(
            receipt
            for eid, receipt in self._receipts.items()
            if eid not in parent_ids
        )


class EditReceiptStore:
    """Application service for building and querying receipt DAGs."""

    def __init__(self, dag: EditReceiptDAG | None = None) -> None:
        self._dag = dag or EditReceiptDAG()

    @property
    def dag(self) -> EditReceiptDAG:
        return self._dag

    def record_generation(
        self,
        *,
        event_id: str,
        content: str,
        actor: str,
        timestamp: str,
        metadata: dict[str, object] | None = None,
    ) -> EditReceipt:
        event = ReceiptEvent(
            event_id=event_id,
            kind=ReceiptEventKind.GENERATION,
            content_sha256=_content_hash(content),
            parent_event_ids=(),
            actor=actor,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        return self._dag.append(event)

    def record_edit(
        self,
        *,
        event_id: str,
        parent_event_id: str,
        original_content: str,
        edited_content: str,
        actor: str,
        timestamp: str,
        edit_class: str,
        severity: float,
        metadata: dict[str, object] | None = None,
    ) -> EditReceipt:
        if not 0.0 <= severity <= 1.0:
            raise EditReceiptError("severity must be in [0, 1]")
        merged_metadata = {"edit_class": edit_class, "severity": severity}
        if metadata:
            merged_metadata.update(metadata)
        event = ReceiptEvent(
            event_id=event_id,
            kind=ReceiptEventKind.EDIT,
            content_sha256=_content_hash(edited_content),
            parent_event_ids=(parent_event_id,),
            actor=actor,
            timestamp=timestamp,
            metadata=merged_metadata,
        )
        return self._dag.append(event)

    def record_detection(
        self,
        *,
        event_id: str,
        parent_event_id: str,
        detector_config_hash: str,
        decision: str,
        score: float | None,
        actor: str,
        timestamp: str,
    ) -> EditReceipt:
        event = ReceiptEvent(
            event_id=event_id,
            kind=ReceiptEventKind.DETECTION,
            content_sha256=detector_config_hash,
            parent_event_ids=(parent_event_id,),
            actor=actor,
            timestamp=timestamp,
            metadata={"decision": decision, "score": score},
        )
        return self._dag.append(event)

    def certificate(
        self,
        *,
        source_receipt_id: str,
        detector_config_hash: str,
        threshold: float,
        edit_family: Sequence[str],
        survival_estimate: float,
        survival_lower: float,
        survival_upper: float,
        confidence: float = 0.95,
        sample_size: int = 0,
        caveats: Sequence[str] = (),
    ) -> CounterfactualCertificate:
        if not 0.0 <= survival_estimate <= 1.0:
            raise EditReceiptError("survival_estimate must be in [0, 1]")
        if not (0.0 <= survival_lower <= survival_upper <= 1.0):
            raise EditReceiptError("survival bounds must satisfy 0 <= lower <= upper <= 1")
        if not 0.0 < confidence < 1.0:
            raise EditReceiptError("confidence must be in (0, 1)")
        if not edit_family:
            raise EditReceiptError("edit_family must declare at least one edit class")
        source = self._dag.get(source_receipt_id)
        return CounterfactualCertificate(
            source_receipt_id=source.event.event_id,
            detector_config_hash=detector_config_hash,
            threshold=threshold,
            edit_family=tuple(edit_family),
            survival_estimate=survival_estimate,
            survival_lower_bound=survival_lower,
            survival_upper_bound=survival_upper,
            confidence_level=confidence,
            sample_size=sample_size,
            caveats=(
                *caveats,
                "Certificate covers only the declared edit family; it is not universal robustness.",
                "Prototype: scientific certificates require T06 calibrated detectors.",
            ),
        )


@dataclass(frozen=True, slots=True)
class SurvivalPoint:
    """One point on an empirical survival surface."""

    edit_class: str
    severity: float
    survived: bool
    pre_score: float | None
    post_score: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SurvivalSurface:
    """Empirical survival surface across edit classes and severities.

    Scientific status: prototype.  This is a simple aggregate of survival
    points from a deterministic sweep, not a fitted model.  A real survival
    surface requires T06 calibrated detectors, a preregistered transformation
    registry, and a proper statistical model of the survival function.
    """

    edit_family: tuple[str, ...]
    severities: tuple[float, ...]
    points: tuple[SurvivalPoint, ...]
    survival_rate_by_class: dict[str, float]
    mean_survival_severity_by_class: dict[str, float]
    sample_size: int
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "edit_family": list(self.edit_family),
            "severities": list(self.severities),
            "points": [p.as_dict() for p in self.points],
            "survival_rate_by_class": dict(self.survival_rate_by_class),
            "mean_survival_severity_by_class": dict(self.mean_survival_severity_by_class),
            "sample_size": self.sample_size,
            "caveats": list(self.caveats),
        }


def compute_survival_surface(
    attack_results: Sequence[dict[str, object]],
    *,
    edit_family: Sequence[str],
) -> SurvivalSurface:
    """Aggregate attack sweep results into an empirical survival surface.

    ``attack_results`` is a sequence of dicts each containing:
    - ``attack_id`` (str): the edit class
    - ``severity`` (float): the severity level
    - ``survived`` (bool): whether the watermark survived
    - ``pre_attack`` / ``post_attack`` (dict with ``score`` key)

    This is a simple aggregation, not a statistical model.  It computes
    survival rates and mean survival severity per edit class.
    """

    if not attack_results:
        raise EditReceiptError("attack_results must not be empty")
    if not edit_family:
        raise EditReceiptError("edit_family must declare at least one edit class")

    family_set = set(edit_family)
    points: list[SurvivalPoint] = []
    severities_seen: set[float] = set()

    for result in attack_results:
        attack_id = str(result.get("attack_id", ""))
        if attack_id not in family_set:
            continue
        severity_raw = result.get("severity", 0.0)
        severity = float(severity_raw) if isinstance(severity_raw, (int, float)) else 0.0
        survived = bool(result.get("survived", False))
        pre = result.get("pre_attack", {})
        post = result.get("post_attack", {})
        pre_score = pre.get("score") if isinstance(pre, dict) else None
        post_score = post.get("score") if isinstance(post, dict) else None
        points.append(
            SurvivalPoint(
                edit_class=attack_id,
                severity=severity,
                survived=survived,
                pre_score=float(pre_score) if pre_score is not None else None,
                post_score=float(post_score) if post_score is not None else None,
            )
        )
        severities_seen.add(severity)

    if not points:
        raise EditReceiptError("no attack results matched the declared edit family")

    survival_by_class: dict[str, list[bool]] = {}
    for point in points:
        survival_by_class.setdefault(point.edit_class, []).append(point.survived)

    survival_rate: dict[str, float] = {}
    mean_survival_severity: dict[str, float] = {}
    for cls, outcomes in survival_by_class.items():
        survival_rate[cls] = sum(1 for s in outcomes if s) / len(outcomes)
        cls_points = [p for p in points if p.edit_class == cls and p.survived]
        if cls_points:
            mean_survival_severity[cls] = sum(p.severity for p in cls_points) / len(cls_points)
        else:
            mean_survival_severity[cls] = 0.0

    return SurvivalSurface(
        edit_family=tuple(edit_family),
        severities=tuple(sorted(severities_seen)),
        points=tuple(points),
        survival_rate_by_class=survival_rate,
        mean_survival_severity_by_class=mean_survival_severity,
        sample_size=len(points),
        caveats=(
            "Empirical survival surface from deterministic sweep; not a fitted model.",
            "Survival rates are simple fractions, not calibrated probabilities.",
            "Scientific certificates require T06 calibrated detectors and a preregistered transformation registry.",
        ),
    )


__all__ = [
    "CounterfactualCertificate",
    "EditReceipt",
    "EditReceiptDAG",
    "EditReceiptError",
    "EditReceiptStore",
    "ReceiptEvent",
    "ReceiptEventKind",
    "SurvivalPoint",
    "SurvivalSurface",
    "compute_survival_surface",
]