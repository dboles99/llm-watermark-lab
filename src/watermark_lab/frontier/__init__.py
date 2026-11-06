"""Frontier research feature skeletons.

Each module is an independently-gated algorithm skeleton, not a registered
product label.  They share the fail-closed, frozen-dataclass, and
inconclusive-by-default conventions of the rest of the lab.

Features:
- EditReceipt: content-addressed receipt DAG with counterfactual certificates.
- PolyMark: multi-provider/key/epoch demixing in mixed documents.
- RobustFragile: robust-plus-fragile dual-channel semantic integrity.
- TranslinguaTW: signed translation handoff with Taiwan-language invariants.
"""

from watermark_lab.frontier.edit_receipt import (
    CounterfactualCertificate,
    EditReceipt,
    EditReceiptDAG,
    EditReceiptError,
    EditReceiptStore,
    ReceiptEvent,
    ReceiptEventKind,
    SurvivalPoint,
    SurvivalSurface,
    compute_survival_surface,
)
from watermark_lab.frontier.polymark import (
    PolymarkDemixer,
    PolymarkDemixingError,
    PolymarkDemixingResult,
    PolymarkProcess,
    PolymarkSpan,
    bonferroni_threshold,
    scan_aware_p_value,
)
from watermark_lab.frontier.robust_fragile import (
    EditClass,
    EditOntology,
    FragileChannelState,
    InvariantKind,
    RobustChannelState,
    RobustFragileError,
    RobustFragileResult,
    RobustFragileWatermarker,
    SemanticInvariant,
    is_benign_edit,
    is_material_edit,
)
from watermark_lab.frontier.translingua_tw import (
    FactualInvariant,
    TaiwanTerminologyInvariant,
    TranslationHandoff,
    TranslinguaTWError,
    TranslinguaTWHandoffService,
    VerifiedHandoff,
    check_factual_invariants,
    create_factual_handoff,
)

__all__ = [
    "CounterfactualCertificate",
    "EditClass",
    "EditOntology",
    "EditReceipt",
    "EditReceiptDAG",
    "EditReceiptError",
    "EditReceiptStore",
    "FactualInvariant",
    "FragileChannelState",
    "InvariantKind",
    "PolymarkDemixer",
    "PolymarkDemixingError",
    "PolymarkDemixingResult",
    "PolymarkProcess",
    "PolymarkSpan",
    "ReceiptEvent",
    "ReceiptEventKind",
    "RobustChannelState",
    "RobustFragileError",
    "RobustFragileResult",
    "RobustFragileWatermarker",
    "SemanticInvariant",
    "SurvivalPoint",
    "SurvivalSurface",
    "TaiwanTerminologyInvariant",
    "TranslationHandoff",
    "TranslinguaTWError",
    "TranslinguaTWHandoffService",
    "VerifiedHandoff",
    "bonferroni_threshold",
    "check_factual_invariants",
    "compute_survival_surface",
    "create_factual_handoff",
    "is_benign_edit",
    "is_material_edit",
    "scan_aware_p_value",
]