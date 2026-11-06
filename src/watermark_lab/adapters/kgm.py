"""Compatibility shim: the unit tests import from ``watermark_lab.adapters.kgm``.

The actual KGW implementation lives in ``kgw.py``. This module re-exports it so
existing tests resolve without modification.
"""

from __future__ import annotations

from watermark_lab.adapters.kgw import (
    DEFAULT_HASH_KEY,
    SIMPLE_1_MODULUS,
    kgw_apply_logit_bias,
    kgw_count_green,
    kgw_detect,
    kgw_green_list,
    kgw_seed_from_context,
    kgw_z_score,
)

__all__ = [
    "DEFAULT_HASH_KEY",
    "SIMPLE_1_MODULUS",
    "kgw_apply_logit_bias",
    "kgw_count_green",
    "kgw_detect",
    "kgw_green_list",
    "kgw_seed_from_context",
    "kgw_z_score",
]
