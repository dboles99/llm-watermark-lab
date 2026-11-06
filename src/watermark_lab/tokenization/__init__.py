"""Tokenizer-faithful, offline-first encoding contracts."""

from watermark_lab.tokenization.boundary import (
    FaithfulTokenizer,
    TokenizerBackend,
    tokenizer_fingerprint_sha256,
)
from watermark_lab.tokenization.contracts import (
    AssetFileFingerprint,
    NormalizationPolicy,
    SpecialTokenPolicy,
    TokenizationError,
    TokenizationResult,
    TokenizerAssetManifest,
    TokenizerAssetUnavailableError,
    TokenizerConfigurationError,
    TokenizerEncodingError,
    TokenizerProvenance,
    TokenizerSpec,
    UnicodeNormalization,
)
from watermark_lab.tokenization.huggingface import (
    fingerprint_local_assets,
    load_local_huggingface_tokenizer,
)
from watermark_lab.tokenization.synthetic import (
    SYNTHETIC_NUMERIC_REVISION,
    SYNTHETIC_NUMERIC_TOKENIZER_ID,
    SyntheticNumericBackend,
    synthetic_numeric_tokenizer,
)

__all__ = [
    "SYNTHETIC_NUMERIC_REVISION",
    "SYNTHETIC_NUMERIC_TOKENIZER_ID",
    "AssetFileFingerprint",
    "FaithfulTokenizer",
    "NormalizationPolicy",
    "SpecialTokenPolicy",
    "SyntheticNumericBackend",
    "TokenizationError",
    "TokenizationResult",
    "TokenizerAssetManifest",
    "TokenizerAssetUnavailableError",
    "TokenizerBackend",
    "TokenizerConfigurationError",
    "TokenizerEncodingError",
    "TokenizerProvenance",
    "TokenizerSpec",
    "UnicodeNormalization",
    "fingerprint_local_assets",
    "load_local_huggingface_tokenizer",
    "synthetic_numeric_tokenizer",
    "tokenizer_fingerprint_sha256",
]
