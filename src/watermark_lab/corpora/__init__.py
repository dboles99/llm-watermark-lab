"""Corpora for pipeline validation and real-corpus ingestion.

Synthetic corpora are deterministically generated, explicitly labeled as
non-scientific, and let the full detect/calibrate/attack/localize pipeline
run end-to-end before real licensed corpora are available.  They must never
support a scientific claim or a frozen calibration manifest.

The Taiwan public adapter ingests locally prepared government/open-data text
with recorded license provenance, cluster metadata, deterministic partitions,
and zh-CN contamination screening.  Its output is real public-domain data but
still requires native Taiwan review before use in a frozen calibration.
"""

from watermark_lab.corpora.synthetic import (
    CorpusDocument,
    CorpusManifest,
    SyntheticCorpusError,
    SyntheticCorpusGenerator,
    generate_en_corpus,
    generate_zh_hant_tw_corpus,
)
from watermark_lab.corpora.taiwan_public import (
    PARTITION_PROPORTIONS,
    PublicSourceRecord,
    RejectedDocument,
    TaiwanCorpusError,
    TaiwanPublicCorpusReport,
    ingest_taiwan_public_corpus,
)

__all__ = [
    "PARTITION_PROPORTIONS",
    "CorpusDocument",
    "CorpusManifest",
    "PublicSourceRecord",
    "RejectedDocument",
    "SyntheticCorpusError",
    "SyntheticCorpusGenerator",
    "TaiwanCorpusError",
    "TaiwanPublicCorpusReport",
    "generate_en_corpus",
    "generate_zh_hant_tw_corpus",
    "ingest_taiwan_public_corpus",
]