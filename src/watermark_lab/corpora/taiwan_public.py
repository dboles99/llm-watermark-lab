"""Offline adapter for Taiwan public/government open-data corpora.

Ingests text from a local directory that the operator populated from
public Taiwan sources (e.g. data.gov.tw open data, government gazettes,
open-licensed news archives).  The adapter never performs network access:
every source must be declared in a local ``source-manifest.json`` with a
recorded license before any text is admitted.

Fail-closed rules:

- a missing or incomplete source manifest aborts the whole ingest;
- every source must declare an explicit open license or government open-data
  terms URL; undocumented sources are rejected;
- every document is screened for Simplified-Chinese (zh-CN) vocabulary
  contamination; contaminated documents are rejected and recorded;
- the locale is normalized to ``zh-Hant-TW`` and the canonical fallback
  order is enforced; ``zh-CN`` sources are structurally rejected;
- partitions are assigned deterministically by cluster hash, never randomly,
  and never split a cluster across partitions.

Scientific status is ``real-public-domain`` with a mandatory
``native_review_status`` field that starts at ``pending``: admission to a
frozen calibration additionally requires native Taiwan review.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from watermark_lab.corpora.synthetic import PARTITION_NAMES, CorpusDocument
from watermark_lab.i18n import canonicalize_locale

PARTITION_PROPORTIONS: tuple[tuple[str, float], ...] = (
    ("development", 0.30),
    ("calibration", 0.30),
    ("attack-tuning", 0.15),
    ("final-locked", 0.25),
)

_ACCEPTED_LICENSE_KINDS = frozenset({
    "spdx-open",
    "government-open-data",
    "public-domain",
})

_ZH_CN_CONTAMINATION_TERMS: tuple[str, ...] = (
    # Taiwan vs. zh-CN lexical markers (subset; extended via source manifest).
    "软件", "信息", "网络", "视频", "计算机",
    "数据", "程序", "服务器", "鼠标", "硬盘",
    "激光", "打印机", "字节", "端口", "带宽",
)

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")


class TaiwanCorpusError(ValueError):
    """Raised when Taiwan public-corpus ingestion fails closed."""


@dataclass(frozen=True, slots=True)
class PublicSourceRecord:
    """One declared public data source with its license provenance."""

    source_id: str
    title: str
    url: str
    license_kind: Literal["spdx-open", "government-open-data", "public-domain"]
    license_reference: str
    retrieved_at: str
    domain: str = "general"
    author_group: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: object) -> PublicSourceRecord:
        if not isinstance(payload, dict):
            raise TaiwanCorpusError("source manifest entries must be objects")
        required = ("source_id", "title", "url", "license_kind", "license_reference", "retrieved_at")
        for name in required:
            if not str(payload.get(name, "")).strip():
                raise TaiwanCorpusError(f"source manifest entry missing {name!r}")
        license_kind = str(payload["license_kind"])
        if license_kind not in _ACCEPTED_LICENSE_KINDS:
            raise TaiwanCorpusError(
                f"source {payload['source_id']!r} license_kind {license_kind!r} is not an "
                "accepted open license; undocumented sources are rejected"
            )
        if not _SOURCE_ID_RE.fullmatch(str(payload["source_id"])):
            raise TaiwanCorpusError(
                f"source_id {payload['source_id']!r} must match {_SOURCE_ID_RE.pattern}"
            )
        return PublicSourceRecord(
            source_id=str(payload["source_id"]),
            title=str(payload["title"]),
            url=str(payload["url"]),
            license_kind=license_kind,  # type: ignore[arg-type]
            license_reference=str(payload["license_reference"]),
            retrieved_at=str(payload["retrieved_at"]),
            domain=str(payload.get("domain", "general")),
            author_group=str(payload.get("author_group", "")),
        )


@dataclass(frozen=True, slots=True)
class RejectedDocument:
    """A document rejected during screening, with the recorded reason."""

    path: str
    source_id: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaiwanPublicCorpusReport:
    """Result of one ingest run."""

    corpus_id: str
    locale: str
    scientific_status: Literal["real-public-domain"]
    native_review_status: Literal["pending"]
    sources: tuple[PublicSourceRecord, ...]
    documents: tuple[CorpusDocument, ...]
    rejected: tuple[RejectedDocument, ...]
    cluster_count: int
    partition_counts: dict[str, int]
    manifest_sha256: str
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "locale": self.locale,
            "scientific_status": self.scientific_status,
            "native_review_status": self.native_review_status,
            "sources": [s.as_dict() for s in self.sources],
            "documents": [d.as_dict() for d in self.documents],
            "rejected": [r.as_dict() for r in self.rejected],
            "cluster_count": self.cluster_count,
            "partition_counts": dict(self.partition_counts),
            "manifest_sha256": self.manifest_sha256,
            "caveats": list(self.caveats),
        }


def _cluster_digest(cluster_key: str) -> str:
    return hashlib.sha256(f"taiwan-public:{cluster_key}".encode()).hexdigest()


def _partition_for_digest(digest_hex: str) -> str:
    """Deterministically map a cluster digest to a frozen partition."""

    value = int(digest_hex[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, proportion in PARTITION_PROPORTIONS:
        cumulative += proportion
        if value <= cumulative:
            return name
    return PARTITION_PROPORTIONS[-1][0]


def _screen_zh_cn_contamination(
    text: str,
    extra_terms: Sequence[str] = (),
) -> str | None:
    """Return a contamination reason if zh-CN vocabulary markers appear."""

    for term in (*_ZH_CN_CONTAMINATION_TERMS, *extra_terms):
        if term in text:
            return f"zh-CN contamination marker {term!r} present"
    return None


def _read_source_manifest(root: Path) -> tuple[dict[str, PublicSourceRecord], list[str]]:
    manifest_path = root / "source-manifest.json"
    if not manifest_path.is_file():
        raise TaiwanCorpusError(
            f"source manifest not found: {manifest_path}. Every public source must be "
            "declared with a license before ingestion; network fetch is prohibited."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaiwanCorpusError(f"source manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise TaiwanCorpusError("source manifest must contain a 'sources' array")

    extra_terms: list[str] = []
    declared_terms = payload.get("additional_zh_cn_markers", [])
    if declared_terms is not None:
        if not isinstance(declared_terms, list) or not all(
            isinstance(t, str) and t.strip() for t in declared_terms
        ):
            raise TaiwanCorpusError("additional_zh_cn_markers must be an array of nonempty strings")
        extra_terms = [str(t) for t in declared_terms]

    records: dict[str, PublicSourceRecord] = {}
    for entry in payload["sources"]:
        record = PublicSourceRecord.from_dict(entry)
        if record.source_id in records:
            raise TaiwanCorpusError(f"duplicate source_id {record.source_id!r}")
        records[record.source_id] = record
    if not records:
        raise TaiwanCorpusError("source manifest declares no sources")
    return records, extra_terms


def ingest_taiwan_public_corpus(
    root: str | Path,
    *,
    locale: str = "zh-Hant-TW",
) -> TaiwanPublicCorpusReport:
    """Ingest a local Taiwan public-data corpus directory.

    Expected layout::

        <root>/source-manifest.json      # declared sources with licenses
        <root>/sources/<source_id>/*.txt # one document per .txt file

    Documents are clustered by ``source_id`` + ``author_group`` when present,
    so repeated material from one origin stays one independent unit.
    """

    from watermark_lab.i18n.policy import LocalePolicyError

    try:
        canonical = canonicalize_locale(locale)
    except LocalePolicyError as exc:
        raise TaiwanCorpusError(str(exc)) from exc
    if canonical != "zh-Hant-TW":
        raise TaiwanCorpusError(
            f"Taiwan public corpus locale must normalize to zh-Hant-TW; got {canonical!r}"
        )

    root_path = Path(root)
    if not root_path.is_dir():
        raise TaiwanCorpusError(f"corpus directory not found: {root_path}")

    sources, extra_terms = _read_source_manifest(root_path)
    sources_root = root_path / "sources"
    if not sources_root.is_dir():
        raise TaiwanCorpusError(f"sources directory not found: {sources_root}")

    documents: list[CorpusDocument] = []
    rejected: list[RejectedDocument] = []
    cluster_keys: dict[str, list[CorpusDocument]] = {}

    for source_dir in sorted(p for p in sources_root.iterdir() if p.is_dir()):
        source_id = source_dir.name
        record = sources.get(source_id)
        if record is None:
            raise TaiwanCorpusError(
                f"source directory {source_id!r} has no declaration in source-manifest.json"
            )
        txt_files = sorted(source_dir.glob("*.txt"))
        if not txt_files:
            rejected.append(
                RejectedDocument(
                    path=source_dir.relative_to(root_path).as_posix(),
                    source_id=source_id,
                    reason="no .txt documents in source directory",
                )
            )
            continue

        cluster_key = source_id if not record.author_group else f"{source_id}:{record.author_group}"
        for txt in txt_files:
            relative = txt.relative_to(root_path).as_posix()
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as exc:
                rejected.append(
                    RejectedDocument(path=relative, source_id=source_id, reason=f"unreadable: {exc}")
                )
                continue
            if not text:
                rejected.append(
                    RejectedDocument(path=relative, source_id=source_id, reason="empty document")
                )
                continue
            contamination = _screen_zh_cn_contamination(text, extra_terms)
            if contamination is not None:
                rejected.append(
                    RejectedDocument(path=relative, source_id=source_id, reason=contamination)
                )
                continue
            cluster_id = f"twpub-{_cluster_digest(cluster_key)[:12]}"
            document = CorpusDocument(
                document_id=f"{cluster_id}-{txt.stem}",
                cluster_id=cluster_id,
                partition="pending",
                locale="zh-Hant-TW",
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                token_count_estimate=len(text),
                domain=record.domain,
            )
            documents.append(document)
            cluster_keys.setdefault(cluster_id, []).append(document)

    if not documents:
        raise TaiwanCorpusError(
            f"ingest admitted zero documents ({len(rejected)} rejected); refusing an empty corpus"
        )

    assigned: list[CorpusDocument] = []
    partition_counts = {name: 0 for name in PARTITION_NAMES}
    for cluster_id, members in sorted(cluster_keys.items()):
        digest_hex = _cluster_digest(cluster_id)
        partition = _partition_for_digest(digest_hex)
        partition_counts[partition] += 1
        for member in members:
            assigned.append(
                CorpusDocument(
                    document_id=member.document_id,
                    cluster_id=member.cluster_id,
                    partition=partition,
                    locale=member.locale,
                    text=member.text,
                    text_sha256=member.text_sha256,
                    token_count_estimate=member.token_count_estimate,
                    domain=member.domain,
                )
            )

    manifest_payload = {
        "locale": "zh-Hant-TW",
        "scientific_status": "real-public-domain",
        "native_review_status": "pending",
        "sources": [s.as_dict() for s in sorted(sources.values(), key=lambda s: s.source_id)],
        "documents": [d.as_dict() for d in sorted(assigned, key=lambda d: d.document_id)],
        "partition_counts": partition_counts,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    corpus_id = f"twpub-{manifest_sha256[:16]}"

    caveats = [
        "Public-domain/government open data; license provenance recorded per source.",
        (
            "native_review_status=pending: native Taiwan review is required before any frozen "
            "calibration manifest may reference this corpus."
        ),
        "zh-CN screening covers declared lexical markers; it is not a full orthographic audit.",
    ]
    if rejected:
        caveats.append(
            f"{len(rejected)} documents rejected during screening; see 'rejected' records."
        )

    return TaiwanPublicCorpusReport(
        corpus_id=corpus_id,
        locale="zh-Hant-TW",
        scientific_status="real-public-domain",
        native_review_status="pending",
        sources=tuple(sorted(sources.values(), key=lambda s: s.source_id)),
        documents=tuple(assigned),
        rejected=tuple(rejected),
        cluster_count=len(cluster_keys),
        partition_counts=partition_counts,
        manifest_sha256=manifest_sha256,
        caveats=tuple(caveats),
    )


__all__ = [
    "PARTITION_PROPORTIONS",
    "PublicSourceRecord",
    "RejectedDocument",
    "TaiwanCorpusError",
    "TaiwanPublicCorpusReport",
    "ingest_taiwan_public_corpus",
]