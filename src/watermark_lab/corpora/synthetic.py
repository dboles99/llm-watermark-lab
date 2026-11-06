"""Deterministic synthetic corpus generator for pipeline validation.

The generated text is simple template-based prose with controlled vocabulary,
cluster metadata, and frozen partitions.  It is sufficient for exercising
the detection, calibration, attack, and localization pipeline but carries
an explicit ``scientific_status: non-scientific-synthetic`` label.

No generated corpus is ever eligible to support a frozen calibration manifest
or a scientific FPR claim.  The ``CorpusManifest.scientific_status`` field
exists precisely so downstream gates can reject it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

_EN_TEMPLATES = (
    "The {adj} {noun} {verb} {adv} near the {place}.",
    "A {adj} {noun} {verb} {adv} beside the {place}.",
    "Every {adj} {noun} must {verb} {adv} before the {place} closes.",
    "Few {adj} {noun}s {verb} {adv} around the {place}.",
    "The {adj} {noun} cannot {verb} {adv} without the {place}.",
)

_EN_WORDS = {
    "adj": ("calm", "bright", "steady", "ancient", "modern", "quiet", "rapid"),
    "noun": ("river", "forest", "scholar", "vessel", "engine", "archive", "signal"),
    "verb": ("flows", "grows", "learns", "travels", "responds", "adapts", "converges"),
    "adv": ("smoothly", "quietly", "swiftly", "carefully", "patiently", "boldly"),
    "place": ("harbor", "library", "laboratory", "station", "institute", "summit"),
}

_ZH_HANT_TW_TEMPLATES = (
    "{adj}的{noun}{verb}{adv}，位於{place}附近。",
    "在{place}旁，{adj}的{noun}{verb}{adv}。",
    "每一{measure}{adj}的{noun}都必須在{place}關閉前{verb}{adv}。",
    "很少有{adj}的{noun}會在{place}周圍{verb}{adv}。",
    "{adj}的{noun}無法在沒有{place}的情況下{verb}{adv}。",
)

_ZH_HANT_TW_WORDS = {
    "adj": ("平靜", "明亮", "穩定", "古老", "現代", "安靜", "迅速"),
    "noun": ("河流", "森林", "學者", "船隻", "引擎", "檔案", "訊號"),
    "verb": ("流動", "生長", "學習", "航行", "回應", "適應", "匯聚"),
    "adv": ("平穩地", "安靜地", "迅速地", "謹慎地", "耐心地", "大膽地"),
    "place": ("港口", "圖書館", "實驗室", "車站", "研究院", "山頂"),
    "measure": ("個", "位", "座", "條"),
}

PARTITION_NAMES = ("development", "calibration", "attack-tuning", "final-locked")


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One deterministically generated document with cluster metadata."""

    document_id: str
    cluster_id: str
    partition: str
    locale: str
    text: str
    text_sha256: str
    token_count_estimate: int
    domain: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Frozen manifest for one synthetic corpus generation run."""

    corpus_id: str
    locale: str
    scientific_status: Literal["non-scientific-synthetic"]
    generator_version: str
    seed: int
    document_count: int
    cluster_count: int
    partitions: tuple[str, ...]
    documents: tuple[CorpusDocument, ...]
    manifest_sha256: str
    generation_note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "locale": self.locale,
            "scientific_status": self.scientific_status,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "document_count": self.document_count,
            "cluster_count": self.cluster_count,
            "partitions": list(self.partitions),
            "documents": [doc.as_dict() for doc in self.documents],
            "manifest_sha256": self.manifest_sha256,
            "generation_note": self.generation_note,
        }


class SyntheticCorpusError(ValueError):
    """Raised when synthetic corpus generation parameters are invalid."""


class SyntheticCorpusGenerator:
    """Deterministic template-based corpus generator.

    Each cluster produces multiple documents so that cluster-aware inference
    has realistic dependence structure.  Partitions are assigned by cluster
    (not by document) to respect the independent-unit principle.
    """

    VERSION = "synthetic-v1"

    def __init__(self, *, seed: int = 20260815) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SyntheticCorpusError("seed must be an integer")
        self._seed = seed

    def generate(
        self,
        locale: Literal["en", "zh-Hant-TW"],
        *,
        clusters: int = 20,
        documents_per_cluster: int = 5,
        sentences_per_document: int = 8,
        domains: Sequence[str] = ("news", "editorial", "technical"),
    ) -> CorpusManifest:
        if locale not in ("en", "zh-Hant-TW"):
            raise SyntheticCorpusError(
                "locale must be exactly 'en' or 'zh-Hant-TW'; zh-CN is not permitted"
            )
        if clusters < 1:
            raise SyntheticCorpusError("clusters must be at least 1")
        if documents_per_cluster < 1:
            raise SyntheticCorpusError("documents_per_cluster must be at least 1")
        if sentences_per_document < 1:
            raise SyntheticCorpusError("sentences_per_document must be at least 1")
        if not domains:
            raise SyntheticCorpusError("at least one domain is required")

        rng = random.Random(f"{self._seed}:{locale}")
        templates, word_sets = self._locale_resources(locale)
        documents: list[CorpusDocument] = []
        partition_assignment = self._assign_partitions(clusters, rng)

        for cluster_index in range(clusters):
            cluster_id = f"cluster-{cluster_index:04d}"
            partition = partition_assignment[cluster_index]
            domain = domains[cluster_index % len(domains)]
            for doc_index in range(documents_per_cluster):
                doc_id = f"{cluster_id}-doc-{doc_index:02d}"
                text = self._generate_document(
                    rng, templates, word_sets, sentences_per_document
                )
                token_estimate = len(text.split()) if locale == "en" else len(text)
                documents.append(
                    CorpusDocument(
                        document_id=doc_id,
                        cluster_id=cluster_id,
                        partition=partition,
                        locale=locale,
                        text=text,
                        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        token_count_estimate=token_estimate,
                        domain=domain,
                    )
                )

        corpus_id = hashlib.sha256(
            f"{locale}:{self._seed}:{clusters}:{documents_per_cluster}".encode()
        ).hexdigest()[:16]
        generation_note = (
            "Synthetic template-generated text for pipeline validation only. "
            "Not eligible for scientific claims or frozen calibration."
        )
        manifest_payload = {
            "corpus_id": corpus_id,
            "locale": locale,
            "scientific_status": "non-scientific-synthetic",
            "generator_version": self.VERSION,
            "seed": self._seed,
            "document_count": len(documents),
            "cluster_count": clusters,
            "partitions": list(PARTITION_NAMES),
            "documents": [doc.as_dict() for doc in documents],
            "generation_note": generation_note,
        }
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

        return CorpusManifest(
            corpus_id=corpus_id,
            locale=locale,
            scientific_status="non-scientific-synthetic",
            generator_version=self.VERSION,
            seed=self._seed,
            document_count=len(documents),
            cluster_count=clusters,
            partitions=PARTITION_NAMES,
            documents=tuple(documents),
            manifest_sha256=manifest_sha256,
            generation_note=generation_note,
        )

    @staticmethod
    def _locale_resources(
        locale: Literal["en", "zh-Hant-TW"],
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        if locale == "en":
            return _EN_TEMPLATES, _EN_WORDS
        return _ZH_HANT_TW_TEMPLATES, _ZH_HANT_TW_WORDS

    @staticmethod
    def _assign_partitions(
        clusters: int, rng: random.Random
    ) -> list[str]:
        """Assign clusters to partitions with approximate balance."""
        base = clusters // len(PARTITION_NAMES)
        remainder = clusters % len(PARTITION_NAMES)
        pool: list[str] = []
        for i, name in enumerate(PARTITION_NAMES):
            count = base + (1 if i < remainder else 0)
            pool.extend([name] * count)
        rng.shuffle(pool)
        return list(pool)

    @staticmethod
    def _generate_document(
        rng: random.Random,
        templates: tuple[str, ...],
        word_sets: dict[str, tuple[str, ...]],
        sentences: int,
    ) -> str:
        parts: list[str] = []
        for _ in range(sentences):
            template = rng.choice(templates)
            replacements: dict[str, str] = {}
            for key, options in word_sets.items():
                replacements[key] = rng.choice(options)
            parts.append(template.format(**replacements))
        return " ".join(parts)


def generate_en_corpus(
    *, seed: int = 20260815, clusters: int = 20, documents_per_cluster: int = 5
) -> CorpusManifest:
    """Generate a deterministic synthetic English corpus for pipeline validation."""

    return SyntheticCorpusGenerator(seed=seed).generate(
        "en", clusters=clusters, documents_per_cluster=documents_per_cluster
    )


def generate_zh_hant_tw_corpus(
    *, seed: int = 20260816, clusters: int = 20, documents_per_cluster: int = 5
) -> CorpusManifest:
    """Generate a deterministic synthetic zh-Hant-TW corpus for pipeline validation.

    The text uses Traditional Chinese characters and Taiwan-flavored vocabulary.
    It is explicitly labeled non-scientific and cannot support a frozen
    calibration manifest.
    """

    return SyntheticCorpusGenerator(seed=seed).generate(
        "zh-Hant-TW", clusters=clusters, documents_per_cluster=documents_per_cluster
    )