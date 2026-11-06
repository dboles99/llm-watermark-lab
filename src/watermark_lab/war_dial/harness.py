"""War-dial harness: enumerate watermark parameter coordinates, run tests, save results.

The harness is intentionally small enough to run inside an agent loop: each
"dial" is one parameter combination, and the runner emits a JSONL record for
every dial so the loop can be paused, inspected, and resumed.
"""

from __future__ import annotations

import json
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watermark_lab.attacks.transformations import AttackConfig, TransformationRegistry
from watermark_lab.core.contracts import DetectionConfig
from watermark_lab.frontier.edit_receipt import compute_survival_surface
from watermark_lab.run_engine.attack import RobustnessService
from watermark_lab.run_engine.create import CreationService
from watermark_lab.statistics.multiplicity import SweepFamily, declare_sweep_family
from watermark_lab.stego import StegoDecoder, StegoEncoder
from watermark_lab.war_dial.space import WarDialCoordinate


@dataclass(frozen=True)
class DialResult:
    """Result of one war-dial coordinate."""

    coordinate: dict[str, Any]
    dial_id: str
    iteration: int
    coding_passed: bool
    coding_stdout: str
    coding_stderr: str
    kgw_created: bool
    kgw_decision: str | None
    kgw_token_count: int | None
    kgw_parent_run: dict[str, Any] | None
    attack_results: list[dict[str, Any]]
    survival_surface: dict[str, Any] | None
    stego_encoded: bool
    stego_bits_embedded: int
    stego_decode_status: str
    stego_verify_status: str
    perturbation_applied: bool
    perturbed_decode_status: str | None
    perturbed_text_hash: str | None
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dial_id": self.dial_id,
            "iteration": self.iteration,
            "coordinate": self.coordinate,
            "coding_passed": self.coding_passed,
            "coding_stdout": self.coding_stdout,
            "coding_stderr": self.coding_stderr,
            "kgw_created": self.kgw_created,
            "kgw_decision": self.kgw_decision,
            "kgw_token_count": self.kgw_token_count,
            "kgw_parent_run": self.kgw_parent_run,
            "attack_results": self.attack_results,
            "survival_surface": self.survival_surface,
            "stego_encoded": self.stego_encoded,
            "stego_bits_embedded": self.stego_bits_embedded,
            "stego_decode_status": self.stego_decode_status,
            "stego_verify_status": self.stego_verify_status,
            "perturbation_applied": self.perturbation_applied,
            "perturbed_decode_status": self.perturbed_decode_status,
            "perturbed_text_hash": self.perturbed_text_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fast_coding_test() -> tuple[bool, str, str]:
    """In-process smoke test: stego encode/decode roundtrip.

    This is the default coding test because it is fast enough to run once per
    dial coordinate inside an agent loop.  It exercises the same code paths as
    the stego unit tests without spawning a subprocess.
    """
    try:
        from watermark_lab.stego import StegoDecoder, StegoEncoder

        cover = (
            "The quick brown fox jumps over the lazy dog several times "
            "to provide enough words for embedding a modest hidden payload "
            "into the cover text and then some extra words to guarantee capacity "
            "with a safety margin of additional words beyond the threshold "
            "so that even the compact short envelope fits comfortably inside the text"
        )
        encoder = StegoEncoder(channel_id="whitespace", secret=b"dial")
        decoder = StegoDecoder(channel_id="whitespace", secret=b"dial")
        encoded = encoder.encode(cover, "ok")
        if encoded.status != "ok":
            return False, "", f"encode failed: {encoded.warnings}"
        decoded = decoder.decode(encoded.text)
        if decoded.status != "ok" or decoded.payload != b"ok":
            return False, "", f"decode failed: {decoded.warnings}"
        return True, "stego roundtrip ok", ""
    except Exception as exc:
        return False, "", str(exc)


class WarDialHarness:
    """Enumerate a parameter grid and run coding + encoding tests at each point."""

    def __init__(
        self,
        output_dir: Path,
        coding_test_args: list[str] | None = None,
        coding_test_timeout: float = 120.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coding_test_args = coding_test_args
        self.coding_test_timeout = coding_test_timeout
        self._registry = TransformationRegistry()
        self._family_file = self.output_dir / "sweep_family.json"

    def run_coding_test(self) -> tuple[bool, str, str]:
        """Run the configured coding test and return (passed, stdout, stderr).

        If ``coding_test_args`` is ``None``, use a fast in-process stego
        roundtrip so the war-dial loop stays lightweight enough to run many
        coordinates inside an agent turn.
        """
        if self.coding_test_args is None:
            return _fast_coding_test()
        try:
            proc = subprocess.run(
                ["python", *self.coding_test_args],
                capture_output=True,
                text=True,
                timeout=self.coding_test_timeout,
                check=False,
            )
            return proc.returncode == 0, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return False, "", "coding test timed out"
        except Exception as exc:
            return False, "", str(exc)

    def run_kgw_test(
        self,
        coordinate: WarDialCoordinate,
        run_attack_sweep: bool = False,
    ) -> dict[str, Any]:
        """Run KGW creation + detection for the KGW part of a coordinate.

        When ``run_attack_sweep`` is true, also run a small attack sweep and
        compute an empirical survival surface using the frontier EditReceipt
        machinery.
        """
        if not coordinate.kgw:
            return {
                "created": False,
                "decision": None,
                "token_count": None,
                "parent_run": None,
                "attack_results": [],
                "survival_surface": None,
            }
        try:
            service = CreationService()
            result = service.create(
                prompt=coordinate.kgw.prompt,
                profile_id="mock",
                method_id="kgw",
                key_id=coordinate.kgw.key_id,
                config={"disclosure_recorded": True, **coordinate.kgw.to_config()},
            )
            decision = result["self_verification"]["decision"]
            token_count = result["self_verification"]["token_count"]
            out: dict[str, Any] = {
                "created": True,
                "decision": decision,
                "token_count": token_count,
                "parent_run": result,
                "attack_results": [],
                "survival_surface": None,
            }
            if run_attack_sweep and decision == "verified compatible watermark":
                out["attack_results"] = self._run_attack_sweep(result, coordinate)
                if out["attack_results"]:
                    surface = compute_survival_surface(
                        out["attack_results"],
                        edit_family=["deletion", "insertion", "substitution", "shuffle"],
                    )
                    out["survival_surface"] = surface.as_dict()
            return out
        except Exception as exc:
            return {
                "created": False,
                "decision": f"error: {exc}",
                "token_count": None,
                "parent_run": None,
                "attack_results": [],
                "survival_surface": None,
            }

    def _run_attack_sweep(
        self,
        parent_run: dict[str, Any],
        coordinate: WarDialCoordinate,
    ) -> list[dict[str, Any]]:
        """Run a deterministic attack sweep against a KGW parent run."""
        assert coordinate.kgw is not None
        detector = DetectionConfig(
            method_id="kgw",
            tokenizer_id=coordinate.kgw.tokenizer_id,
            key_id=coordinate.kgw.key_id,
            threshold=coordinate.kgw.detection_threshold,
            options={
                "gamma": coordinate.kgw.gamma,
                "context_width": coordinate.kgw.context_width,
                "seeding_scheme": coordinate.kgw.seeding_scheme,
                "hash_key": coordinate.kgw.hash_key,
            },
        )
        svc = RobustnessService()
        results: list[dict[str, Any]] = []
        for attack_id in ("deletion", "insertion", "substitution", "shuffle"):
            for severity in (0.1, 0.3, 0.5):
                results.extend(
                    svc.sweep(
                        parent_run,
                        attack_id,
                        [severity],
                        detector,
                        seed=f"war-dial-{coordinate.iteration}",
                    )
                )
        return results

    def run_stego_test(self, coordinate: WarDialCoordinate) -> dict[str, Any]:
        """Run stego encode/decode/verify for the stego part of a coordinate."""
        if not coordinate.stego:
            return {
                "encoded": False,
                "bits_embedded": 0,
                "decode_status": "none",
                "verify_status": "none",
                "stego_text": "",
            }
        try:
            encoder = StegoEncoder(
                channel_id=coordinate.stego.channel,
                secret=coordinate.stego.secret.encode("utf-8"),
                key_id=coordinate.stego.key_id,
                vocab_size=coordinate.stego.vocab_size,
                gamma=coordinate.stego.gamma,
                hash_key=coordinate.stego.hash_key,
            )
            decoder = StegoDecoder(
                channel_id=coordinate.stego.channel,
                secret=coordinate.stego.secret.encode("utf-8"),
                key_id=coordinate.stego.key_id,
                vocab_size=coordinate.stego.vocab_size,
                gamma=coordinate.stego.gamma,
                hash_key=coordinate.stego.hash_key,
            )
            encoded = encoder.encode(coordinate.stego.cover, coordinate.stego.payload)
            if encoded.status != "ok":
                return {
                    "encoded": False,
                    "bits_embedded": encoded.bits_embedded,
                    "decode_status": "encode_failed",
                    "verify_status": "encode_failed",
                    "stego_text": encoded.text,
                }
            decoded = decoder.decode(encoded.text)
            verified = decoder.verify(encoded.text)
            return {
                "encoded": True,
                "bits_embedded": encoded.bits_embedded,
                "decode_status": decoded.status,
                "verify_status": verified.status,
                "stego_text": encoded.text,
            }
        except Exception as exc:
            return {
                "encoded": False,
                "bits_embedded": 0,
                "decode_status": f"error: {exc}",
                "verify_status": f"error: {exc}",
                "stego_text": "",
            }

    def _default_perturb(self, text: str, rng: random.Random, rate: float) -> str:
        """Default word-level perturbation: duplicate, drop, or swap words."""
        words = text.split()
        if not words:
            return text
        out: list[str] = []
        for word in words:
            r = rng.random()
            if r < rate * 0.5:
                continue
            if r < rate:
                out.extend([word, word])
            else:
                out.append(word)
        if len(out) > 1 and rng.random() < rate:
            i, j = rng.randrange(len(out)), rng.randrange(len(out))
            out[i], out[j] = out[j], out[i]
        return " ".join(out)

    def run_perturbation_test(
        self,
        coordinate: WarDialCoordinate,
        stego_text: str,
        perturb_fn: Callable[[str, random.Random, float], str] | None = None,
    ) -> dict[str, Any]:
        """Apply the configured perturbation and try to decode the result."""
        if not coordinate.perturbation or not stego_text:
            return {"applied": False, "decode_status": None, "text_hash": None}
        try:
            rng = random.Random(coordinate.perturbation.seed)
            if coordinate.perturbation.attack_id == "builtin-word-jitter":
                fn = perturb_fn or self._default_perturb
                perturbed = fn(stego_text, rng, coordinate.perturbation.severity)
            else:
                cfg = AttackConfig(
                    attack_id=coordinate.perturbation.attack_id,
                    severity=coordinate.perturbation.severity,
                    seed=coordinate.perturbation.seed,
                )
                perturbed = self._registry.transform(stego_text, cfg)["transformed_text"]

            decoder = StegoDecoder(
                channel_id=coordinate.stego.channel if coordinate.stego else "whitespace",
                secret=(coordinate.stego.secret.encode("utf-8") if coordinate.stego else b""),
            )
            result = decoder.decode(perturbed)
            return {
                "applied": True,
                "decode_status": result.status,
                "text_hash": _sha256(perturbed),
            }
        except Exception as exc:
            return {
                "applied": True,
                "decode_status": f"error: {exc}",
                "text_hash": _sha256(stego_text),
            }

    def dial(
        self,
        coordinate: WarDialCoordinate,
        dial_id: str,
        run_attack_sweep: bool = False,
    ) -> DialResult:
        """Run one coordinate and return a frozen result record."""
        coding_passed, coding_stdout, coding_stderr = self.run_coding_test()
        kgw = self.run_kgw_test(coordinate, run_attack_sweep=run_attack_sweep)
        stego = self.run_stego_test(coordinate)
        perturbed = self.run_perturbation_test(coordinate, stego.get("stego_text", ""))

        return DialResult(
            coordinate=coordinate.as_dict(),
            dial_id=dial_id,
            iteration=coordinate.iteration,
            coding_passed=coding_passed,
            coding_stdout=coding_stdout,
            coding_stderr=coding_stderr,
            kgw_created=kgw["created"],
            kgw_decision=kgw["decision"],
            kgw_token_count=kgw["token_count"],
            kgw_parent_run=kgw.get("parent_run"),
            attack_results=kgw.get("attack_results", []),
            survival_surface=kgw.get("survival_surface"),
            stego_encoded=stego["encoded"],
            stego_bits_embedded=stego["bits_embedded"],
            stego_decode_status=stego["decode_status"],
            stego_verify_status=stego["verify_status"],
            perturbation_applied=perturbed["applied"],
            perturbed_decode_status=perturbed["decode_status"],
            perturbed_text_hash=perturbed["text_hash"],
            timestamp=_now_iso(),
        )

    def declare_family(
        self,
        coordinates: list[WarDialCoordinate],
        *,
        family_level: float = 0.05,
        method: str = "bonferroni",
    ) -> SweepFamily:
        """Declare the multiple-comparison family for a grid and persist it.

        A sweep tests one text against every coordinate, so the probability that
        at least one coordinate fires on unwatermarked text grows with the grid.
        The family must be declared from the grid that is actually executed, and
        it must be declared before results are inspected: choosing it afterwards
        reintroduces the error the correction removes.

        ``dimensions`` is derived from the coordinates so a reviewer can check
        that the recorded family size matches the grid that ran.
        """

        dimensions: set[str] = set()
        for coordinate in coordinates:
            if coordinate.kgw is not None:
                dimensions.update(("key_id", "gamma", "delta", "context_width"))
            if coordinate.stego is not None:
                dimensions.add("stego_channel")
            if coordinate.perturbation is not None:
                dimensions.update(("attack_id", "severity"))

        family = declare_sweep_family(
            family_level=family_level,
            family_size=max(1, len(coordinates)),
            method=method,  # type: ignore[arg-type]
            dimensions=tuple(sorted(dimensions)),
        )
        self._family_file.write_text(
            json.dumps(family.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return family

    def load_family(self) -> dict[str, Any] | None:
        """Return the persisted sweep family, or ``None`` if none was declared."""

        if not self._family_file.exists():
            return None
        loaded = json.loads(self._family_file.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def run_grid(
        self,
        coordinates: list[WarDialCoordinate],
        resume: bool = True,
        perturb_fn: Callable[[str, random.Random, float], str] | None = None,
        run_attack_sweep: bool = False,
        family_level: float = 0.05,
        multiplicity_method: str = "bonferroni",
    ) -> list[DialResult]:
        """Enumerate all coordinates and append results to the JSONL output.

        The comparison family is declared from ``coordinates`` before any dial
        runs, so the per-test level recorded alongside the results corresponds to
        the grid that produced them.
        """
        self.declare_family(
            coordinates,
            family_level=family_level,
            method=multiplicity_method,
        )
        output_file = self.output_dir / "war_dial_results.jsonl"
        results: list[DialResult] = []
        existing_dial_ids: set[str] = set()
        if resume and output_file.exists():
            for line in output_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                existing_dial_ids.add(record["dial_id"])

        for index, coordinate in enumerate(coordinates):
            dial_id = f"{coordinate.iteration:04d}-{index:04d}"
            if dial_id in existing_dial_ids:
                continue
            result = self.dial(coordinate, dial_id=dial_id, run_attack_sweep=run_attack_sweep)
            results.append(result)
            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")

        return results

    def gate_report(self) -> dict[str, Any]:
        """Produce a four-state gate report for the saved war-dial results.

        The four states are artifact_complete, checks_passed,
        promotion_eligible, and deployment_ready.  Each state is accompanied
        by reasoning that cites the JSONL evidence.
        """
        summary = self.summary()
        total = summary.get("total_dials", 0)
        records: list[dict[str, Any]] = []
        output_file = self.output_dir / "war_dial_results.jsonl"
        if output_file.exists():
            for line in output_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                records.append(json.loads(line))

        family = self.load_family()
        multiplicity_controlled = (
            family is not None and family.get("method") in ("bonferroni", "sidak")
        )

        artifact_complete = total > 0
        checks_passed = (
            summary.get("coding_failed", 0) == 0
            and summary.get("kgw_verified", 0) > 0
            and summary.get("stego_decoded_ok", 0) > 0
            and multiplicity_controlled
        )
        promotion_eligible = checks_passed and artifact_complete
        deployment_ready = False  # Mock provider + synthetic text only.

        findings: list[dict[str, Any]] = []
        if summary.get("coding_failed", 0) > 0:
            findings.append({
                "severity": "blocker",
                "category": "coding",
                "message": f"{summary['coding_failed']} dial(s) failed the coding test",
            })
        if summary.get("kgw_verified", 0) == 0:
            findings.append({
                "severity": "blocker",
                "category": "kgw",
                "message": "No KGW coordinate self-verified",
            })
        if summary.get("stego_decoded_ok", 0) == 0:
            findings.append({
                "severity": "blocker",
                "category": "stego",
                "message": "No stego coordinate decoded successfully",
            })
        if not multiplicity_controlled:
            findings.append({
                "severity": "blocker",
                "category": "multiplicity",
                "message": (
                    "No multiplicity correction is recorded for this sweep. The per-test "
                    "level is not the family-wise level, so any false-positive rate "
                    "reported from this grid is a multiplicity artifact."
                ),
            })
        if not any(r.get("survival_surface") for r in records):
            findings.append({
                "severity": "info",
                "category": "calibration",
                "message": "No survival surface produced; run with --attack-sweep to generate one",
            })

        return {
            "report_type": "four-state-gate",
            "output_dir": str(self.output_dir),
            "artifact_complete": artifact_complete,
            "checks_passed": checks_passed,
            "promotion_eligible": promotion_eligible,
            "deployment_ready": deployment_ready,
            "summary": summary,
            "sweep_family": family,
            "findings": findings,
            "reasoning": {
                "artifact_complete": f"{total} dial record(s) written to {output_file.name}",
                "checks_passed": (
                    "All coding tests passed, at least one KGW and one stego coordinate "
                    "succeeded, and a multiplicity-corrected comparison family is recorded"
                ),
                "promotion_eligible": "Checks passed and artifacts exist; scientific promotion requires frozen calibration",
                "deployment_ready": "False: demo grid uses mock provider and synthetic text only",
            },
            "timestamp": _now_iso(),
        }

    def write_gate_report(self) -> Path:
        """Write the four-state gate report to the output directory."""
        report = self.gate_report()
        report_file = self.output_dir / "gate_report.json"
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report_file

    def summary(self) -> dict[str, Any]:
        """Compute a summary of saved results."""
        output_file = self.output_dir / "war_dial_results.jsonl"
        if not output_file.exists():
            return {"total_dials": 0}

        records: list[dict[str, Any]] = []
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))

        total = len(records)
        coding_passed = sum(1 for r in records if r["coding_passed"])
        kgw_ok = sum(1 for r in records if r["kgw_decision"] == "verified compatible watermark")
        stego_ok = sum(1 for r in records if r["stego_decode_status"] == "ok")
        perturbed_ok = sum(
            1 for r in records if r.get("perturbed_decode_status") in ("ok", None)
        )
        return {
            "total_dials": total,
            "coding_passed": coding_passed,
            "coding_failed": total - coding_passed,
            "kgw_created": sum(1 for r in records if r["kgw_created"]),
            "kgw_verified": kgw_ok,
            "stego_encoded": sum(1 for r in records if r["stego_encoded"]),
            "stego_decoded_ok": stego_ok,
            "perturbed_decoded_ok": perturbed_ok,
            "survival_rate": perturbed_ok / total if total else 0.0,
        }
