"""Reconcile war-dial results and propose the next parameter grid.

Given a JSONL result file, the reconciler extracts failed dials, identifies
boundary coordinates that changed outcomes, and suggests a refined grid for the
next iteration.  It is intentionally small so it can be run inside an agent
loop after each war-dial batch.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any


class WarDialReconciler:
    """Analyze saved war-dial results and recommend the next iteration."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.results_file = self.output_dir / "war_dial_results.jsonl"

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.results_file.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.results_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    def failed_dials(self) -> list[dict[str, Any]]:
        """Return dials where coding, KGW, stego, or perturbation tests failed."""
        failures: list[dict[str, Any]] = []
        for record in self._load_records():
            if not record["coding_passed"]:
                failures.append({"dial_id": record["dial_id"], "failure": "coding"})
            if record["kgw_decision"] not in ("verified compatible watermark", None):
                failures.append({"dial_id": record["dial_id"], "failure": "kgw"})
            if record["stego_decode_status"] not in ("ok", "none"):
                failures.append({"dial_id": record["dial_id"], "failure": "stego_decode"})
            perturbed = record.get("perturbed_decode_status")
            if perturbed not in ("ok", None, "none"):
                failures.append({"dial_id": record["dial_id"], "failure": "perturbed_decode"})
        return failures

    def boundary_coordinates(self) -> list[dict[str, Any]]:
        """Find adjacent parameter pairs where the outcome flips.

        For numeric KGW parameters (gamma, delta, max_new_tokens) we sort the
        records by that parameter and report adjacent pairs whose survival flag
        differs.  This helps locate the edge of the watermark's operating range.
        """
        records = self._load_records()
        if not records:
            return []

        boundaries: list[dict[str, Any]] = []
        numeric_axes = [
            ("gamma", lambda r: r["coordinate"]["kgw"]["gamma"] if r["coordinate"]["kgw"] else None),
            ("delta", lambda r: r["coordinate"]["kgw"]["delta"] if r["coordinate"]["kgw"] else None),
            (
                "max_new_tokens",
                lambda r: r["coordinate"]["kgw"]["max_new_tokens"]
                if r["coordinate"]["kgw"]
                else None,
            ),
        ]

        def _survival(record: dict[str, Any]) -> bool:
            return record.get("perturbed_decode_status") in ("ok", None)

        for axis, getter in numeric_axes:
            axis_records: list[tuple[Any, dict[str, Any]]] = [
                (getter(r), r) for r in records if getter(r) is not None
            ]
            axis_records.sort(key=lambda pair: pair[0])
            for (a, rec_a), (b, rec_b) in pairwise(axis_records):
                if _survival(rec_a) != _survival(rec_b):
                    boundaries.append(
                        {
                            "axis": axis,
                            "left": {"value": a, "survived": _survival(rec_a)},
                            "right": {"value": b, "survived": _survival(rec_b)},
                            "dial_ids": [rec_a["dial_id"], rec_b["dial_id"]],
                        }
                    )
        return boundaries

    def next_grid_suggestion(self) -> dict[str, Any]:
        """Propose a refined grid based on boundaries and failures."""
        failures = self.failed_dials()
        boundaries = self.boundary_coordinates()
        records = self._load_records()
        total = len(records)

        failure_counts: dict[str, int] = defaultdict(int)
        for f in failures:
            failure_counts[f["failure"]] += 1

        # If most failures are on one channel, zoom in there.
        channel_counts: dict[str, int] = defaultdict(int)
        for r in records:
            ch = r["coordinate"]["stego"]["channel"] if r["coordinate"]["stego"] else "none"
            if r["stego_decode_status"] not in ("ok", "none"):
                channel_counts[ch] += 1

        focus_channel = (
            max(channel_counts.items(), key=lambda item: item[1])[0]
            if channel_counts
            else None
        )

        # If we have numeric boundaries, bisect them.
        refinements: list[dict[str, Any]] = []
        for b in boundaries[:3]:
            left = b["left"]["value"]
            right = b["right"]["value"]
            refinements.append(
                {
                    "axis": b["axis"],
                    "suggested_values": sorted({left, (left + right) / 2, right}),
                    "reason": "outcome flips between adjacent values",
                }
            )

        return {
            "total_dials": total,
            "failure_count": len(failures),
            "failure_breakdown": dict(failure_counts),
            "focus_channel": focus_channel,
            "boundaries": boundaries,
            "refinements": refinements,
            "recommendation": "bisect reported boundaries and re-run the failed channel"
            if refinements
            else "expand parameter ranges or add stochastic perturbation seeds",
        }

    def reconcile(self) -> dict[str, Any]:
        """Full reconciliation report for the current result file."""
        return {
            "output_dir": str(self.output_dir),
            "summary": self._summary(),
            "failures": self.failed_dials()[:20],
            "boundaries": self.boundary_coordinates()[:10],
            "next_grid": self.next_grid_suggestion(),
        }

    def _summary(self) -> dict[str, Any]:
        from watermark_lab.war_dial.harness import WarDialHarness

        return WarDialHarness(self.output_dir).summary()
