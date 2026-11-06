#!/usr/bin/env python3
"""Watermark war-dial CLI: systematically enumerate textual watermark parameters.

Like wardialing phone numbers, but for watermark parameter space: each "dial"
is one coordinate (prompt + KGW params + stego channel + perturbation).  The
runner executes a deterministic coding test, an encoding roundtrip, and a
stochastic perturbation, then saves every result so the grid can be refined in
the next iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from watermark_lab.war_dial.harness import WarDialHarness
from watermark_lab.war_dial.reconcile import WarDialReconciler
from watermark_lab.war_dial.space import (
    KgwDial,
    PerturbationDial,
    StegoDial,
    WarDialCoordinate,
)


def _build_demo_grid(iteration: int = 0, attack_sweep: bool = False) -> list[WarDialCoordinate]:
    """Return a small default grid for the war-dial demonstration.

    The grid is intentionally tiny when ``attack_sweep`` is enabled, because
    each KGW coordinate triggers a full attack sweep with before/after
    detection calls.
    """
    prompts = ["the quick brown fox"]
    gammas = [0.25, 0.4]
    deltas = [2.0]
    max_new_tokens_list = [80]
    stego_channels = ["whitespace", "unicode"]
    perturbations = [PerturbationDial("builtin-word-jitter", 0.1, "seed-a")]
    if not attack_sweep:
        # Add an extra prompt and a second perturbation only when we are not
        # also paying for the expensive KGW attack sweep.
        prompts.append("a different prompt text")
        perturbations.append(PerturbationDial("deletion", 0.1, "seed-b"))

    # Long enough cover for the short stego envelope (~104 bits) on both the
    # whitespace and unicode channels.
    cover = (
        "The quick brown fox jumps over the lazy dog several times "
        "to provide enough words for embedding a modest hidden payload "
        "into the cover text and then some extra words to guarantee capacity "
        "with a safety margin of additional words beyond the threshold "
        "so that even the compact short envelope fits comfortably inside the text"
    )

    coordinates: list[WarDialCoordinate] = []
    for prompt in prompts:
        for gamma in gammas:
            for delta in deltas:
                for max_new_tokens in max_new_tokens_list:
                    kgw = KgwDial(
                        prompt=prompt,
                        gamma=gamma,
                        delta=delta,
                        context_width=1,
                        max_new_tokens=max_new_tokens,
                    )
                    for channel in stego_channels:
                        stego = StegoDial(
                            channel=channel,
                            cover=cover,
                            payload="ok",
                            secret="war-dial-secret",
                        )
                        for perturbation in perturbations:
                            coordinates.append(
                                WarDialCoordinate(
                                    kgw=kgw,
                                    stego=stego,
                                    perturbation=perturbation,
                                    iteration=iteration,
                                )
                            )
    return coordinates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate watermark parameters and run coding + encoding tests."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/war-dial",
        help="Directory to save war_dial_results.jsonl and summaries.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterative passes over the grid.",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Only reconcile existing results and print the next-grid suggestion.",
    )
    parser.add_argument(
        "--coding-test",
        nargs="*",
        default=None,
        help=(
            "Arguments passed to `python` for the coding test. "
            "If omitted, a fast in-process stego roundtrip is used."
        ),
    )
    parser.add_argument(
        "--attack-sweep",
        action="store_true",
        help="Run a KGW attack sweep for each KGW coordinate and compute an empirical survival surface.",
    )
    parser.add_argument(
        "--survival-surface",
        action="store_true",
        help="Aggregate attack sweep results into a survival surface JSON file.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)

    if args.reconcile:
        reconciler = WarDialReconciler(output_dir)
        print(json.dumps(reconciler.reconcile(), indent=2))
        return 0

    harness = WarDialHarness(
        output_dir=output_dir,
        coding_test_args=list(args.coding_test) if args.coding_test else None,
    )

    for iteration in range(args.iterations):
        coordinates = _build_demo_grid(iteration=iteration, attack_sweep=args.attack_sweep)
        harness.run_grid(
            coordinates,
            resume=True,
            run_attack_sweep=args.attack_sweep,
        )

    if args.survival_surface:
        _write_survival_surface(output_dir)

    gate_path = harness.write_gate_report()
    summary = harness.summary()
    summary["attack_sweep_enabled"] = args.attack_sweep
    summary["gate_report_path"] = str(gate_path)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("coding_failed", 0) == 0 else 1


def _write_survival_surface(output_dir: Path) -> None:
    """Aggregate attack results from all dial records into one survival surface."""
    from watermark_lab.frontier.edit_receipt import compute_survival_surface

    results_file = output_dir / "war_dial_results.jsonl"
    surface_file = output_dir / "survival_surface.json"
    if not results_file.exists():
        return

    attack_results: list[dict[str, Any]] = []
    for line in results_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        attack_results.extend(record.get("attack_results", []))

    if not attack_results:
        surface_file.write_text(
            json.dumps({"error": "no attack results found"}, indent=2),
            encoding="utf-8",
        )
        return

    surface = compute_survival_surface(
        attack_results,
        edit_family=["deletion", "insertion", "substitution", "shuffle"],
    )
    surface_file.write_text(
        json.dumps(surface.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
