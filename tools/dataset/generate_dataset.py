#!/usr/bin/env python3
"""Optionally prepare the same LeakDB scenarios the trainer creates on demand."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support both `python -m tools.dataset.generate_dataset` and direct invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.trainer.scenarios import ensure_scenario, validate_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-dir", default="hanoi",
                        help="Direct child folder of data/scenarios/ (default: hanoi)")
    parser.add_argument("--uncertainty-seed", type=int, default=42)
    parser.add_argument("--scenario-id", type=int, action="append",
                        help="LeakDB ID (0..999); repeatable; defaults to 0..99")
    return parser


def prepare_dataset(*, scenarios_dir: str = "hanoi", uncertainty_seed: int = 42,
                    scenario_ids: tuple[int, ...] = tuple(range(100))) -> None:
    validate_source("hanoi", scenarios_dir, uncertainty_seed)
    if not scenario_ids or any(isinstance(value, bool) or not isinstance(value, int)
                               or not 0 <= value < 1000 for value in scenario_ids):
        raise ValueError("scenario_ids must contain LeakDB IDs in 0..999")
    for scenario_id in dict.fromkeys(scenario_ids):
        path = ensure_scenario("hanoi", f"scenario-{scenario_id}",
                               scenarios_dir=scenarios_dir, uncertainty_seed=uncertainty_seed)
        print(f"Ready: {path}")


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(scenarios_dir=args.scenarios_dir, uncertainty_seed=args.uncertainty_seed,
                    scenario_ids=tuple(args.scenario_id) if args.scenario_id is not None else tuple(range(100)))


if __name__ == "__main__":
    main()
