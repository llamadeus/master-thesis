#!/usr/bin/env python3
"""Simulate and export the nodal demands needed by the paper's Hanoi scenario.

Usage: .venv/bin/python tools/analysis/dataset/export_demand_patterns.py scenario-0 --week 2

The JSON preserves half-hourly EPANET demands through the scenario endpoint,
including fixed parameter uncertainty and pressure-dependent demand. One extra
complete day preserves the unchanged Typst reader's whole-day array contract
without wrapping the endpoint to day zero. Annual input patterns stay intact.
No aggregation, rounding, normalization, or resampling occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def scenario_identifier(value: str) -> str:
    suffix = value.removeprefix("scenario-")
    if not suffix.isdecimal() or not 0 <= int(suffix) <= 99:
        raise argparse.ArgumentTypeError("Expected a scenario ID from 0 to 99 or scenario-0 to scenario-99")
    return f"scenario-{int(suffix)}"


def week_number(value: str) -> int:
    if not value.isdecimal() or not 1 <= int(value) <= 52:
        raise argparse.ArgumentTypeError("Expected a complete week from 1 to 52")
    return int(value)


def figure_selection(week: int) -> dict:
    """Use day zero as Monday and select the Thursday of a complete week."""
    start = (week - 1) * 7
    return {"week_number": week, "week_start_day": start,
            "day_start_day": start + 3, "day_zero_weekday": "Monday"}


def time_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Expected an INP time in hours:minutes:seconds, got {value!r}")
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def read_raw_patterns(scenario_dir: Path) -> dict:
    """Read the prepared dataset's one-pattern-per-junction INP contract."""
    inp_path = scenario_dir / "scenario.inp"
    manifest_path = scenario_dir / "manifest.json"
    inp_bytes = inp_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sections: dict[str, list[list[str]]] = {}
    section = ""
    for source_line in inp_bytes.decode("utf-8").splitlines():
        line = source_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.upper()
        else:
            sections.setdefault(section, []).append(line.split())

    times = {" ".join(row[:-1]).upper(): row[-1] for row in sections["[TIMES]"]}
    step = time_seconds(times["PATTERN TIMESTEP"])
    if step <= 0 or 86400 % step:
        raise ValueError("Pattern timestep must divide one day exactly")
    if time_seconds(times["PATTERN START"]) != 0:
        raise ValueError("Expected PATTERN START zero")
    samples = 365 * 86400 // step
    start_day = manifest["calendar"]["start_day"]
    if type(start_day) is not int or start_day != 0:
        raise ValueError("Expected the scenario to begin at calendar day zero")

    patterns: dict[str, list[float]] = {}
    for pattern, *values in sections["[PATTERNS]"]:
        patterns.setdefault(pattern, []).extend(float(value) for value in values)

    demands: dict[str, tuple[float, str]] = {}
    for node, _elevation, *demand in sections["[JUNCTIONS]"]:
        if demand:
            raise ValueError("Expected prepared scenarios with demands in [DEMANDS] only")
    for row in sections["[DEMANDS]"]:
        if len(row) != 3 or row[0] in demands:
            raise ValueError("Expected exactly one explicit demand pattern per junction")
        node, base, pattern = row
        demands[node] = (float(base), pattern)

    node_ids = manifest["monitored_node_ids"]
    if set(node_ids) != set(demands) or len(set(node_ids)) != len(node_ids):
        raise ValueError("Manifest nodes and demand junctions do not match")
    nodes = []
    for node in node_ids:
        base, pattern = demands[node]
        values = patterns[pattern]
        if len(values) != samples:
            raise ValueError(f"{pattern} has {len(values)} values, expected {samples} for 365 days")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"{pattern} contains non-finite or negative multipliers")
        nodes.append({"id": node, "pattern_id": pattern, "base_demand": base, "multipliers": values})

    options = {" ".join(row[:-1]).upper(): row[-1] for row in sections["[OPTIONS]"]}
    return {
        "schema_version": 1,
        "scenario_id": manifest["id"],
        "network": manifest["network"],
        "quantity": "nodal_demand_pattern_multiplier",
        "unit": "dimensionless",
        "base_demand_unit": options["UNITS"],
        "demand_multiplier": float(options["DEMAND MULTIPLIER"]),
        "timestep_seconds": step,
        "samples_per_node": samples,
        "calendar_days": 365,
        "first_sample_calendar_day": start_day,
        "simulation_duration_seconds": manifest["duration_seconds"],
        "source": {
            "inp": inp_path.relative_to(PROJECT_ROOT).as_posix() if inp_path.is_relative_to(PROJECT_ROOT) else str(inp_path),
            "inp_sha256": hashlib.sha256(inp_bytes).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "nodes": nodes,
    }


def paper_record_days(raw: dict) -> int:
    """Keep complete days and the next day's first sample at the right endpoint."""
    duration = raw["simulation_duration_seconds"]
    if duration <= 0 or duration % 86400:
        raise ValueError("Expected a positive, whole-day scenario duration")
    days = duration // 86400 + 1
    if days > raw["calendar_days"]:
        raise ValueError("Scenario endpoint exceeds the available annual demand record")
    return days


def read_raw_demands(scenario_dir: Path) -> dict:
    """Run hydraulics only for the complete days needed by the paper's panels."""
    from epanet_plus import EPyT
    from epyt_flow.simulation import ScenarioConfig

    raw = read_raw_patterns(scenario_dir)
    record_days = paper_record_days(raw)
    raw["pattern_calendar_days"] = raw["calendar_days"]
    raw["calendar_days"] = record_days
    raw["samples_per_node"] = record_days * 86400 // raw["timestep_seconds"]
    if raw["base_demand_unit"] != "CMH":
        raise ValueError("Expected Hanoi flow units CMH (cubic metres per hour)")
    config_path = scenario_dir / "scenario.epytflow_scenario_config"
    config = ScenarioConfig.load_from_file(str(config_path))
    if config.system_events or config.custom_controls:
        raise ValueError("Expected prepared scenarios without system events or custom controls")

    with TemporaryDirectory(prefix="demand-patterns-") as temporary:
        inp_path = Path(temporary) / "scenario.inp"
        copyfile(scenario_dir / "scenario.inp", inp_path)
        with EPyT(str(inp_path), use_project=True,
                  rpt_file_out=str(Path(temporary) / "hydraulics.rpt")) as api:
            api.set_error_handling(raise_exception_on_error=True, warn_on_error=True)
            # Match ScenarioSimulator._prepare_simulation: apply the stored
            # uncertainty once to a fresh copy of the scenario inputs.
            if config.model_uncertainty is not None:
                config.model_uncertainty.apply(api)
            api.set_simulation_duration(raw["calendar_days"] * 86400)
            indices = [api.get_node_idx(node["id"]) for node in raw["nodes"]]
            nodes = [{"id": node["id"],
                      "base_demand": api.get_node_base_demand(index),
                      "demand": []}
                     for node, index in zip(raw["nodes"], indices)]
            api.openH()
            try:
                api.initH(0)
                for sample in range(raw["samples_per_node"]):
                    time = api.runH()
                    if time != sample * raw["timestep_seconds"]:
                        raise ValueError(f"Unexpected hydraulic sample time {time}")
                    for node, index in zip(nodes, indices):
                        value = api.get_node_demand(index)
                        # Additive pattern uncertainty can produce signed demands.
                        # Preserve the simulator output without clipping it.
                        if not math.isfinite(value):
                            raise ValueError(f"Invalid demand at node {node['id']}, time {time}")
                        node["demand"].append(value)
                    if api.nextH() <= 0 and sample + 1 < raw["samples_per_node"]:
                        raise ValueError("Hydraulic simulation ended before the paper record was complete")
            finally:
                api.closeH()

    raw.update(schema_version=3, quantity="actual_nodal_demand", unit="m3/h", nodes=nodes)
    raw["source"]["scenario_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    raw["hydraulic_simulation_duration_seconds"] = raw["calendar_days"] * 86400
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_id", type=scenario_identifier, help="0 to 99, optionally prefixed with scenario-")
    parser.add_argument("--week", type=week_number, default=2,
                        help="Week to plot, numbered 1 to 52 (default: 2). Week 1 starts on day 0, treated as Monday.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "dataset")
    args = parser.parse_args()
    scenario_dir = PROJECT_ROOT / "data" / "scenarios" / "hanoi" / args.scenario_id
    try:
        selection = figure_selection(args.week)
        # Reject weeks outside the paper's scenario before starting hydraulics.
        manifest = json.loads((scenario_dir / "manifest.json").read_bytes())
        if (selection["week_start_day"] + 7) * 86400 > manifest["duration_seconds"]:
            raise ValueError("Selected week must lie entirely within the scenario period")
        raw = read_raw_demands(scenario_dir)
        raw["figure_selection"] = selection
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"demand-patterns.json"
    # Keep one raw node record per line without sacrificing float precision.
    metadata = {key: value for key, value in raw.items() if key != "nodes"}
    header = json.dumps(metadata, indent=2, allow_nan=False)
    node_lines = ",\n".join(json.dumps(node, separators=(",", ":"), allow_nan=False) for node in raw["nodes"])
    output.write_text(header[:-2] + ',\n  "nodes": [\n' + node_lines + "\n  ]\n}\n", encoding="utf-8")
    print(f"Exported {len(raw['nodes'])} nodes x {raw['samples_per_node']} raw samples to {output}")


if __name__ == "__main__":
    main()
