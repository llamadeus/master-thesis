"""Reproducible, lazily materialized LeakDB Hanoi scenarios.

Native simulator and legacy NumPy operations are serialized within each process.
An OS-backed per-scenario lock coordinates publication between processes.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
from shutil import copy2
import sys
from tempfile import TemporaryDirectory
from threading import RLock

import numpy as np
from filelock import FileLock

from .constants import ARSENIC_EVENT_EXCLUDED_NODE_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MSX_TEMPLATE = PROJECT_ROOT / "external/Environment/hanoi_arsenic_contamination.msx"
SCENARIO_MANIFEST_VERSION = 2
DURATION_DAYS = 25
HYDRAULIC_STEPS_PER_DAY = 48
CHLORINE_SOURCE_NODE = "1"
CHLORINE_PATTERN_ID = "chlorine-injection-at-node-1"
SCENARIO_FILES = ("scenario.inp", "scenario.msx", "scenario.epytflow_scenario_config")
SCENARIO_CONFIG_FILE = "scenario.epytflow_scenario_config"
_NATIVE_LOCK = RLock()


def validate_source(network: str, scenarios_dir: str, uncertainty_seed: int) -> None:
    if network != "hanoi":
        raise ValueError("trainer scenarios currently support only LeakDB Hanoi")
    if not isinstance(scenarios_dir, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scenarios_dir) is None:
        raise ValueError("scenarios_dir must name a direct child folder of data/scenarios/")
    if isinstance(uncertainty_seed, bool) or not isinstance(uncertainty_seed, int) or uncertainty_seed < 0:
        raise ValueError("uncertainty_seed must be a non-negative integer")


def scenario_number(reference: str) -> int:
    if not isinstance(reference, str) or re.fullmatch(r"scenario-(0|[1-9][0-9]{0,2})", reference) is None:
        raise ValueError("scenario ID must be canonical scenario-<ID>, with ID in 0..999")
    return int(reference.removeprefix("scenario-"))


def scenario_directory_name(reference: str) -> str:
    """Validate and return the canonical on-disk scenario directory name."""
    return f"scenario-{scenario_number(reference)}"


def _eligible_arsenic_node_ids(junctions) -> list[str]:
    return [
        str(node)
        for node in junctions
        if str(node) != CHLORINE_SOURCE_NODE
        and str(node) not in ARSENIC_EVENT_EXCLUDED_NODE_IDS
    ]


def derive_scenario_seeds(scenario_id: int, uncertainty_seed: int = 42) -> dict[str, int]:
    validate_source("hanoi", "hanoi", uncertainty_seed)
    if isinstance(scenario_id, bool) or not isinstance(scenario_id, int) or not 0 <= scenario_id < 1000:
        raise ValueError("LeakDB scenario ID must be an integer in 0..999")
    return {
        "master_seed": uncertainty_seed,
        "demand_seed": 1000 + scenario_id,
        "uncertainty_seed": int(np.random.SeedSequence([uncertainty_seed, scenario_id]).generate_state(1, dtype=np.uint32)[0]),
    }


@contextmanager
def legacy_numpy_seed(seed: int):
    with _NATIVE_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


def load_seeded_leakdb_scenario(scenario_id: int, download_dir: Path):
    from epyt_flow.data.benchmarks import load_leakdb_scenarios
    with legacy_numpy_seed(1000 + scenario_id), silence_native_stdout():
        [config] = load_leakdb_scenarios(
            scenarios_id=[str(scenario_id)], use_net1=False,
            download_dir=str(download_dir), verbose=False,
        )
    return config


def model_uncertainty(seed: int):
    from epyt_flow.uncertainty import ModelUncertainty, RelativeUniformUncertainty, AbsoluteGaussianUncertainty
    return ModelUncertainty(
        seed=seed,
        global_pipe_length_uncertainty=RelativeUniformUncertainty(low=0.8, high=1.2),
        global_pipe_roughness_uncertainty=RelativeUniformUncertainty(low=0.8, high=1.2),
        global_base_demand_uncertainty=RelativeUniformUncertainty(low=0.8, high=1.2),
        global_demand_pattern_uncertainty=AbsoluteGaussianUncertainty(mean=0, scale=0.1),
    )


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scenario_file_sha256(path: Path) -> str:
    """Hash canonical scenario content, excluding gzip container metadata."""
    if path.name != SCENARIO_CONFIG_FILE:
        return file_sha256(path)
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_content_hash(path: Path) -> str:
    files = {name: scenario_file_sha256(path / name) for name in SCENARIO_FILES}
    files["manifest.json"] = file_sha256(path / "manifest.json")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _contract(scenario_id: int, uncertainty_seed: int) -> dict:
    return {
        "schema_version": SCENARIO_MANIFEST_VERSION,
        "id": f"scenario-{scenario_id}",
        "network": "hanoi",
        "source": {"loader": "epyt_flow.data.benchmarks.load_leakdb_scenarios",
                   "scenario_id": str(scenario_id), "use_net1": False,
                   "epyt_flow_version": importlib.metadata.version("epyt-flow"),
                   "msx_template_sha256": file_sha256(MSX_TEMPLATE)},
        "seeds": derive_scenario_seeds(scenario_id, uncertainty_seed),
        "uncertainties": {
            "pipe_length": {"distribution": "relative-uniform", "low": 0.8, "high": 1.2},
            "pipe_roughness": {"distribution": "relative-uniform", "low": 0.8, "high": 1.2},
            "base_demand": {"distribution": "relative-uniform", "low": 0.8, "high": 1.2},
            "demand_pattern": {"distribution": "absolute-gaussian", "mean": 0, "scale": 0.1},
        },
        "leakage_events": "removed",
        "hydraulic_timestep_seconds": 1800,
        "duration_seconds": DURATION_DAYS * 86400,
        "calendar": {"start_day": 0},
        "chlorine_source_ids": [CHLORINE_SOURCE_NODE],
    }


def validate_scenario(path: Path, scenario_id: int, uncertainty_seed: int) -> dict:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        expected = _contract(scenario_id, uncertainty_seed)
        if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("manifest does not match the requested scenario settings or seeds")
        monitored_node_ids = manifest.get("monitored_node_ids")
        if (
            not isinstance(monitored_node_ids, list)
            or manifest.get("eligible_arsenic_node_ids")
            != _eligible_arsenic_node_ids(monitored_node_ids)
        ):
            raise ValueError(
                "manifest eligible_arsenic_node_ids do not match the arsenic event node policy"
            )
        hashes = {name: scenario_file_sha256(path / name) for name in SCENARIO_FILES}
        if manifest.get("files") != hashes:
            raise ValueError("scenario files do not match their manifest hashes")
        return manifest
    except (OSError, ValueError) as exc:
        raise ValueError(f"Incompatible or incomplete scenario at {path}: {exc}. "
                         "Choose a different scenarios_dir; existing files are not overwritten.") from exc


def ensure_scenario(network: str, reference: str, *, scenarios_dir: str = "hanoi",
                    uncertainty_seed: int = 42, project_root: Path = PROJECT_ROOT) -> Path:
    validate_source(network, scenarios_dir, uncertainty_seed)
    scenario_id = scenario_number(reference)
    base = (project_root / "data" / "scenarios").resolve()
    root = (base / scenarios_dir).resolve()
    if root.parent != base:
        raise ValueError("scenarios_dir must resolve to a direct child of data/scenarios/")
    target = root / scenario_directory_name(reference)
    if target.exists():
        validate_scenario(target, scenario_id, uncertainty_seed)
        return target
    root.mkdir(parents=True, exist_ok=True)
    # Never unlink lock files: replacing their inode can let two writers enter.
    with FileLock(str(root / f".{reference}.lock"), timeout=600):
        if target.exists():
            validate_scenario(target, scenario_id, uncertainty_seed)
            return target
        with TemporaryDirectory(prefix=f".{reference}-", dir=root) as staging:
            with _NATIVE_LOCK:
                _generate_scenario(Path(staging), scenario_id, uncertainty_seed)
            validate_scenario(Path(staging), scenario_id, uncertainty_seed)
            Path(staging).rename(target)
    return target


def _generate_scenario(output_dir: Path, scenario_id: int, uncertainty_seed: int) -> None:
    from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScenarioSimulator, SensorConfig
    from epyt_flow.simulation.events import Leakage
    seeds = derive_scenario_seeds(scenario_id, uncertainty_seed)
    # A fresh loader workspace is essential: its INP cache otherwise bypasses the seed.
    with TemporaryDirectory(prefix="trainer-leakdb-") as raw_dir:
        loaded = load_seeded_leakdb_scenario(scenario_id, Path(raw_dir))
        # Remove leaks before simulator initialization can modify the network.
        # Rebuild explicitly: ScenarioConfig(scenario_config=..., system_events=[])
        # inherits the original events instead of clearing them.
        attributes = loaded.get_attributes()
        attributes.pop("type_id", None)
        attributes.update(
            f_msx_in=str(MSX_TEMPLATE),
            model_uncertainty=model_uncertainty(seeds["uncertainty_seed"]),
            system_events=[event for event in loaded.system_events if not isinstance(event, Leakage)],
        )
        working = ScenarioConfig(**attributes)
        with silence_native_stdout(), ScenarioSimulator(scenario_config=working) as simulator:
            if simulator.epanet_api.get_hydraulic_time_step() != 1800:
                raise ValueError("LeakDB scenarios must use a 30-minute hydraulic timestep")
            simulator.set_general_parameters(simulation_duration=DURATION_DAYS * 86400,
                                             reporting_time_step=1800)
            simulator.epanet_api.settimeparam(EpanetConstants.EN_PATTERNSTART, 0)
            all_nodes = list(simulator.sensor_config.nodes)
            all_links = list(simulator.sensor_config.links)
            excluded = set(simulator.epanet_api.get_all_reservoirs_id()) | set(simulator.epanet_api.get_all_tanks_id())
            junctions = [node for node in all_nodes if node not in excluded]
            simulator.sensor_config = SensorConfig.create_empty_sensor_config(simulator.sensor_config)
            simulator.set_pressure_sensors(junctions)
            simulator.set_demand_sensors(junctions)
            simulator.set_flow_sensors(all_links)
            simulator.set_bulk_species_node_sensors({"Chlorine": all_nodes, "AsIII": all_nodes})
            # No simulation here: uncertainty is applied exactly once at runtime.
            simulator.save_to_epanet_file(
                str(output_dir / "scenario.inp"), str(output_dir / "scenario.msx"),
                export_sensor_config=False,
            )
            configure_epanet_report(output_dir / "scenario.inp", junctions)
            config = simulator.get_scenario_config()
            attributes = config.get_attributes()
            attributes.pop("type_id", None)
            attributes["f_inp_in"] = "scenario.inp"
            attributes["f_msx_in"] = "scenario.msx"
            if attributes["network_topology"] is not None:
                attributes["network_topology"].name = "scenario.inp"
            ScenarioConfig(**attributes).save_to_file(str(output_dir / "scenario"))
    # Keep the project's chemistry and actuator pattern; native export changes neither.
    copy2(MSX_TEMPLATE, output_dir / "scenario.msx")
    configure_chlorine_control_source(output_dir / "scenario.msx")
    (output_dir / "scenario.inp.rpt").unlink(missing_ok=True)
    manifest = _contract(scenario_id, uncertainty_seed)
    manifest.update(monitored_node_ids=junctions,
                    eligible_arsenic_node_ids=_eligible_arsenic_node_ids(junctions),
                    files={name: scenario_file_sha256(output_dir / name) for name in SCENARIO_FILES})
    (output_dir / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))


@contextmanager
def silence_native_stdout():
    """Suppress output written directly to stdout by EPANET and EPANET-MSX."""
    try:
        sys.stdout.flush()
    except (AttributeError, OSError):
        pass
    _flush_c_streams()
    saved_stdout = os.dup(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            try:
                yield
            finally:
                try:
                    sys.stdout.flush()
                except (AttributeError, OSError):
                    pass
                _flush_c_streams()
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def _flush_c_streams() -> None:
    try:
        # Python 3.14 no longer accepts ``None`` as a LoadLibrary name on
        # Windows.  The Universal CRT owns the process' C stdio streams there;
        # on POSIX, the existing process handle remains the appropriate target.
        library = ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)
        library.fflush(None)
    except (AttributeError, OSError, TypeError):
        pass


def _replace_msx_section(lines: list[str], name: str, body: list[str]) -> list[str]:
    header = f"[{name}]"
    start = next((index for index, line in enumerate(lines) if line.strip().upper() == header), None)
    if start is None:
        return [*lines, "", header, *body]
    end = next((index for index in range(start + 1, len(lines)) if lines[index].strip().startswith("[")), len(lines))
    return [*lines[:start + 1], *body, *lines[end:]]


def configure_epanet_report(inp_path: Path, node_ids: list[str]) -> None:
    """Install the sensor report without mixing text and byte offsets."""
    lines = inp_path.read_text(encoding="utf-8").splitlines()
    report_start = next(
        (index for index, line in enumerate(lines) if line.strip().upper() == "[REPORT]"),
        None,
    )
    if report_start is not None:
        report_end = next(
            (index for index in range(report_start + 1, len(lines))
             if lines[index].strip().startswith("[")),
            len(lines),
        )
        del lines[report_start:report_end]
    end = next(
        (index for index, line in enumerate(lines) if line.strip().upper() == "[END]"),
        None,
    )
    if end is None:
        raise ValueError("EPANET export has no [END] section")
    prefix = lines[:end]
    while prefix and not prefix[-1].strip():
        prefix.pop()
    report = [
        "[REPORT]",
        "ENERGY YES",
        "STATUS YES",
        "Pressure YES",
        "Flow YES",
        "Demand YES",
        f"NODES {' '.join(node_ids)}",
        "LINKS ALL",
    ]
    lines = [*prefix, "", *report, "", *lines[end:]]
    inp_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def configure_chlorine_control_source(msx_path: Path) -> None:
    """Install the unit pattern whose source level is changed by the controller."""
    lines = msx_path.read_text(encoding="utf-8").splitlines()
    source = f"MASS        {CHLORINE_SOURCE_NODE:<33} Chlorine                          1.000000e+00  {CHLORINE_PATTERN_ID}"
    values = ["1.000000e+00"] * (DURATION_DAYS * HYDRAULIC_STEPS_PER_DAY)
    pattern = [f"{CHLORINE_PATTERN_ID:<33} " + "  ".join(values[index:index + 6]) for index in range(0, len(values), 6)]
    lines = _replace_msx_section(lines, "SOURCES", [source])
    lines = _replace_msx_section(lines, "PATTERNS", pattern)
    msx_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
