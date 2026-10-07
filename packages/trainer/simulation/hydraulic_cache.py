"""Persistent EPANET hydraulics shared by independent simulation sessions."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import sys
from tempfile import TemporaryDirectory
import zlib

from epyt_flow.simulation import ScadaData, ScenarioSimulator
from filelock import FileLock

from ..scenarios import SCENARIO_FILES, file_sha256, scenario_file_sha256

DEFAULT_HYDRAULIC_CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "hydraulics"
HYDRAULIC_CACHE_VERSION = 1
EPANET_HYDRAULICS_MAGIC = 516114521
EPANET_HYDRAULICS_VERSION = 201
EPANET_HYDRAULICS_END_MARKER = 0x1A
_HYDRAULICS_HEADER = struct.Struct("=8i")


class HydraulicCache:
    """Store one immutable hydraulic realization and its readings per hash.

    Runtime species injections are excluded from the identity: they only affect
    water quality. The saved scenario contains the hydraulic events and seeds.
    All readers and writers take the same OS-backed lock before opening files.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root if root is not None else DEFAULT_HYDRAULIC_CACHE_ROOT).resolve()

    def reference(self, scenario_dir: Path) -> HydraulicCacheReference:
        contract = source_contract(scenario_dir)
        identity = hashlib.sha256(_json_bytes(contract)).hexdigest()
        target = self.root / contract["network"] / contract["scenario_id"] / identity
        return HydraulicCacheReference(identity, target, contract)

    def resolve(self, scenario_dir: Path, simulator: ScenarioSimulator | None = None, *,
                simulator_factory=None,
                reapply_uncertainties: bool = False) -> tuple[Path, ScadaData]:
        reference = self.reference(scenario_dir)
        identity, target, contract = reference.identity, reference.cache_dir, reference.contract
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep lock inodes stable, outside the published entry directory.
        with reference.lock():
            cached = _load(target, identity, contract)
            if cached is not None:
                # EPyT includes the absolute INP path in topology equality.
                # Identical scenarios in another folder share this cache, while
                # their physical topology and readings must remain unchanged.
                if simulator is not None and cached.network_topo.name != simulator.f_inp_in:
                    attributes = cached.get_attributes()
                    attributes["network_topo"].name = simulator.f_inp_in
                    cached = ScadaData(**attributes)
                return target / "hydraulics.hyd", cached
            with ExitStack() as stack:
                if simulator is None:
                    if simulator_factory is None:
                        raise ValueError("a simulator factory is required for cache misses")
                    simulator = stack.enter_context(simulator_factory())
                raw = stack.enter_context(TemporaryDirectory(prefix=f".{identity}-", dir=target.parent))
                staging = Path(raw)
                hyd_path = staging / "hydraulics.hyd"
                scada = simulator.run_hydraulic_simulation(
                    hyd_export=str(hyd_path), reapply_uncertainties=reapply_uncertainties,
                )
                if not isinstance(scada, ScadaData):
                    raise TypeError("hydraulic simulation must return ScadaData")
                if not hyd_path.is_file() or hyd_path.stat().st_size == 0:
                    raise RuntimeError("hydraulic simulation did not export a .hyd file")
                if not _valid_hydraulics_file(hyd_path, contract):
                    raise RuntimeError(
                        "hydraulic simulation exported an invalid EPANET .hyd file"
                    )
                payload = scada.dump()
                metadata = {
                    "schema_version": HYDRAULIC_CACHE_VERSION,
                    "identity": identity,
                    "source": contract,
                    "hydraulics_sha256": file_sha256(hyd_path),
                    "scada_encoding": "epyt-flow-msgpack+zlib+base64",
                    "scada_sha256": hashlib.sha256(payload).hexdigest(),
                    "scada": base64.b64encode(zlib.compress(payload)).decode("ascii"),
                }
                (staging / "metadata.json").write_bytes(_json_bytes(metadata))
                target.mkdir(exist_ok=True)
                os.replace(hyd_path, target / "hydraulics.hyd")
                # Publish metadata last. Interrupted publication is a cache miss.
                os.replace(staging / "metadata.json", target / "metadata.json")
                return target / "hydraulics.hyd", scada


@dataclass(frozen=True)
class HydraulicCacheReference:
    identity: str
    cache_dir: Path
    contract: dict

    def lock(self):
        return FileLock(str(self.cache_dir.parent / f".{self.identity}.lock"), timeout=1800)

    def load(self) -> ScadaData | None:
        if not self.cache_dir.parent.is_dir():
            return None
        with self.lock():
            return _load(self.cache_dir, self.identity, self.contract)


def source_contract(scenario_dir: Path) -> dict:
    """Include seeds explicitly and fingerprint all saved simulator inputs."""
    scenario_dir = Path(scenario_dir)
    manifest = json.loads((scenario_dir / "manifest.json").read_text(encoding="utf-8"))
    seeds = manifest.get("seeds")
    if not isinstance(seeds, dict) or any(
        isinstance(seeds.get(key), bool) or not isinstance(seeds.get(key), int)
        for key in ("master_seed", "demand_seed", "uncertainty_seed")
    ):
        raise ValueError("hydraulic cache requires the scenario's master, demand and uncertainty seeds")
    for name in (manifest["network"], manifest["id"]):
        if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
            raise ValueError("hydraulic network and scenario IDs must be path components")
    return {
        "schema_version": HYDRAULIC_CACHE_VERSION,
        "scenario_id": manifest["id"],
        "network": manifest["network"],
        "calendar": manifest["calendar"],
        "hydraulic_timestep_seconds": manifest["hydraulic_timestep_seconds"],
        "duration_seconds": manifest["duration_seconds"],
        "seeds": seeds,
        "files": {
            **{name: scenario_file_sha256(scenario_dir / name) for name in SCENARIO_FILES},
            "manifest.json": file_sha256(scenario_dir / "manifest.json"),
        },
        "epyt_flow_version": importlib.metadata.version("epyt-flow"),
        "epanet_plus_version": importlib.metadata.version("epanet-plus"),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "byteorder": sys.byteorder},
    }


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _load(target: Path, identity: str, contract: dict) -> ScadaData | None:
    try:
        hyd_path = target / "hydraulics.hyd"
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict)
                or metadata.get("schema_version") != HYDRAULIC_CACHE_VERSION
                or metadata.get("identity") != identity
                or metadata.get("source") != contract
                or metadata.get("scada_encoding") != "epyt-flow-msgpack+zlib+base64"
                or metadata.get("hydraulics_sha256") != file_sha256(hyd_path)
                or not _valid_hydraulics_file(hyd_path, contract)):
            return None
        payload = zlib.decompress(base64.b64decode(metadata["scada"], validate=True))
        if hashlib.sha256(payload).hexdigest() != metadata.get("scada_sha256"):
            return None
        scada = ScadaData.load(payload)
        return scada if isinstance(scada, ScadaData) else None
    except (OSError, ValueError, TypeError, KeyError, zlib.error):
        return None


def _valid_hydraulics_file(path: Path, contract: dict) -> bool:
    """Reject non-hydraulic or incomplete binaries before cache publication."""
    try:
        size = path.stat().st_size
        if size <= _HYDRAULICS_HEADER.size:
            return False
        with path.open("rb") as stream:
            raw = stream.read(_HYDRAULICS_HEADER.size)
            if len(raw) != _HYDRAULICS_HEADER.size:
                return False
            magic, version, nodes, links, tanks, pumps, valves, duration = (
                _HYDRAULICS_HEADER.unpack(raw)
            )
            if (magic != EPANET_HYDRAULICS_MAGIC
                    or version != EPANET_HYDRAULICS_VERSION
                    or nodes < 1 or links < 0
                    or not 0 <= tanks <= nodes
                    or not 0 <= pumps <= links
                    or not 0 <= valves <= links
                    or duration != contract["duration_seconds"]):
                return False
            # Each hydraulic period contains its timestamp, two node arrays,
            # three link arrays and the following hydraulic time step.
            record_size = 8 + 8 * nodes + 12 * links
            if (size - _HYDRAULICS_HEADER.size - 1) % record_size:
                return False
            stream.seek(-1, os.SEEK_END)
            return stream.read(1) == bytes((EPANET_HYDRAULICS_END_MARKER,))
    except (OSError, KeyError, TypeError, ValueError, struct.error):
        return False
