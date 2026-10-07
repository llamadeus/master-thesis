"""EPyT-Flow implementation of the trainer2 simulation contract."""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from epyt_flow.gym import ScenarioControlEnv
from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScenarioSimulator
from epyt_flow.simulation.events import SpeciesInjectionEvent

from ..schedules import ArsenicEvent
from .core import SimulationSample
from .hydraulic_cache import HydraulicCache


class EpytFlowBackend:
    def __init__(self, *, hydraulic_cache_root: str | Path | None = None):
        self.hydraulic_cache_root = hydraulic_cache_root

    def create_session(self, scenario_path: str, *, hydraulic_timestep_seconds: int,
                       calendar_start_day: int,
                       chlorine_source_ids: tuple[str, ...], monitored_node_ids: tuple[str, ...]):
        return EpytFlowSession(Path(scenario_path), hydraulic_timestep_seconds, calendar_start_day,
                               chlorine_source_ids, monitored_node_ids, hydraulic_cache_root=self.hydraulic_cache_root)

    def prepare_hydraulics(self, scenario_path: Path, *, cache_root=None):
        cache = HydraulicCache(cache_root if cache_root is not None else self.hydraulic_cache_root)
        scenario_path = Path(scenario_path).resolve()

        @contextmanager
        def simulator_factory():
            with _cwd(scenario_path), _silence_native_stdout():
                config = ScenarioConfig.load_from_file(str(scenario_path / "scenario.epytflow_scenario_config"))
                with ScenarioSimulator(scenario_config=config) as simulator:
                    yield simulator

        cache.resolve(scenario_path, simulator_factory=simulator_factory)
        return cache.reference(scenario_path)

    @staticmethod
    def load_hydraulic_outflow(reference, node_ids):
        from .hydraulics import HydraulicOutflowProfile, FLOW_TO_LITERS_PER_MINUTE
        scada = reference.load()
        if scada is None:
            raise ValueError(f"invalid hydraulic cache in {reference.cache_dir}")
        topology = scada.network_topo
        endpoints = dict(topology.get_all_links())
        sensors = scada.sensor_config
        links = tuple(sensors.flow_sensors)
        required = {link for link, ends in endpoints.items() if set(ends).intersection(node_ids)}
        if required - set(links):
            raise ValueError("hydraulic outflow requires sensors for every incident link")
        # Hydraulic SCADA retains the scenario's quality sensor declarations,
        # but contains no MSX concentrations. Select only hydraulic readings in
        # this local copy so EPyT does not require absent quality arrays.
        sensors.quality_node_sensors = []
        sensors.quality_link_sensors = []
        sensors.bulk_species_node_sensors = {}
        sensors.bulk_species_link_sensors = {}
        sensors.surface_species_sensors = {}
        scada.sensor_config = sensors
        units = ("CFS", "GPM", "MGD", "IMGD", "AFD", "LPS", "LPM", "MLD", "CMH", "CMD", "CMS")
        factor = FLOW_TO_LITERS_PER_MINUTE[units[topology.flow_units]]
        outflow = _hydraulic_outflow(
            node_ids, links, endpoints,
            scada.get_data_flows(sensor_locations=list(links)),
            scada.get_data_demands(sensor_locations=list(node_ids)),
        )
        return HydraulicOutflowProfile(
            reference.identity, reference.contract["scenario_id"],
            np.asarray(scada.sensor_readings_time, dtype=np.int64), tuple(node_ids),
            (outflow * factor).astype(np.float32),
        )


class EpytFlowSession(ScenarioControlEnv):
    def __init__(self, scenario_dir: Path, timestep: int, calendar_start_day: int,
                 chlorine_source_ids: tuple[str, ...], monitored_node_ids: tuple[str, ...],
                 *, hydraulic_cache_root: str | Path | None = None):
        self.scenario_dir=scenario_dir.resolve(); self.timestep=timestep; self.elapsed=0; self.events=()
        self._hydraulic_cache = HydraulicCache(hydraulic_cache_root)
        self.calendar_start_day=calendar_start_day
        config_path=self.scenario_dir/"scenario.epytflow_scenario_config"
        with _cwd(self.scenario_dir): config=ScenarioConfig.load_from_file(str(config_path))
        self.config=config
        if len(chlorine_source_ids)!=1:raise ValueError("trainer2 control currently requires exactly one chlorine source")
        self.chlorine_source_id=chlorine_source_ids[0]
        chlorine_sensors=tuple(str(item) for item in config.sensor_config.bulk_species_node_sensors.get("Chlorine", []))
        arsenic_sensors=tuple(str(item) for item in config.sensor_config.bulk_species_node_sensors.get("AsIII", []))
        self.node_ids=tuple(monitored_node_ids)
        if not self.node_ids or not set(self.node_ids).issubset(chlorine_sensors) or not set(self.node_ids).issubset(arsenic_sensors):
            raise ValueError("monitored nodes must have both chlorine and arsenic sensors")
        self.link_ids=tuple(str(item) for item in config.sensor_config.flow_sensors)
        self.link_endpoints,self.flow_units=_read_network_metadata(self.scenario_dir/"scenario.inp")
        missing_links=set(self.link_ids)-set(self.link_endpoints)
        if missing_links:raise ValueError(f"flow sensors reference links absent from scenario.inp: {sorted(missing_links)}")
        required_links={link for link,(source,target) in self.link_endpoints.items()
                        if source in self.node_ids or target in self.node_ids}
        unobserved=required_links-set(self.link_ids)
        if unobserved:raise ValueError(f"hydraulic outflow requires sensors for incident links: {sorted(unobserved)}")
        with _cwd(self.scenario_dir): super().__init__(scenario_config=config, autoreset=False)
    def reset(self, arsenic_events):
        self.events=tuple(arsenic_events); self.elapsed=0; self._install_events()
        # EPANET-MSX writes its input summary through C stdio when EPyT-Flow
        # opens a project without a report file. Redirecting Python's
        # sys.stdout does not catch that output, so silence the native file
        # descriptor for the duration of the reset instead.
        try:
            with _cwd(self.scenario_dir), _silence_native_stdout(): result=self._reset_simulation()
            scada,terminated=_simulation_result(result)
            if scada is None or terminated:
                raise RuntimeError("EPyT-Flow terminated while resetting the control simulation")
            return self._sample(scada,0.0,0.0)
        except BaseException:
            self.close()
            raise
    def _reset_simulation(self):
        # The shared cache owns hydraulics; each session owns its quality state.
        self.close()
        self._scenario_sim = ScenarioSimulator(scenario_config=self._scenario_config)
        if self._scenario_sim.f_msx_in is not None:
            hyd_path, self._hydraulic_scada_data = self._hydraulic_cache.resolve(
                self.scenario_dir, self._scenario_sim,
                reapply_uncertainties=self.reapply_uncertainties_at_reset,
            )
            self._sim_generator = self._scenario_sim.run_advanced_quality_simulation_as_generator(
                str(hyd_path), support_abort=True,
            )
        else:
            self._sim_generator = self._scenario_sim.run_hydraulic_simulation_as_generator(
                support_abort=True,
                reapply_uncertainties=self.reapply_uncertainties_at_reset,
            )
        return self._next_sim_itr()
    def step(self, injection, action):
        with _cwd(self.scenario_dir):
            self.set_node_species_source_value(species_id="Chlorine", node_id=self.chlorine_source_id, source_type=EpanetConstants.EN_MASS,
                                               pattern_id=f"chlorine-injection-at-node-{self.chlorine_source_id}", source_strength=float(injection))
            result=self._next_sim_itr()
        scada,terminated=_simulation_result(result)
        if scada is None:
            raise RuntimeError("EPyT-Flow ended without returning a final simulation sample")
        self.elapsed+=self.timestep
        return self._sample(scada,injection,action),terminated
    def close(self):
        with _cwd(self.scenario_dir), _silence_native_stdout(): super().close()
        self._sim_generator = None
        self._hydraulic_scada_data = None
    def _install_events(self):
        retained=[event for event in self.config.system_events if not (event.__class__.__name__=="SpeciesInjectionEvent" and getattr(event,"species_id",None)=="AsIII")]
        for event in self.events:
            retained.append(SpeciesInjectionEvent(species_id=event.species,node_id=event.node_id,
                profile=np.asarray([event.strength],dtype=np.float64),source_type=EpanetConstants.EN_MASS,
                start_time=event.start_time_seconds,end_time=event.end_time_seconds))
        setattr(self.config,"_ScenarioConfig__system_events",retained)
    def _sample(self,scada,injection,action):
        chlorine=_species(scada,"Chlorine",self.node_ids); arsenic=_species(scada,"AsIII",self.node_ids)
        pressure=_reading(scada,"get_data_pressures",self.node_ids); demand=_reading(scada,"get_data_demands",self.node_ids)
        flow=_reading(scada,"get_data_flows",self.link_ids)
        hydraulic_outflow=_hydraulic_outflow(
            self.node_ids,self.link_ids,self.link_endpoints,flow,demand
        ).astype(np.float32)
        strengths=np.zeros(len(self.node_ids),dtype=np.float32); index={node:i for i,node in enumerate(self.node_ids)}
        for event in self.events:
            if event.start_time_seconds<=self.elapsed<event.end_time_seconds and event.node_id in index: strengths[index[event.node_id]]+=event.strength
        return SimulationSample(self.elapsed,self.calendar_start_day,self.node_ids,self.link_ids,
            pressure,demand,flow,chlorine,arsenic,float(injection),float(action),bool(np.any(strengths)),strengths,
            hydraulic_outflow)


def _read_network_metadata(inp_path: Path):
    section=None; endpoints={}; flow_units=None
    for raw_line in inp_path.read_text(encoding="utf-8").splitlines():
        line=raw_line.split(";",1)[0].strip()
        if not line:continue
        if line.startswith("[") and line.endswith("]"):
            section=line.upper();continue
        parts=line.split()
        if section=="[OPTIONS]" and len(parts)>=2 and parts[0].upper()=="UNITS":
            flow_units=parts[1].upper()
        elif section in ("[PIPES]","[PUMPS]","[VALVES]") and len(parts)>=3:
            endpoints[str(parts[0])]=(str(parts[1]),str(parts[2]))
    if flow_units is None:raise ValueError(f"scenario does not declare [OPTIONS] UNITS: {inp_path}")
    return endpoints,flow_units


def _hydraulic_outflow(node_ids,link_ids,endpoints,flow,demand):
    """Mixed-node outflow: positive withdrawals plus flow leaving on links."""
    result=np.clip(np.asarray(demand,dtype=np.float64),0.0,None)
    node_index={node:index for index,node in enumerate(node_ids)}
    flows=np.asarray(flow,dtype=np.float64)
    for column,link_id in enumerate(link_ids):
        source,target=endpoints[link_id]
        if source in node_index:
            result[...,node_index[source]]+=np.maximum(flows[...,column],0.0)
        if target in node_index:
            result[...,node_index[target]]+=np.maximum(-flows[...,column],0.0)
    return result


def _species(scada,species,nodes):
    values=np.asarray(scada.get_data_bulk_species_node_concentration({species:list(nodes)}),dtype=np.float32).reshape(-1)
    if len(values)!=len(nodes):raise ValueError(f"EPyT-Flow returned {len(values)} {species} readings for {len(nodes)} nodes")
    return np.clip(values,0,None)
def _reading(scada,name,locations):
    if not locations:return np.zeros(0,dtype=np.float32)
    values=np.asarray(getattr(scada,name)(sensor_locations=list(locations)),dtype=np.float32).reshape(-1)
    if len(values)!=len(locations):raise ValueError(f"EPyT-Flow returned {len(values)} readings for {len(locations)} {name} locations")
    return values
def _simulation_result(result):
    if isinstance(result,tuple) and len(result)==2:return result[0],bool(result[1])
    return result,False


@contextmanager
def _silence_native_stdout():
    """Suppress output emitted directly through C's stdout file descriptor."""
    try:sys.stdout.flush()
    except (AttributeError, OSError):pass
    _flush_c_streams()
    saved_stdout=os.dup(1)
    try:
        with open(os.devnull,"w",encoding="utf-8") as sink:
            os.dup2(sink.fileno(),1)
            try:yield
            finally:
                try:sys.stdout.flush()
                except (AttributeError,OSError):pass
                _flush_c_streams()
    finally:
        os.dup2(saved_stdout,1);os.close(saved_stdout)


def _flush_c_streams():
    try:
        library=ctypes.CDLL("ucrtbase") if os.name=="nt" else ctypes.CDLL(None)
        library.fflush(None)
    except (AttributeError,OSError,TypeError):pass


@contextmanager
def _cwd(path):
    old=os.getcwd(); os.chdir(path)
    try:yield
    finally:os.chdir(old)
