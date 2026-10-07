"""Gymnasium control environment over the trainer2 simulation contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..data import ScenarioEntry
from ..observations import ControlObservationBuilder
from ..schedules import ArsenicSchedule, ArsenicScheduleStream
from ..simulation import SimulationBackend, SimulationSample, Trajectory, TrajectoryStore
from .rewards import Reward, RewardInput


class ControlEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, scenarios: tuple[ScenarioEntry, ...], backend: SimulationBackend, schedule_stream: ArsenicScheduleStream | None,
                 observation: ControlObservationBuilder, reward: Reward,
                 timestep_seconds: int, episode_steps: int, max_injection: float, max_delta: float, initial_injection: float,
                 network: str, trajectory_root: Path | None = None, fixed_schedule_loader=None):
        if not scenarios: raise ValueError("control environment requires scenarios")
        self.scenarios=scenarios; self.backend=backend; self.stream=schedule_stream; self.builder=observation
        self.reward=reward; self.timestep=timestep_seconds
        self.episode_steps=episode_steps; self.max_injection=max_injection; self.max_delta=max_delta
        self.initial_injection=initial_injection
        self.trajectory_root=trajectory_root; self.fixed_schedule_loader=fixed_schedule_loader; self.index=0; self.episode=0
        self.network=network; self.current_scenario=None; self.schedule_hash=""
        self.action_space=spaces.Box(-1,1,(1,),dtype=np.float32); self.observation_space=spaces.Box(-np.inf,np.inf,observation.shape,dtype=np.float32)
        self.session=None; self.sample=None; self.injection=0.0; self.samples=[]; self.step_count=0
    def runtime_state(self):
        return {"schedule":self.stream.state_dict() if self.stream else None,"scenario_index":self.index,"episode":self.episode,
                "provider":self.builder.provider.state_dict() if self.builder.provider else None,
                "observation":self.builder.state_dict(),
                "reward":self.reward.state_dict()}
    def load_runtime_state(self, state):
        if self.stream and state.get("schedule") is not None:self.stream.load_state_dict(state["schedule"])
        self.index=int(state.get("scenario_index",self.index));self.episode=int(state.get("episode",self.episode))
        if self.builder.provider and state.get("provider") is not None:self.builder.provider.load_state_dict(state["provider"])
        self.builder.load_state_dict(state["observation"])
        self.reward.load_state_dict(state["reward"])
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.session is not None: self.session.close()
        scenario=self.scenarios[self.index%len(self.scenarios)]; self.index+=1; self.current_scenario=scenario
        schedule = self.fixed_schedule_loader(scenario, self.episode) if self.fixed_schedule_loader else (
            self.stream.next_schedule(scenario.eligible_arsenic_nodes,scenario_id=scenario.scenario_id) if self.stream else ArsenicSchedule(())
        )
        self.schedule_hash=schedule.hash
        self.session=self.backend.create_session(str(scenario.path), hydraulic_timestep_seconds=self.timestep,
                                                 calendar_start_day=scenario.calendar_start_day,
                                                 chlorine_source_ids=scenario.chlorine_source_ids,
                                                 monitored_node_ids=scenario.monitored_node_ids)
        try:
            self.sample=self.session.reset(schedule.events); self._validate_sample_calendar(self.sample); self.injection=self.initial_injection
            self.sample=replace(self.sample,injection=self.injection); self.samples=[self.sample]; self.step_count=0
            self.builder.reset(); self.reward.reset()
            pre_roll_steps=self.builder.required_history_samples-1
            if pre_roll_steps>=self.episode_steps:
                raise ValueError(
                    "provider history pre-roll leaves no controllable episode steps"
                )
            for _ in range(pre_roll_steps):
                self.builder.prime(self.sample)
                self.sample,backend_done=self.session.step(self.injection,0.0)
                self._validate_sample_calendar(self.sample);self.samples.append(self.sample);self.step_count+=1
                self.reward.compute(self._reward_input(self.sample,0.0))
                if backend_done:
                    raise RuntimeError(
                        "simulation ended before provider history pre-roll completed"
                    )
            observation=self.builder.build(self.sample)
        except BaseException:
            self.close()
            raise
        self.episode+=1
        return observation, {"scenario_id": scenario.scenario_id, "arsenic_schedule_hash": schedule.hash,
            "calendar_start_day":scenario.calendar_start_day,"pre_roll_steps":pre_roll_steps,
            "time_seconds":self.sample.time_seconds}
    def step(self, action):
        raw=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1)); self.injection=float(np.clip(self.injection+raw*self.max_delta,0,self.max_injection))
        self.sample, backend_done=self.session.step(self.injection, raw);self._validate_sample_calendar(self.sample); self.samples.append(self.sample); self.step_count+=1
        reward, info=self.reward.compute(self._reward_input(self.sample,raw))
        done=backend_done or self.step_count>=self.episode_steps
        info.update({"time_seconds":self.sample.time_seconds,"node_ids":list(self.sample.node_ids),
                     "chlorine":np.asarray(self.sample.chlorine).tolist(),"arsenic":np.asarray(self.sample.arsenic).tolist(),
                     "injection":self.injection,"action":raw,"event_active":self.sample.event_active})
        if done: self._record_trajectory()
        return self.builder.build(self.sample), reward, done, False, info
    def close(self):
        if self.session is not None: self.session.close(); self.session=None
    def _validate_sample_calendar(self,sample):
        if self.current_scenario is None or sample.calendar_start_day!=self.current_scenario.calendar_start_day:
            raise ValueError("simulation sample calendar disagrees with scenario manifest")
    def _reward_input(self,sample,raw_action):
        return RewardInput(
            nodes_quality=np.asarray(sample.chlorine), contaminant_values=np.asarray(sample.arsenic),
            pressure=np.asarray(sample.pressure), demand=np.asarray(sample.demand), flow=np.asarray(sample.flow),
            current_injection=self.injection, raw_action=raw_action, step=self.step_count, current_time=sample.time_seconds,
            node_ids=sample.node_ids, link_ids=sample.link_ids, event_active=sample.event_active,
            event_strength_by_node=np.asarray(sample.event_strength_by_node) if sample.event_strength_by_node is not None else np.zeros(len(sample.node_ids),dtype=np.float32),
        )
    def _record_trajectory(self):
        if self.trajectory_root is None or self.current_scenario is None: return
        samples=self.samples; node_ids=samples[0].node_ids; link_ids=samples[0].link_ids
        trajectory=Trajectory(self.network,self.current_scenario.scenario_id,
            self.current_scenario.calendar_start_day,
            "policy",self.schedule_hash,self.stream.spec.seed if self.stream else -1,node_ids,link_ids,
            self.current_scenario.chlorine_source_ids,
            np.asarray([s.time_seconds for s in samples]),np.stack([s.pressure for s in samples]),np.stack([s.demand for s in samples]),
            np.stack([s.flow for s in samples]),np.stack([s.chlorine for s in samples]),np.stack([s.arsenic for s in samples]),
            np.asarray([s.injection for s in samples]),np.asarray([s.action for s in samples]),np.asarray([s.event_active for s in samples],dtype=bool),
            np.stack([s.event_strength_by_node for s in samples]))
        TrajectoryStore().save(self.trajectory_root/f"episode-{self.episode-1:08d}",trajectory)
