from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from packages.trainer.simulation import SimulationSample, Trajectory, TrajectoryStore
from tools.analysis.mpc.validate_mpc_teacher_epanet import (
    HORIZON_STEPS,
    Selection,
    TerminationState,
    WorkerTask,
    _atomic_json,
    _base_case,
    _quality_metrics,
    build_parser,
    cases_to_arrays,
    merge_worker_result,
    replay_to_query,
    run_case_processes,
    run_worker_task,
    select_representative_indices,
    summarize_cases,
    write_plots,
)
from training.rewards.v89 import DEFAULT_PARAMETERS as V89_PARAMETERS


NODE_IDS = ("2", "3")
LINK_IDS = ("L",)
TIMESTEP = 1_800


def _sample(elapsed: int, injection: float, action: float) -> SimulationSample:
    chlorine = np.asarray([0.2 + injection / 1_000_000.0, 0.3], dtype=np.float32)
    node_values = np.zeros(2, dtype=np.float32)
    return SimulationSample(
        time_seconds=elapsed,
        calendar_start_day=0,
        node_ids=NODE_IDS,
        link_ids=LINK_IDS,
        pressure=node_values,
        demand=node_values,
        flow=np.zeros(1, dtype=np.float32),
        chlorine=chlorine,
        arsenic=node_values,
        injection=float(injection),
        action=float(action),
        event_active=False,
        event_strength_by_node=node_values,
    )


class FakeSession:
    def __init__(self, *, mismatch: float = 0.0, early: bool = False, delay: float = 0.0):
        self.elapsed = 0
        self.mismatch = mismatch
        self.early = early
        self.delay = delay
        self.closed = False

    def reset(self, _events):
        self.elapsed = 0
        return _sample(0, 0.0, 0.0)

    def step(self, injection, action):
        if self.delay:
            time.sleep(self.delay)
        self.elapsed += TIMESTEP
        sample = _sample(self.elapsed, float(injection), float(action))
        if self.mismatch:
            sample = SimulationSample(
                **{
                    **sample.__dict__,
                    "chlorine": np.asarray(sample.chlorine) + self.mismatch,
                }
            )
        return sample, bool(self.early and self.elapsed >= TIMESTEP)

    def close(self):
        self.closed = True


class FakeBackend:
    created: list[FakeSession] = []

    def create_session(self, *_args, **_kwargs):
        session = FakeSession()
        self.__class__.created.append(session)
        return session


class MismatchBackend:
    def create_session(self, *_args, **_kwargs):
        return FakeSession(mismatch=1e-3)


class EarlyBackend:
    def create_session(self, *_args, **_kwargs):
        return FakeSession(early=True)


class SlowBackend:
    def create_session(self, *_args, **_kwargs):
        return FakeSession(delay=0.1)


class CrashBackend:
    def create_session(self, *_args, **_kwargs):
        os._exit(7)


def _trajectory(rows: int = 24) -> Trajectory:
    injections = np.linspace(0.0, 180_000.0, rows, dtype=np.float32)
    actions = np.zeros(rows, dtype=np.float32)
    chlorine = np.stack(
        [np.asarray([0.2 + value / 1_000_000.0, 0.3], dtype=np.float32) for value in injections]
    )
    node_values = np.zeros((rows, 2), dtype=np.float32)
    return Trajectory(
        network="synthetic",
        scenario_id="scenario",
        calendar_start_day=0,
        chlorine_profile_hash="chlorine",
        arsenic_schedule_hash="arsenic",
        arsenic_schedule_seed=1,
        node_ids=NODE_IDS,
        link_ids=LINK_IDS,
        source_ids=("1",),
        time_seconds=np.arange(rows, dtype=np.int64) * TIMESTEP,
        pressure=node_values,
        demand=node_values,
        flow=np.zeros((rows, 1), dtype=np.float32),
        chlorine=chlorine,
        arsenic=node_values,
        injection=injections,
        actions=actions,
        event_active=np.zeros(rows, dtype=bool),
        event_strength_by_node=node_values,
    )


def _task(cache: Path, *, dataset_index: int = 0, query_index: int = 3) -> WorkerTask:
    trajectory = TrajectoryStore().load(cache)
    current = float(trajectory.injection[query_index])
    actions = np.linspace(-0.5, 0.5, HORIZON_STEPS, dtype=np.float32)
    injections = np.clip(
        current + np.cumsum(actions) * 1_000.0, 0.0, 500_000.0
    ).astype(np.float32)
    return WorkerTask(
        dataset_index=dataset_index,
        cache_dir=str(cache),
        scenario_path="synthetic",
        calendar_start_day=0,
        chlorine_source_ids=("1",),
        monitored_node_ids=NODE_IDS,
        arsenic_events=(),
        query_index=query_index,
        timestep_seconds=TIMESTEP,
        mpc_actions=actions,
        mpc_injections=injections,
        hold_actions=np.zeros(HORIZON_STEPS, dtype=np.float32),
        hold_injections=np.full(HORIZON_STEPS, current, dtype=np.float32),
        reward_parameters=dict(V89_PARAMETERS),
        discount_factor=0.99,
        replay_tolerance=1e-5,
    )


def _summary_case(index: int, predicted_gain: float, realized_gain: float, *, valid: bool = True):
    case = {
        "dataset_index": index,
        "scenario_id": f"s{index}",
        "trajectory_identity": f"t{index}",
        "query_time_stratum": "query-time-q1",
        "predicted_return_stratum": "predicted-return-q1",
        "action_magnitude_stratum": "low",
        "status": "valid" if valid else "failed-case",
        "valid": valid,
        "completed": valid,
        "cancelled": False,
        "failure_stage": "" if valid else "worker",
        "failure_type": "" if valid else "Failure",
        "failure_message": "" if valid else "failed",
        "predicted_mpc_return": predicted_gain + 10.0,
        "predicted_hold_return": 10.0,
        "predicted_gain": predicted_gain,
        "realized_mpc_return": realized_gain + 20.0,
        "realized_hold_return": 20.0,
        "realized_gain": realized_gain,
    }
    for metric in ("violation_below_0_2", "violation_below_0_25", "target_mae"):
        case[f"predicted_mpc_{metric}"] = 0.1
        case[f"predicted_hold_{metric}"] = 0.2
        case[f"realized_mpc_{metric}"] = 0.1
        case[f"realized_hold_{metric}"] = 0.2
    for metric in ("injection_sum", "injection_mean", "normalized_injection_mean", "action_smoothness", "injection_smoothness"):
        case[f"mpc_{metric}"] = 2.0
        case[f"hold_{metric}"] = 1.0
    return case


class MPCValidationTests(unittest.TestCase):
    def test_representative_selection_covers_scenarios_and_pairs_signs(self):
        scenarios = np.repeat(np.asarray([f"s{i}" for i in range(4)]), 12)
        query = np.tile(np.arange(12), 4)
        returns = np.linspace(-100.0, -1.0, len(scenarios))
        pattern = np.asarray([-1.0, -0.95, -0.8, -0.4, -0.2, -0.1, 0.1, 0.3, 0.6, 0.85, 0.95, 1.0])
        actions = np.tile(pattern, 4)
        first = select_representative_indices(
            scenarios, query, returns, actions, sample_count=8, seed=17
        )
        second = select_representative_indices(
            scenarios, query, returns, actions, sample_count=8, seed=17
        )
        self.assertEqual(first, second)
        indexes = [item.dataset_index for item in first]
        self.assertEqual(set(scenarios[indexes]), set(np.unique(scenarios)))
        for scenario in np.unique(scenarios):
            selected_actions = actions[[index for index in indexes if scenarios[index] == scenario]]
            self.assertEqual(set(np.sign(selected_actions)), {-1.0, 1.0})

    def test_replay_uses_rows_one_through_query(self):
        trajectory = _trajectory()
        session = FakeSession()
        sample, maximum, mean = replay_to_query(session, trajectory, (), 5)
        self.assertEqual(sample.time_seconds, 5 * TIMESTEP)
        self.assertEqual(sample.injection, float(trajectory.injection[5]))
        self.assertAlmostEqual(maximum, 0.0)
        self.assertAlmostEqual(mean, 0.0)

    def test_worker_uses_two_fresh_sessions_and_cloned_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            TrajectoryStore().save(cache, _trajectory())
            FakeBackend.created.clear()
            result = run_worker_task(_task(cache), FakeBackend)
        self.assertTrue(result["mpc"]["complete"])
        self.assertTrue(result["hold"]["complete"])
        self.assertEqual(len(FakeBackend.created), 2)
        self.assertIsNot(FakeBackend.created[0], FakeBackend.created[1])
        self.assertTrue(all(session.closed for session in FakeBackend.created))
        self.assertEqual(result["mpc"]["rewards"].shape, (HORIZON_STEPS,))
        self.assertEqual(result["hold"]["rewards"].shape, (HORIZON_STEPS,))

    def test_mismatch_and_early_termination_are_explicit_branch_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            TrajectoryStore().save(cache, _trajectory())
            mismatch = run_worker_task(_task(cache), MismatchBackend)
            early = run_worker_task(_task(cache), EarlyBackend)
        self.assertTrue(mismatch["mpc"]["complete"])
        self.assertFalse(mismatch["mpc"]["replay_valid"])
        self.assertGreater(mismatch["mpc"]["replay_max_abs_error"], 1e-5)
        self.assertFalse(early["mpc"]["complete"])
        self.assertIn("terminated", early["mpc"]["failure_message"])

    def test_spawn_scheduler_is_ordered_and_captures_native_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            TrajectoryStore().save(cache, _trajectory())
            tasks = (_task(cache, dataset_index=2), _task(cache, dataset_index=1))
            results, interrupted = run_case_processes(
                tasks, workers=2, shutdown_timeout=1.0, backend_factory=FakeBackend
            )
            crashed, _ = run_case_processes(
                (_task(cache, dataset_index=4),),
                workers=1,
                shutdown_timeout=1.0,
                backend_factory=CrashBackend,
            )
        self.assertFalse(interrupted)
        self.assertEqual([item["dataset_index"] for item in results], [1, 2])
        self.assertTrue(all("mpc" in item for item in results))
        self.assertTrue(crashed[0]["worker_failure"])
        self.assertIn("exit", crashed[0]["failure_message"])

    def test_scheduler_graceful_and_forced_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            TrajectoryStore().save(cache, _trajectory())
            state = TerminationState()
            timer = threading.Timer(0.2, state.request)
            timer.start()
            results, interrupted = run_case_processes(
                (_task(cache, dataset_index=1), _task(cache, dataset_index=2)),
                workers=1,
                shutdown_timeout=0.0,
                termination=state,
                backend_factory=SlowBackend,
            )
            timer.join()
            forced = TerminationState()
            forced.request()
            forced.request()
            forced_results, forced_interrupted = run_case_processes(
                (_task(cache, dataset_index=3),),
                workers=1,
                shutdown_timeout=30.0,
                termination=forced,
                backend_factory=SlowBackend,
            )
        self.assertTrue(interrupted)
        self.assertTrue(forced_interrupted)
        self.assertTrue(all(item["cancelled"] for item in results))
        self.assertTrue(forced_results[0]["cancelled"])

    def test_summary_metrics_bootstrap_and_failures_are_deterministic(self):
        cases = [
            _summary_case(0, 1.0, 2.0),
            _summary_case(1, -1.0, -2.0),
            _summary_case(2, 0.0, 0.0),
            _summary_case(3, 5.0, 1.0, valid=False),
        ]
        first = summarize_cases(cases, seed=9, bootstrap_samples=200, tie_tolerance=1e-6, interrupted=False)
        second = summarize_cases(cases, seed=9, bootstrap_samples=200, tie_tolerance=1e-6, interrupted=False)
        self.assertEqual(first, second)
        self.assertEqual(first["valid_count"], 3)
        self.assertEqual(first["failed_count"], 1)
        self.assertAlmostEqual(first["realized_outcome"]["win_fraction"], 1 / 3)
        self.assertAlmostEqual(first["realized_outcome"]["tie_fraction"], 1 / 3)
        self.assertAlmostEqual(first["realized_outcome"]["loss_fraction"], 1 / 3)
        self.assertAlmostEqual(first["predicted_vs_realized"]["gain"]["mae"], 2 / 3)
        self.assertEqual(len(first["failures"]), 1)

    def test_outputs_are_pickle_free_strict_json_and_plot_ready(self):
        arrays = {
            "scenario_id": np.asarray(["s"]),
            "trajectory_identity": np.asarray(["t"]),
            "chlorine_profile_hash": np.asarray(["c"]),
            "arsenic_schedule_hash": np.asarray(["a"]),
            "calendar_start_day": np.asarray([0]),
            "query_index": np.asarray([3]),
            "time_seconds": np.asarray([3 * TIMESTEP]),
            "current_injection": np.asarray([100_000.0]),
            "first_action": np.asarray([0.5]),
            "best_raw_actions": np.zeros((1, HORIZON_STEPS), dtype=np.float32),
            "best_injections": np.full((1, HORIZON_STEPS), 100_000.0, dtype=np.float32),
            "predicted_chlorine": np.full((1, HORIZON_STEPS, 2), 0.3, dtype=np.float32),
            "predicted_return": np.asarray([1.0]),
        }
        case = _base_case(
            arrays,
            Selection(0, "query-time-q1", "predicted-return-q1", "medium", "positive"),
            2,
        )
        case["status"] = "valid"
        case["valid"] = case["completed"] = case["prediction_complete"] = case["replay_valid"] = True
        case["predicted_hold_return"] = case["realized_hold_return"] = 0.0
        case["predicted_gain"] = case["realized_gain"] = 1.0
        case["realized_mpc_return"] = 1.0
        case["realized_mpc_chlorine"] = np.full((HORIZON_STEPS, 2), 0.3, dtype=np.float32)
        case["realized_hold_chlorine"] = np.full((HORIZON_STEPS, 2), 0.3, dtype=np.float32)
        for arm in ("mpc", "hold"):
            for metric, value in _quality_metrics(case[f"realized_{arm}_chlorine"]).items():
                case[f"realized_{arm}_{metric}"] = value
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_arrays = cases_to_arrays([case])
            np.savez_compressed(root / "cases.npz", **output_arrays)
            with np.load(root / "cases.npz", allow_pickle=False) as stored:
                self.assertIn("realized_gain", stored.files)
            _atomic_json(root / "summary.json", {"finite": 1.0, "missing": float("nan")})
            parsed = json.loads((root / "summary.json").read_text())
            self.assertIsNone(parsed["missing"])
            write_plots(root, [case])
            self.assertTrue((root / "predicted_vs_realized_gain.png").is_file())
            self.assertTrue((root / "realized_gain_distribution.png").is_file())

    def test_cli_defaults_and_worker_validation(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.sample_count, 160)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.shutdown_timeout, 30.0)
        self.assertIsNone(args.forecast_bundle)
        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            run_case_processes((), workers=0, shutdown_timeout=1.0)


if __name__ == "__main__":
    unittest.main()
