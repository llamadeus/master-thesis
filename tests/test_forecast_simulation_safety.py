from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from packages.trainer.forecasting.materialization import (
    ForecastTrajectoryResolver,
    TrajectoryReference,
)
from packages.trainer.forecasting.runner import _SessionForecastBackend
from packages.trainer.simulation import SimulationSample, Trajectory, TrajectoryStore
from packages.trainer.simulation import native_temp


def _sample(time_seconds: int) -> SimulationSample:
    nodes, links = ("1",), ("p",)
    node_values = np.zeros(1, dtype=np.float32)
    return SimulationSample(
        time_seconds, 0, nodes, links,
        node_values, node_values, node_values,
        node_values, node_values, 0.0, 0.0, False, node_values,
        node_values,
    )


class ForecastSimulationSafetyTests(unittest.TestCase):
    def _trajectory(self, times):
        times = np.asarray(times, dtype=np.int64)
        rows = len(times)
        node_values = np.zeros((rows, 1), dtype=np.float32)
        return Trajectory(
            "hanoi", "scenario-26", 0, "chlorine", "arsenic", -1,
            ("1",), ("p",), ("1",), times,
            node_values, node_values, node_values, node_values, node_values,
            np.zeros(rows, dtype=np.float32), np.zeros(rows, dtype=np.float32),
            np.zeros(rows, dtype=bool), node_values,
            hydraulic_outflow=node_values,
        )

    def test_premature_epanet_termination_is_not_returned_as_a_trajectory(self):
        session = Mock(flow_units="LPS")
        session.reset.return_value = _sample(0)
        session.step.return_value = (_sample(1800), True)
        backend = Mock()
        backend.create_session.return_value = session
        dataset = SimpleNamespace(
            network="hanoi", hydraulic_timestep_seconds=1800,
        )
        scenario = SimpleNamespace(
            path=Path("scenario"), scenario_id="scenario-26",
            calendar_start_day=0, chlorine_source_ids=("1",),
            monitored_node_ids=("1",),
        )
        arsenic = SimpleNamespace(events=(), hash="arsenic")
        chlorine = SimpleNamespace(values=(0.0, 1.0, 2.0), hash="chlorine")

        with self.assertRaisesRegex(
            RuntimeError, "scenario-26 after 1/3 control intervals"
        ):
            _SessionForecastBackend(backend, dataset).simulate(
                scenario, arsenic, chlorine
            )
        session.close.assert_called_once_with()

    def test_every_prescribed_interval_is_executed_and_endpoint_is_recorded(self):
        session = Mock(flow_units="LPS")
        session.reset.return_value = _sample(0)
        session.step.side_effect = [(_sample(1800), False), (_sample(3600), True)]
        backend = Mock()
        backend.create_session.return_value = session
        dataset = SimpleNamespace(network="hanoi", hydraulic_timestep_seconds=1800)
        scenario = SimpleNamespace(
            path=Path("scenario"), scenario_id="scenario-26",
            calendar_start_day=0, chlorine_source_ids=("1",),
            monitored_node_ids=("1",),
        )
        trajectory = _SessionForecastBackend(backend, dataset).simulate(
            scenario,
            SimpleNamespace(events=(), hash="arsenic"),
            SimpleNamespace(values=(1.0, 2.0), hash="chlorine"),
        )
        np.testing.assert_array_equal(trajectory.time_seconds, [0, 1800, 3600])
        self.assertEqual(session.step.call_count, 2)
        self.assertEqual(session.step.call_args_list[0].args, (1.0, 0.0))
        self.assertEqual(session.step.call_args_list[1].args, (2.0, 0.0))
        session.close.assert_called_once_with()

    def test_short_trajectory_cache_is_treated_as_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            reference = TrajectoryReference(
                "identity", cache, "scenario-26", "arsenic", "chlorine"
            )
            resolver = ForecastTrajectoryResolver(
                SimpleNamespace(duration_seconds=3600, hydraulic_timestep_seconds=1800),
                SimpleNamespace(), None,
            )
            TrajectoryStore().save(cache, self._trajectory([0]))
            self.assertFalse(resolver._cache_is_valid(reference))
            TrajectoryStore().save(cache, self._trajectory([0, 1800]))
            self.assertFalse(resolver._cache_is_valid(reference))
            TrajectoryStore().save(cache, self._trajectory([0, 1800, 3600]))
            self.assertTrue(resolver._cache_is_valid(reference))

    @unittest.skipUnless(os.name == "nt", "Windows-specific EPANET isolation")
    def test_windows_worker_gets_a_private_native_temp_namespace(self):
        previous_tempdir = tempfile.tempdir
        with tempfile.TemporaryDirectory() as root, patch.object(
            native_temp, "_WORKER_TEMP", None
        ), patch.object(
            native_temp.tempfile, "TemporaryDirectory",
            return_value=SimpleNamespace(name=root),
        ), patch.dict(os.environ, {}, clear=False):
            try:
                path = native_temp.isolate_native_worker_temp()
                self.assertEqual(path, Path(root).resolve())
                self.assertEqual(os.environ["TMP"], str(path))
                self.assertEqual(os.environ["TEMP"], str(path))
                self.assertEqual(os.environ["TMPDIR"], str(path))
                self.assertEqual(tempfile.tempdir, str(path))
                self.assertEqual(native_temp.isolate_native_worker_temp(), path)
            finally:
                tempfile.tempdir = previous_tempdir


if __name__ == "__main__":
    unittest.main()
