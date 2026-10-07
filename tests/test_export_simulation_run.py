from types import SimpleNamespace
import unittest

import numpy as np

from tools.analysis.dataset.export_simulation_run import validate_trajectory


def fixture():
    dataset = SimpleNamespace(duration_seconds=7200, hydraulic_timestep_seconds=1800)
    event = SimpleNamespace(node_id="5", start_time_seconds=1800,
                            end_time_seconds=5400, strength=1200.0)
    schedule = SimpleNamespace(events=(event,))
    chlorine = SimpleNamespace(values=(10.0, 20.0, 30.0, 40.0))
    trajectory = SimpleNamespace(
        time_seconds=np.arange(0, 7200 + 1800, 1800), node_ids=("5", "25"), link_ids=("1",),
        flow_units="CMH", injection=np.array((0.0, *chlorine.values)), actions=np.zeros(5),
        event_active=np.array([False, True, True, False, False]), flow=np.ones((5, 1)),
        event_strength_by_node=np.array([[0, 0], [1200, 0], [1200, 0], [0, 0], [0, 0]], dtype=float),
        **{name: np.ones((5, 2)) for name in
           ("pressure", "demand", "chlorine", "arsenic", "hydraulic_outflow")})
    return trajectory, dataset, schedule, chlorine


class SimulationExportTests(unittest.TestCase):
    def test_half_open_source_intervals_match_recorded_flags(self):
        args = fixture()
        validate_trajectory(*args)
        args[0].event_strength_by_node[3, 0] = 1200
        with self.assertRaises(AssertionError):
            validate_trajectory(*args)

    def test_shifted_timestamps_are_not_silently_relabelled(self):
        args = fixture()
        args[0].time_seconds += 1800
        with self.assertRaisesRegex(ValueError, "reporting grid"):
            validate_trajectory(*args)

    def test_nonfinite_raw_values_are_rejected(self):
        args = fixture()
        args[0].pressure[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite pressure"):
            validate_trajectory(*args)

    def test_chlorine_profile_must_match_cached_input(self):
        args = fixture()
        args[0].injection[2] = 999
        with self.assertRaises(AssertionError):
            validate_trajectory(*args)


if __name__ == "__main__":
    unittest.main()
