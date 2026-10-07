from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from packages.trainer.evaluation.plots import (
    render_forecast_metric_breakdowns,
    render_scenario_forecast_targets,
    render_forecast_training_history,
)
from packages.trainer.specs import ForecastDiagnosticPlotsSpec


class ForecastPlotDiagnosticsTests(unittest.TestCase):
    def test_training_history_and_metric_breakdown_plots_are_written(self):
        history = [
            {
                "epoch": 1,
                "train": {"combined_loss": 0.8},
                "validation": {
                    "combined_loss": 0.9,
                    "metrics": {"chlorine": {"overall": {"rmse": 0.4}}},
                },
            },
            {
                "epoch": 2,
                "train": {"combined_loss": 0.5},
                "validation": {
                    "combined_loss": 0.6,
                    "metrics": {"chlorine": {"overall": {"rmse": 0.3}}},
                },
            },
        ]
        metrics = {
            "chlorine": {
                "node": {
                    "mae": {"2": 0.1, "3": 0.2},
                    "rmse": {"2": 0.15, "3": 0.25},
                },
                "horizon": {
                    "mae": {"0": 0.1, "1": 0.2},
                    "rmse": {"0": 0.15, "1": 0.25},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = render_forecast_training_history(
                root / "training-history.png",
                history=history,
                checkpoint_metric="chlorine.rmse",
            )
            metric_paths = render_forecast_metric_breakdowns(
                root / "validation",
                metrics=metrics,
                breakdowns=("node", "horizon"),
            )
            self.assertTrue(history_path.is_file())
            self.assertEqual(
                {path.name for path in metric_paths},
                {"chlorine-node-metrics.png", "chlorine-horizon-metrics.png"},
            )
            self.assertTrue(all(path.is_file() for path in metric_paths))

    def test_metric_plot_breakdowns_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            ForecastDiagnosticPlotsSpec(metric_breakdown_plots=("node", "node"))

    def test_scenario_plots_write_rolling_and_bounded_horizon_per_node(self):
        windows, horizon = 1_100, 16
        starts = np.arange(windows, dtype=np.float32)[:, None]
        leads = np.arange(horizon, dtype=np.float32)[None, :]
        times = (starts + leads) * 1_800.0
        base = np.sin((starts + leads) / 48.0)
        actual = np.stack((base, base + 1.0), axis=-1)
        predicted = actual + 0.1
        spec = ForecastDiagnosticPlotsSpec(
            node_selection="all",
            separate_node_files=True,
            rolling_prediction_steps=1,
            include_random_horizon_plot=True,
            random_horizon_cutoff_day_range=(3.0, 22.0),
            random_horizon_include_history=False,
            random_horizon_line_style="dashed",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = render_scenario_forecast_targets(
                Path(directory),
                actual={"demand": actual},
                predicted={"demand": predicted},
                node_ids=("1", "node/2"),
                forecast_time_seconds=times,
                spec=spec,
            )
            self.assertEqual(
                {path.name for path in paths},
                {
                    "demand-1-rolling.png",
                    "demand-1-random-horizon.png",
                    "demand-node_2-rolling.png",
                    "demand-node_2-random-horizon.png",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths))

    def test_random_horizon_cutoff_day_range_must_be_ordered(self):
        with self.assertRaisesRegex(ValueError, "non-negative and ordered"):
            ForecastDiagnosticPlotsSpec(
                random_horizon_cutoff_day_range=(22.0, 3.0)
            )


if __name__ == "__main__":
    unittest.main()
