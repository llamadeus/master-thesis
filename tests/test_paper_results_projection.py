import json
from pathlib import Path
import tempfile
import unittest

from paper.tools.results_projection import SEEDS, forecast_projection, scenario_metrics


class PaperResultsProjectionTests(unittest.TestCase):
    def test_control_retains_na_counts_and_equal_scenario_weighting(self):
        rows = [dict(approach="baseline", seed="17", scenario=scenario,
                     metric="chlorine_shortfall_fraction", component="main_normal", value=value)
                for scenario, value in [("a", 0.), ("a", 0.), ("b", 2.), ("b", None), ("c", None)]]
        projected = scenario_metrics(rows)
        self.assertEqual([r["value"] for r in projected], [0., 2., None])
        self.assertEqual([r["applicable_count"] for r in projected], [2, 1, 0])
        self.assertEqual([r["not_applicable_count"] for r in projected], [0, 1, 1])
        valid = [r["value"] for r in projected if r["value"] is not None]
        self.assertEqual(sum(valid) / len(valid), 1.)

    def test_forecast_rmse_pools_steps_before_averaging_scenarios(self):
        rows = [dict(training_seed=seed, fold_id=fold, scenario_id=str(scenario),
                     horizon_step=step, observation_count=count,
                     absolute_error_sum_mg_l=error * count,
                     squared_error_sum_mg2_l2=error**2 * count)
                for seed in SEEDS for fold in range(5) for scenario in range(10)
                for step in range(1, 17)
                for count, error in [(1 if scenario < 5 else 100,
                                      (0. if step <= 8 else 2.) * (1 if scenario < 5 else 3))]]
        projected = forecast_projection(rows, "synthetic")
        model = projected["rows"][:17]
        self.assertEqual(model[0]["mae"], 2.)
        self.assertAlmostEqual(model[0]["rmse"], 2 * 2.**.5)
        self.assertEqual(model[1]["rmse"], 0.)
        self.assertEqual(model[16]["rmse"], 4.)
        with self.assertRaises(ValueError):
            forecast_projection(rows[:-1], "measured")
        with self.assertRaises(ValueError):
            forecast_projection(rows[:-1] + [rows[0]], "measured")
        self.assertEqual(forecast_projection([], "pending")["rows"], [])
        from tools.analysis.export_evaluation_results import write_plot_data
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast-plot-data.json"
            write_plot_data(path, rows)
            measured = json.loads(path.read_text())
            self.assertEqual(measured["data_kind"], "measured")
            self.assertEqual(measured["rows"], projected["rows"])
            write_plot_data(path)
            self.assertEqual(json.loads(path.read_text()), forecast_projection([], "pending"))


if __name__ == "__main__":
    unittest.main()
