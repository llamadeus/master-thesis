"""Focused checks for low-source-outflow paired chlorine trials."""

from contextlib import redirect_stdout
from io import StringIO
import multiprocessing as mp
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from tqdm import tqdm

from tools.analysis.analyze_network_delays import (
    HISTORY_WINDOWS_HOURS, HORIZON_WINDOWS_HOURS, NODE_IDS, SCENARIO_IDS,
    Trial, _RunInterrupted, _StopState, _cached_trial, _check_stop,
    _report_progress, _run_tasks, _scenario_trial_cases, _trial_digest, _trial_path,
    analyze_response, batch_trials, build_time_slots,
    paired_rates, select_trials, source_outflow_liters_per_minute, summarize,
)


def _interruptible_test_task(task):
    loops, value = task
    for _ in range(loops):
        _check_stop()
        time.sleep(.02)
    _report_progress()
    return value


class NetworkDelayDiagnosticTests(unittest.TestCase):
    def test_cartesian_weekday_clock_slots_have_three_core_week_occurrences(self):
        slots = build_time_slots()
        self.assertEqual(len(slots), 168)
        self.assertEqual(
            {(slot.weekday_phase, slot.clock_hour) for slot in slots},
            {(weekday, hour) for weekday in range(7) for hour in range(24)},
        )
        self.assertTrue(all(len(slot.candidate_steps) == 3 for slot in slots))
        self.assertEqual(
            {step // 48 for slot in slots for step in slot.candidate_steps},
            set(range(2, 23)),
        )
        self.assertTrue(all(
            step // 48 % 7 == slot.weekday_phase
            and step % 48 == slot.clock_hour * 2
            for slot in slots for step in slot.candidate_steps
        ))
        last = next(slot for slot in slots
                    if (slot.weekday_phase, slot.clock_hour) == (1, 23))
        self.assertEqual(last.candidate_steps[-1], 22 * 48 + 46)
        self.assertEqual(last.candidate_steps[-1] + 96, 24 * 48 + 46)
        self.assertLess(24 * 48 + 46, 25 * 48)
        self.assertEqual(len(build_time_slots(5)), 7 * 5)
        with self.assertRaises(ValueError):
            build_time_slots(0)

    def test_selection_uses_each_scenarios_lowest_24_hour_volume_and_unique_top_ten(self):
        slot = next(slot for slot in build_time_slots()
                    if (slot.weekday_phase, slot.clock_hour) == (2, 0))
        profiles = {}
        for index in range(11):
            flow = np.full(1201, 100 + index, dtype=float)
            profiles[f"scenario-{index}"] = (flow, f"cache-{index}")
        # The last weekly occurrence wins for scenario 10, even though the
        # first two weeks have the highest outflow in the candidate pool.
        third = slot.candidate_steps[2]
        profiles["scenario-10"][0][third:third + 48] = 1
        profiles["scenario-10"][0][third + 48:third + 96] = 1000
        selected = select_trials((slot,), profiles)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({trial.scenario_id for trial in selected}), 10)
        self.assertEqual(selected[0].scenario_id, "scenario-10")
        self.assertEqual(selected[0].start_step, third)
        self.assertEqual(selected[0].day, 16)
        self.assertEqual(selected[0].selection_rank, 1)
        self.assertEqual(selected[0].source_outflow_24h_liters, 48 * 30)
        self.assertEqual(selected[0].hydraulic_cache_identity, "cache-10")
        self.assertEqual(selected[1].scenario_id, "scenario-0")
        self.assertEqual(selected[1].start_step, slot.candidate_steps[0])
        self.assertNotIn("scenario-9", {trial.scenario_id for trial in selected})

    def test_source_outflow_uses_only_links_flowing_out_of_reservoir(self):
        flows = np.array([[5.0, -2.0], [-3.0, 4.0]])
        outflow = source_outflow_liters_per_minute(
            flows, (1, -1), 1000.0 / 60.0
        )
        np.testing.assert_allclose(outflow, [7 * 1000 / 60, 0])

    def test_full_slot_selection_has_1680_distinct_trials(self):
        profiles = {
            scenario_id: (np.full(1201, index + 1.0), f"cache-{index}")
            for index, scenario_id in enumerate(SCENARIO_IDS)
        }
        trials = select_trials(build_time_slots(), profiles)
        self.assertEqual(len(trials), 1680)
        self.assertEqual(
            len({(trial.scenario_id, trial.start_step) for trial in trials}),
            1680,
        )
        self.assertEqual(
            {trial.selection_rank for trial in trials}, set(range(1, 11))
        )

    def test_batching_requires_three_days_and_minimizes_runs(self):
        trials = tuple(
            Trial("scenario-3", 2, 0, start // 48, start, 1, 1200.0, "cache-3")
            for start in (384, 239, 96, 383, 240)
        )
        groups = batch_trials(trials)
        self.assertEqual(
            [[trial.start_step for trial in group] for group in groups],
            [[96, 240, 384], [239, 383]],
        )
        with self.assertRaises(ValueError):
            batch_trials((*trials, Trial("scenario-4", 2, 0, 2, 97, 1, 0, "cache-4")))

    def test_scenario_run_batches_pulses_and_keeps_individual_results(self):
        first = Trial("scenario-3", 2, 4, 2, 104, 1, 1200.0, "cache-3")
        second = Trial("scenario-3", 5, 4, 5, 248, 1, 1200.0, "cache-3")
        scenario = type("Scenario", (), {
            "scenario_id": "scenario-3", "content_hash": "scenario-hash",
        })()
        output = StringIO()
        events = []
        simulated_rates = []

        def simulate(_scenario, _cache_root, rates, *, progress_label):
            simulated_rates.append(rates.copy())
            chlorine = np.zeros((len(rates) + 1, len(NODE_IDS)))
            for start in np.flatnonzero(rates == 125_000):
                chlorine[start + 1, 0] = .1
                chlorine[start + 12, 0] = .2
            return np.arange(len(rates) + 1) * 1800, chlorine

        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = root / "cases" / "baselines" / "scenario-3.npz"
            baseline.parent.mkdir(parents=True)
            np.savez_compressed(
                baseline, times=np.arange(1201) * 1800,
                chlorine=np.zeros((1201, len(NODE_IDS))),
            )
            with patch(
                "tools.analysis.analyze_network_delays._simulate", side_effect=simulate,
            ), patch(
                "tools.analysis.analyze_network_delays._PROGRESS_CALLBACK",
                side_effect=lambda kind, label: events.append((kind, label)),
            ), redirect_stdout(output):
                results = _scenario_trial_cases((scenario, (second, first), raw))
            self.assertEqual(len(list((root / "cases" / "trials").glob("*.npz"))), 2)
            self.assertEqual(len(list((root / "cases" / "trials").glob("*.json"))), 0)
            for trial in (first, second):
                path = _trial_path(root, trial)
                with np.load(path, allow_pickle=False) as stored:
                    np.testing.assert_array_equal(
                        stored["times_seconds"],
                        np.arange(trial.start_step, trial.start_step + 97) * 1800,
                    )
                    np.testing.assert_array_equal(
                        stored["reference_chlorine_mg_l"], np.zeros((97, len(NODE_IDS))),
                    )
                    self.assertEqual(stored["action_chlorine_mg_l"].shape,
                                     (97, len(NODE_IDS)))
                    self.assertEqual(stored["action_chlorine_mg_l"][1, 0], .1)
                    self.assertEqual(stored["action_chlorine_mg_l"][12, 0], .2)
                    np.testing.assert_array_equal(
                        stored["reference_injection_mg_min"], np.full(96, 100_000),
                    )
                    np.testing.assert_array_equal(
                        stored["action_injection_mg_min"],
                        np.r_[125_000, np.full(95, 100_000)],
                    )
                digest = _trial_digest(scenario, trial)
                with patch(
                    "tools.analysis.analyze_network_delays.HORIZON_WINDOWS_HOURS", (4, 8, 12),
                ):
                    old_windows = _cached_trial(path, digest, trial)
                new_windows = _cached_trial(path, digest, trial)
                with patch(
                    "tools.analysis.analyze_network_delays.ABSOLUTE_THRESHOLD_MG_L", .15,
                ):
                    higher_threshold = _cached_trial(path, digest, trial)
                self.assertNotIn("fraction_6h", old_windows["rows"][0])
                self.assertAlmostEqual(new_windows["rows"][0]["fraction_4h"], 1 / 3)
                self.assertEqual(new_windows["rows"][0]["fraction_6h"], 1)
                self.assertEqual(higher_threshold["rows"][0]["onset_hours"], 6)
                self.assertEqual(higher_threshold["rows"][0]["fraction_4h"], 0)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(len(simulated_rates), 1)
        self.assertEqual(len(simulated_rates[0]), second.start_step + 96)
        np.testing.assert_array_equal(
            np.flatnonzero(simulated_rates[0] == 125_000),
            [first.start_step, second.start_step],
        )
        self.assertEqual([result["schedule"]["start_step"] for result in results],
                         [second.start_step, first.start_step])
        self.assertTrue(all(result["rows"][0]["responsive"] for result in results))
        self.assertEqual(events, [
            ("status", "scenario-3 2 pulses, d2 04:00–d5 04:00"),
            ("progress", "scenario-3 d2 04:00"),
            ("progress", "scenario-3 d5 04:00"),
        ])

    def test_parallel_progress_and_graceful_stop(self):
        context = mp.get_context("spawn")
        prior_lock = tqdm.get_lock()
        tqdm.set_lock(context.RLock())
        try:
            state = _StopState(context.Event())
            with redirect_stdout(StringIO()):
                results = _run_tasks(
                    _interruptible_test_task, ((2, 1), (2, 2)), 2,
                    stop_state=state, description="Test", total=2, unit="case",
                )
            self.assertEqual(results, [1, 2])

            state = _StopState(context.Event())
            timer = threading.Timer(.4, state.request, args=(signal.SIGINT, None))
            timer.start()
            try:
                with redirect_stdout(StringIO()), self.assertRaises(_RunInterrupted):
                    _run_tasks(
                        _interruptible_test_task, ((200, 1), (200, 2), (200, 3)),
                        2, stop_state=state, description="Test", total=3,
                        unit="case",
                    )
            finally:
                timer.join()
            self.assertTrue(state.requested)
        finally:
            tqdm.set_lock(prior_lock)

    def test_ties_choose_earliest_week_then_numeric_scenario_order(self):
        slot = build_time_slots()[0]
        profiles = {
            f"scenario-{index}": (np.full(1201, 1.0), f"cache-{index}")
            for index in range(12)
        }
        selected = select_trials((slot,), profiles)
        self.assertEqual(
            [trial.scenario_id for trial in selected],
            [f"scenario-{index}" for index in range(10)],
        )
        self.assertTrue(all(trial.start_step == slot.candidate_steps[0]
                            for trial in selected))

    def test_reference_is_constant_and_intervention_changes_one_step(self):
        start = 22 * 48 + 46
        reference, intervention = paired_rates(start)
        self.assertTrue(np.all(reference == 100_000))
        np.testing.assert_array_equal(np.flatnonzero(reference != intervention), [start])
        self.assertEqual(intervention[start], 125_000)
        self.assertEqual(intervention[start + 1], 100_000)
        self.assertLessEqual(np.max(np.abs(np.diff(intervention))), 50_000)
        with self.assertRaises(ValueError):
            paired_rates(23 * 48)

    def test_fixed_threshold_peak_t90_and_horizon_fractions(self):
        self.assertEqual(list(HORIZON_WINDOWS_HOURS), [4, 6, 8, 10, 12, 14, 16])
        response = np.zeros(97)
        response[2:4] = .2
        response[10:12] = .3
        response[20:22] = .5
        response[30] = 9e-5
        metric = analyze_response(response)
        self.assertTrue(metric["responsive"])
        self.assertEqual(metric["threshold_mg_l"], 1e-4)
        self.assertEqual(metric["onset_hours"], 1)
        self.assertEqual(metric["peak_hours"], 10)
        self.assertEqual(metric["t90_hours"], 10.5)
        self.assertEqual(metric["last_detectable_hours"], 10.5)
        self.assertEqual(metric["visible_duration_hours"], 3)
        self.assertAlmostEqual(metric["fraction_4h"], .4 / 2)
        self.assertAlmostEqual(metric["fraction_6h"], 1 / 2)
        self.assertAlmostEqual(metric["fraction_8h"], 1 / 2)
        self.assertEqual(metric["fraction_12h"], 1)
        self.assertEqual(metric["fraction_16h"], 1)
        self.assertFalse(metric["peak_reached_8h"])
        self.assertTrue(metric["peak_reached_12h"])

    def test_evaluation_windows_and_threshold_leave_trajectory_contract_unchanged(self):
        scenario = type("Scenario", (), {"content_hash": "scenario-hash"})()
        trial = Trial("scenario-3", 2, 4, 2, 104, 1, 1200.0, "cache-3")
        with patch(
            "tools.analysis.analyze_network_delays.HORIZON_WINDOWS_HOURS", (4, 8, 12),
        ), patch(
            "tools.analysis.analyze_network_delays.HISTORY_WINDOWS_HOURS", (12, 24),
        ), patch(
            "tools.analysis.analyze_network_delays.ABSOLUTE_THRESHOLD_MG_L", 2e-4,
        ):
            previous_digest = _trial_digest(scenario, trial)
        self.assertEqual(previous_digest, _trial_digest(scenario, trial))
        different_selection = Trial(
            "scenario-3", 2, 4, 2, 104, 8, 999.0, "cache-3",
        )
        self.assertEqual(previous_digest, _trial_digest(scenario, different_selection))

    def test_old_metric_only_json_is_not_a_trial_cache(self):
        scenario = type("Scenario", (), {"content_hash": "scenario-hash"})()
        trial = Trial("scenario-3", 2, 4, 2, 104, 1, 1200.0, "cache-3")
        with TemporaryDirectory() as raw:
            path = _trial_path(Path(raw), trial)
            path.parent.mkdir(parents=True)
            path.with_suffix(".json").write_text('{"rows": []}')
            self.assertEqual(path.suffix, ".npz")
            self.assertIsNone(_cached_trial(path, _trial_digest(scenario, trial), trial))

    def test_single_detectable_sample_counts_and_last_sample_is_censored(self):
        isolated = np.zeros(97)
        isolated[5] = .1
        metric = analyze_response(isolated)
        self.assertTrue(metric["responsive"])
        self.assertEqual(metric["onset_hours"], 2.5)
        self.assertEqual(metric["t90_hours"], 2.5)
        self.assertEqual(metric["visible_duration_hours"], .5)
        late = np.zeros(97)
        late[-1] = .1
        metric = analyze_response(late)
        self.assertTrue(metric["tail_censored"])
        self.assertEqual(metric["last_detectable_hours"], 48)
        self.assertEqual(metric["fraction_12h"], 0)
        tiny = np.zeros(97)
        tiny[5] = 1e-4
        self.assertFalse(analyze_response(tiny)["responsive"])

    def test_summary_keeps_all_nodes_and_censored_cases_in_denominator(self):
        early = np.zeros(97)
        early[2:4] = .1
        late = np.zeros(97)
        late[-1] = .1
        responsive = analyze_response(early)
        censored = analyze_response(late)
        silent = analyze_response(np.zeros(97))
        rows = [
            {"scenario_id": "scenario-0", "node_id": str(node), **responsive}
            for node in range(2, 32)
        ]
        rows.extend([
            {"scenario_id": "scenario-0", "node_id": "32", **censored},
            {"scenario_id": "scenario-0", "node_id": "2", **silent},
        ])
        trial = Trial("scenario-0", 2, 0, 2, 96, 1, 1200.0, "cache-0")
        cases = [{"schedule": vars(trial), "rows": rows}]
        summary = summarize(cases)
        self.assertEqual(summary["counts"]["node_trial_cases"], 32)
        self.assertEqual(summary["counts"]["responsive"], 31)
        self.assertEqual(summary["counts"]["nonresponsive"], 1)
        self.assertEqual(summary["counts"]["tail_censored"], 1)
        self.assertEqual(summary["counts"]["combinations"], 1)
        self.assertEqual(summary["counts"]["action_batches"], 1)
        self.assertEqual(summary["design"]["action_batch_separation_hours"], 72)
        self.assertEqual(len(summary["per_node"]), len(NODE_IDS))
        self.assertEqual(summary["history_coverage"]["24"]["covered"], 30)
        self.assertEqual(list(HISTORY_WINDOWS_HOURS), list(range(12, 37, 2)))
        self.assertEqual(set(summary["history_coverage"]),
                         {str(hour) for hour in HISTORY_WINDOWS_HOURS})
        self.assertEqual(set(summary["horizons"]),
                         {str(hour) for hour in HORIZON_WINDOWS_HOURS})
        self.assertEqual(summary["horizons"]["8"]["peak_reached"], 30)
        self.assertEqual(summary["horizons"]["8"]["at_least_90_percent"], 30)
        self.assertAlmostEqual(
            summary["horizons"]["8"]["at_least_90_percent_pct"], 100 * 30 / 31
        )
        self.assertEqual(summary["source_outflow_24h_liters"]["median"], 1200)
        self.assertEqual(SCENARIO_IDS, tuple(f"scenario-{index}" for index in range(80)))


if __name__ == "__main__":
    unittest.main()
