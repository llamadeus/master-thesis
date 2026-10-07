from __future__ import annotations

from dataclasses import replace
import json
import unittest

import numpy as np

from packages.trainer import ControlObservationSpec, RewardSpec
from packages.trainer.control.rewards import create_reward
from packages.trainer.observations import create_control_observation
from tests.reward_inputs import reward_input
from training.rewards.v89 import DEFAULT_PARAMETERS, V89Observation, V89Reward


def precise_input(chlorine, **kwargs):
    return replace(
        reward_input(chlorine, **kwargs),
        nodes_quality=np.asarray(chlorine, dtype=float),
    )


def observe(runtime, chlorine=(0.3, 0.3), injection=125_000.0, action=0.0):
    return runtime.observe(
        node_ids=tuple(str(i) for i in range(len(chlorine))),
        chlorine=np.asarray(chlorine, dtype=float),
        current_injection=injection,
        maximum_injection=500_000.0,
        time_of_year_seconds=123_456,
        raw_action=action,
    )


def active_reward(parameters=None):
    """Exercise the alpha contract separately from the startup schedule."""
    return V89Reward({"warmup_steps": 0, "blend_steps": 0, **(parameters or {})})


def active_observation(parameters=None):
    return V89Observation(
        {"warmup_steps": 0, "blend_steps": 0, **(parameters or {})}
    )


class V89RewardTests(unittest.TestCase):
    def test_exact_defaults_and_runtime_specs(self):
        expected = {
            "target": 0.30,
            "tracking_reference": 0.01,
            "alpha_one_threshold": 0.15,
            "alpha_zero_threshold": 0.20,
            "blend_grace_steps": 4.0,
            "blend_decay_steps": 12.0,
            "warmup_steps": 12.0,
            "blend_steps": 12.0,
            "upper_penalty_min_factor": 0.0001,
            "wasted_movement_reference": 0.01,
            "wasted_movement_weight": 0.1,
            "wasted_movement_gate_full_threshold": 0.35,
            "wasted_movement_gate_zero_threshold": 0.40,
            "action_jerk_reference": 0.05,
            "action_jerk_weight": 0.1,
        }
        self.assertEqual(DEFAULT_PARAMETERS, expected)
        reward = create_reward(RewardSpec(V89Reward))
        observation = create_control_observation(ControlObservationSpec(V89Observation))
        self.assertEqual(reward.reward_id, "v89")
        self.assertEqual(reward.p, expected)
        self.assertEqual(observation.p, expected)

    def test_raw_alpha_uses_smoothstep_of_the_current_minimum(self):
        for minimum, expected in (
            (0.0, 1.0),
            (0.15, 1.0),
            (0.175, 0.5),
            (0.20, 0.0),
            (0.50, 0.0),
        ):
            with self.subTest(minimum=minimum):
                _, info = active_reward().compute(
                    precise_input([minimum, 0.30, 0.80])
                )
                self.assertAlmostEqual(info["alpha_raw"], expected)
                self.assertAlmostEqual(info["alpha"], expected)

    def test_alpha_depends_only_on_the_single_current_minimum(self):
        _, first = active_reward().compute(precise_input([0.175, 0.30, 0.30]))
        _, second = active_reward().compute(precise_input([0.90, 0.175, 0.60]))
        self.assertEqual(first["minimum_chlorine"], second["minimum_chlorine"])
        self.assertEqual(first["alpha_raw"], second["alpha_raw"])
        self.assertEqual(first["alpha"], second["alpha"])

    def test_attack_grace_and_smoothstep_release(self):
        reward = active_reward({"blend_grace_steps": 2, "blend_decay_steps": 4})
        _, triggered = reward.compute(precise_input([0.15, 0.40]))
        released = [
            reward.compute(precise_input([0.30, 0.40]))[1] for _ in range(7)
        ]
        self.assertEqual(triggered["alpha"], 1.0)
        self.assertEqual(triggered["release_remaining"], 1.0)
        np.testing.assert_allclose(
            [info["alpha"] for info in released],
            [1.0, 1.0, 0.84375, 0.5, 0.15625, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            [info["release_remaining"] for info in released],
            [1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0],
        )
        self.assertEqual(
            [info["release_step"] for info in released],
            [1, 2, 3, 4, 5, 6, 6],
        )

    def test_equal_or_larger_raw_alpha_restarts_release(self):
        reward = active_reward({"blend_grace_steps": 0, "blend_decay_steps": 4})
        reward.compute(precise_input([0.15]))
        reward.compute(precise_input([0.30]))
        _, half_released = reward.compute(precise_input([0.30]))
        self.assertEqual(half_released["alpha"], 0.5)
        _, equal = reward.compute(precise_input([0.175]))
        self.assertAlmostEqual(equal["alpha_raw"], 0.5)
        self.assertAlmostEqual(equal["alpha"], 0.5)
        self.assertEqual(equal["release_step"], 0)
        self.assertEqual(equal["release_remaining"], 1.0)
        _, equal_again = reward.compute(precise_input([0.175]))
        self.assertEqual(equal_again["release_step"], 0)
        _, larger = reward.compute(precise_input([0.15]))
        self.assertEqual(larger["attack_alpha"], 1.0)
        self.assertEqual(larger["release_step"], 0)

    def test_alpha_scales_only_upper_quadratic_tracking(self):
        _, active = active_reward().compute(precise_input([0.15, 0.40]))
        self.assertEqual(active["alpha"], 1.0)
        self.assertEqual(active["upper_penalty_factor"], 0.0001)
        self.assertAlmostEqual(active["lower_tracking_penalty"], 112.5)
        self.assertAlmostEqual(active["upper_tracking_penalty"], 0.005)
        self.assertAlmostEqual(active["tracking_penalty"], 112.505)
        np.testing.assert_allclose(active["tracking_cost_by_node"], [225.0, 0.01])

        _, inactive = active_reward().compute(precise_input([0.30, 0.40]))
        self.assertEqual(inactive["alpha"], 0.0)
        self.assertEqual(inactive["upper_penalty_factor"], 1.0)
        self.assertAlmostEqual(inactive["tracking_penalty"], 50.0)

    def test_upper_penalty_min_factor_is_configurable(self):
        reward = active_reward({"upper_penalty_min_factor": 0.25})
        _, info = reward.compute(precise_input([0.15, 0.40]))
        self.assertEqual(info["alpha"], 1.0)
        self.assertEqual(info["upper_penalty_factor"], 0.25)
        self.assertAlmostEqual(info["upper_tracking_penalty"], 12.5)

    def test_alpha_parameters_are_validated(self):
        invalid = (
            {"alpha_one_threshold": -0.01},
            {"alpha_one_threshold": 0.20},
            {"alpha_zero_threshold": 0.15},
            {"alpha_zero_threshold": 0.31},
            {"blend_grace_steps": -1},
            {"blend_grace_steps": 1.5},
            {"blend_decay_steps": 0},
            {"blend_decay_steps": 1.5},
            {"warmup_steps": -1},
            {"warmup_steps": 1.5},
            {"blend_steps": -1},
            {"blend_steps": 1.5},
            {"upper_penalty_min_factor": -0.01},
            {"upper_penalty_min_factor": 1.01},
            {"wasted_movement_gate_full_threshold": -0.01},
            {"wasted_movement_gate_full_threshold": 0.40},
            {"wasted_movement_gate_zero_threshold": 0.35},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                V89Reward(parameters)

    def test_wasted_movement_uses_all_node_mean_when_alpha_is_zero(self):
        reward = active_reward()
        reward.compute(precise_input([0.30, 0.40], raw_action=0.0))
        _, info = reward.compute(precise_input([0.31, 0.42], raw_action=0.0))
        np.testing.assert_allclose(info["movement_by_node"], [0.01, 0.02])
        np.testing.assert_allclose(info["useful_movement_by_node"], [0.0, 0.0])
        np.testing.assert_allclose(info["wasted_movement_by_node"], [0.01, 0.02])
        self.assertEqual(info["alpha"], 0.0)
        self.assertAlmostEqual(info["all_node_wasted_movement_cost"], 2.5)
        self.assertAlmostEqual(info["wasted_movement_penalty"], 0.25)

    def test_wasted_movement_gate_uses_previous_chlorine_and_smoothstep(self):
        reward = active_reward()
        reward.compute(precise_input([0.15, 0.35, 0.375, 0.40]))
        _, info = reward.compute(precise_input([0.15, 0.41, 0.385, 0.42]))
        np.testing.assert_allclose(
            info["wasted_movement_gate_by_node"], [1.0, 1.0, 0.5, 0.0]
        )
        self.assertEqual(info["alpha"], 1.0)

    def test_removed_weights_are_rejected(self):
        for name in ("wasted_movement_extra_weight", "midpoint_tracking_weight"):
            for runtime in (V89Reward, V89Observation):
                with self.subTest(name=name, runtime=runtime):
                    with self.assertRaises(ValueError):
                        runtime({name: 0.0})

    def test_removed_penalty_diagnostics_are_absent(self):
        _, info = active_reward().compute(precise_input([0.15, 0.40]))
        self.assertFalse(any(name.startswith("midpoint_") for name in info))
        self.assertNotIn("effective_wasted_movement_weight", info)

    def test_wasted_movement_gate_thresholds_are_configurable(self):
        reward = active_reward(
            {
                "wasted_movement_gate_full_threshold": 0.30,
                "wasted_movement_gate_zero_threshold": 0.50,
            }
        )
        reward.compute(precise_input([0.20, 0.30, 0.40, 0.50]))
        _, info = reward.compute(precise_input([0.20, 0.31, 0.41, 0.51]))
        np.testing.assert_allclose(
            info["wasted_movement_gate_by_node"], [1.0, 1.0, 0.5, 0.0]
        )

    def test_filtered_wasted_movement_is_normalized_by_active_gate_sum(self):
        reward = active_reward()
        reward.compute(precise_input([0.15, 0.40, 0.40]))
        _, info = reward.compute(precise_input([0.14, 0.42, 0.42]))
        np.testing.assert_allclose(
            info["wasted_movement_cost_by_node"], [1.0, 4.0, 4.0]
        )
        np.testing.assert_allclose(
            info["wasted_movement_gate_by_node"], [1.0, 0.0, 0.0]
        )
        self.assertEqual(info["wasted_movement_gate_sum"], 1.0)
        self.assertAlmostEqual(info["filtered_wasted_movement_cost"], 1.0)
        self.assertAlmostEqual(info["wasted_movement_penalty"], 0.1)

    def test_wasted_movement_blends_normal_and_filtered_cost_with_alpha(self):
        reward = active_reward()
        reward.compute(precise_input([0.175, 0.35, 0.40]))
        _, info = reward.compute(precise_input([0.175, 0.36, 0.42]))
        self.assertAlmostEqual(info["alpha"], 0.5)
        self.assertAlmostEqual(
            info["all_node_wasted_movement_cost"], 5.0 / 3.0
        )
        self.assertAlmostEqual(info["filtered_wasted_movement_cost"], 0.5)
        self.assertAlmostEqual(info["blended_wasted_movement_cost"], 13.0 / 12.0)
        self.assertAlmostEqual(info["wasted_movement_penalty"], 13.0 / 120.0)

    def test_zero_gate_sum_has_zero_filtered_wasted_cost(self):
        reward = active_reward()
        reward.compute(precise_input([0.15, 0.40]))
        reward.compute(precise_input([0.40, 0.40]))
        _, info = reward.compute(precise_input([0.41, 0.42]))
        self.assertGreater(info["alpha"], 0.0)
        self.assertEqual(info["wasted_movement_gate_sum"], 0.0)
        self.assertEqual(info["filtered_wasted_movement_cost"], 0.0)
        self.assertEqual(info["wasted_movement_penalty"], 0.0)

    def test_first_transition_and_action_jerk_use_v88_formula(self):
        reward = active_reward()
        _, first = reward.compute(precise_input([0.30], raw_action=0.40))
        _, steady = reward.compute(precise_input([0.30], raw_action=0.40))
        _, changed = reward.compute(precise_input([0.30], raw_action=0.45))
        self.assertEqual(first["wasted_movement_penalty"], 0.0)
        self.assertEqual(first["action_jerk_penalty"], 0.0)
        self.assertEqual(steady["action_jerk_penalty"], 0.0)
        self.assertAlmostEqual(changed["action_jerk_penalty"], 0.1)

    def test_total_contains_the_three_top_level_penalties(self):
        reward = active_reward()
        reward.compute(precise_input([0.30, 0.30], raw_action=-0.5))
        value, info = reward.compute(precise_input([0.31, 0.50], raw_action=0.5))
        self.assertAlmostEqual(
            info["tracking_penalty"],
            info["lower_tracking_penalty"] + info["upper_tracking_penalty"],
        )
        total = sum(
            info[name]
            for name in (
                "tracking_penalty",
                "wasted_movement_penalty",
                "action_jerk_penalty",
            )
        )
        self.assertAlmostEqual(info["total_penalty"], total)
        self.assertAlmostEqual(value, -total)
        self.assertEqual(info["total_reward"], value)

    def test_reward_checkpoint_round_trip_and_reset(self):
        original = active_reward()
        for chlorine, action in (
            ([0.20, 0.30], 0.4),
            ([0.30, 0.20], 0.1),
            ([0.30, 0.30], -0.1),
        ):
            original.compute(precise_input(chlorine, raw_action=action))
        saved = json.loads(json.dumps(original.state_dict()))
        restored = active_reward()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        sample = precise_input([0.30, 0.40], raw_action=0.2)
        self.assertEqual(original.compute(sample), restored.compute(sample))
        original.reset()
        self.assertEqual(original.state_dict(), active_reward().state_dict())


class V89ObservationTests(unittest.TestCase):
    def test_layout_omits_aggregates_and_contains_alpha_state(self):
        observation = active_observation()
        observe(observation, [0.30, 0.30, 0.30])
        actual = observe(
            observation,
            [0.15, 0.30, 0.40],
            injection=150_000.0,
            action=0.4,
        )
        self.assertEqual(
            observation.observation_size(tuple(str(i) for i in range(31))), 43
        )
        self.assertEqual(actual.shape, (15,))
        np.testing.assert_allclose(
            actual[:5],
            [0.15, 0.30, 0.40, 150_000, 25_000],
        )
        self.assertEqual(actual[5:11].shape, (6,))
        self.assertEqual(actual[11], np.float32(0.4))
        self.assertEqual(actual[12], np.float32(1.0))
        self.assertEqual(actual[13], np.float32(1.0))
        self.assertEqual(actual[14], np.float32(1.0))

    def test_inactive_observation_has_no_release_remaining(self):
        actual = observe(V89Observation(), [0.30, 0.40])
        np.testing.assert_array_equal(actual[-3:], np.zeros(3, dtype=np.float32))

    def test_observation_alpha_state_matches_reward(self):
        parameters = {"blend_grace_steps": 2, "blend_decay_steps": 4}
        observation = active_observation(parameters)
        reward = active_reward(parameters)
        observe(observation, [0.30, 0.30])
        sequence = ([0.15, 0.40],) + ([0.30, 0.40],) * 7 + ([0.175, 0.40],)
        for chlorine in sequence:
            _, info = reward.compute(precise_input(chlorine))
            actual = observe(observation, chlorine)
            self.assertEqual(actual[-3], np.float32(info["alpha"]))
            self.assertEqual(actual[-2], np.float32(info["release_remaining"]))
            self.assertEqual(actual[-1], np.float32(info["activation_progress"]))
            self.assertEqual(
                observation.state_dict()["attack_alpha"],
                reward.state_dict()["attack_alpha"],
            )
            self.assertEqual(
                observation.state_dict()["release_step"],
                reward.state_dict()["release_step"],
            )

    def test_observation_checkpoint_round_trip_and_reset(self):
        original = active_observation()
        observe(original, [0.30, 0.30], injection=125_000.0)
        observe(original, [0.20, 0.30], injection=150_000.0)
        saved = json.loads(json.dumps(original.state_dict()))
        restored = active_observation()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        np.testing.assert_array_equal(
            observe(original, [0.30, 0.40], injection=175_000.0),
            observe(restored, [0.30, 0.40], injection=175_000.0),
        )
        original.reset()
        self.assertEqual(original.state_dict(), active_observation().state_dict())


class V89WarmupTests(unittest.TestCase):
    def test_default_schedule_freezes_alpha_and_blends_its_effect(self):
        reward, observation = V89Reward(), V89Observation()
        initial = observe(observation, [0.0, 0.5])
        np.testing.assert_array_equal(initial[-3:], np.zeros(3, dtype=np.float32))
        self.assertEqual(observation.state_dict()["episode_step"], 0)

        for step in range(1, 28):
            chlorine = [0.15, 0.4 + step * 0.01]
            _, info = reward.compute(
                precise_input(chlorine, raw_action=step * 0.01)
            )
            actual = observe(observation, chlorine, action=step * 0.01)
            expected = min(max((step - 12) / 12, 0.0), 1.0)
            upper_factor = 1.0 - expected + 0.0001 * expected

            self.assertAlmostEqual(info["activation_progress"], expected)
            self.assertAlmostEqual(info["alpha"], expected)
            self.assertAlmostEqual(info["upper_penalty_factor"], upper_factor)
            self.assertAlmostEqual(
                info["upper_tracking_penalty"],
                upper_factor * ((chlorine[1] - 0.30) / 0.01) ** 2 / 2,
            )
            self.assertAlmostEqual(info["lower_tracking_penalty"], 112.5)
            if step > 1:
                self.assertAlmostEqual(
                    info["wasted_movement_penalty"],
                    0.1 * 0.5 * (1.0 - expected),
                )
                self.assertAlmostEqual(info["action_jerk_penalty"], 0.004)
            if step <= 12:
                self.assertEqual(info["attack_alpha"], 0.0)
                self.assertEqual(info["release_step"], 16)
                self.assertEqual(info["release_remaining"], 0.0)
            np.testing.assert_allclose(
                actual[-3:],
                [info["alpha"], info["release_remaining"], expected],
            )
            for key in ("episode_step", "attack_alpha", "release_step"):
                self.assertEqual(
                    observation.state_dict()[key], reward.state_dict()[key]
                )
            self.assertEqual(info["episode_step"], step)

    def test_warmup_deficits_are_not_remembered(self):
        reward = V89Reward()
        for _ in range(12):
            reward.compute(precise_input([0.0, 0.4]))
        for step in range(13, 26):
            _, info = reward.compute(precise_input([0.30, 0.4]))
            self.assertEqual(info["alpha"], 0.0)
            self.assertEqual(info["attack_alpha"], 0.0)
            self.assertEqual(info["upper_penalty_factor"], 1.0)
            self.assertEqual(
                info["activation_progress"], min((step - 12) / 12, 1.0)
            )

    def test_checkpoints_resume_schedule_and_reset_restarts_warmup(self):
        for prefix in (0, 6, 12, 13, 18, 24, 25):
            with self.subTest(prefix=prefix):
                reward, observation = V89Reward(), V89Observation()
                observe(observation, [0.0, 0.4])
                for _ in range(prefix):
                    reward.compute(precise_input([0.20, 0.4]))
                    observe(observation, [0.20, 0.4])
                restored_reward, restored_observation = V89Reward(), V89Observation()
                restored_reward.load_state_dict(
                    json.loads(json.dumps(reward.state_dict()))
                )
                restored_observation.load_state_dict(
                    json.loads(json.dumps(observation.state_dict()))
                )
                for minimum in [0.20, 0.30] * 13:
                    current = precise_input([minimum, 0.4])
                    self.assertEqual(
                        reward.compute(current), restored_reward.compute(current)
                    )
                    np.testing.assert_array_equal(
                        observe(observation, [minimum, 0.4]),
                        observe(restored_observation, [minimum, 0.4]),
                    )
                reward.reset()
                observation.reset()
                self.assertEqual(reward.state_dict(), V89Reward().state_dict())
                self.assertEqual(
                    observation.state_dict(), V89Observation().state_dict()
                )

    def test_invalid_schedule_states_are_rejected_without_mutation(self):
        for runtime in (V89Reward(), V89Observation()):
            if isinstance(runtime, V89Reward):
                runtime.compute(precise_input([0.20]))
            else:
                observe(runtime, [0.20])
            saved = runtime.state_dict()
            for changes in (
                {"episode_step": -1},
                {"episode_step": True},
                {"episode_step": 1.5},
                {"attack_alpha": 1.0, "release_step": 0},
            ):
                with self.subTest(runtime=type(runtime).__name__, changes=changes):
                    with self.assertRaises(ValueError):
                        runtime.load_state_dict({**saved, **changes})
                    self.assertEqual(runtime.state_dict(), saved)


if __name__ == "__main__":
    unittest.main()
