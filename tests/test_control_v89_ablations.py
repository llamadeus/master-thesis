from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from packages.trainer import ControlHooksSpec, OutputSpec
from packages.trainer.control.runner import (_initialize_pretrained_policy,
                                             _training_vec_normalize)
from packages.trainer.serialization import canonical_data, config_hash
from tools.control.bc.train_mpc_teacher_recurrent_ppo import (_PolicySpaceEnv,
                                                              build_model)
from training import run_control_v89_ablations as ablations
from training import run_principal_v89_pipeline as principal


class ControlV89AblationTests(unittest.TestCase):
    def test_variants_keep_principal_contract_and_separate_outputs(self) -> None:
        seed = 17
        bundle = Path("/hypothetical-v89-bc-bundle")
        reference = principal.build_control_experiment(seed, bundle)
        experiment = ablations.build_experiment(seed, bundle)
        self.assertEqual(tuple(job.name for job in experiment.jobs), tuple(ablations.VARIANTS))
        for job in experiment.jobs:
            baseline = reference.job("baseline")
            self.assertEqual(job.training_seed, seed)
            if job.name == ablations.LONG_BASELINE_NAME:
                self.assertEqual(job.total_timesteps, 3_400_000)
                self.assertEqual(ablations._training_registry().runner_for("control").training_target(job),
                                 4_000_000)
                self.assertEqual(job.algorithm, baseline.algorithm)
                self.assertIsNone(job.algorithm.pretrained_policy)
            else:
                self.assertEqual(job.total_timesteps, 1_000_000)
                self.assertEqual(job.algorithm.pretrained_policy, bundle)
                self.assertEqual(job.algorithm.pretrained_components, ablations.VARIANTS[job.name])
            self.assertEqual(job.algorithm.eval_frequency, 50_000)
            self.assertEqual(job.algorithm.checkpoint_frequency, 50_000)
            self.assertEqual(job.reward, baseline.reward)
            self.assertEqual(job.observation, baseline.observation)
            self.assertEqual(job.dataset, baseline.dataset)
            self.assertFalse(job.hooks.on_checkpoint or job.hooks.on_finish)
            self.assertTrue(job.evaluate_test)
            self.assertIn(f"control/ablations/v89/{job.name}",
                          str(ablations.principal.expected_run_directory(experiment, job.name)))
            self.assertNotEqual(config_hash(job), config_hash(baseline))
        for job in reference.jobs:
            self.assertNotIn("pretrained_components", canonical_data(job)["algorithm"])

    def test_selected_bc_components_preserve_ordinary_actor_and_critic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            job = principal.build_control_experiment(17, bundle).job("baseline")
            obs_size = 43
            source_env = VecNormalize(
                DummyVecEnv([lambda: _PolicySpaceEnv(obs_size)]),
                norm_obs=True, norm_reward=True,
            )
            source_env.obs_rms.update(np.full((5, obs_size), 3.0, dtype=np.float32))
            source = build_model(torch.device("cpu"), 7, source_env, observation_size=obs_size)
            with torch.no_grad():
                source.policy.action_net.bias.fill_(0.75)
                source.policy.log_std.fill_(float(np.log(0.08)))
                source.policy.value_net.bias.fill_(0.5)
            source.save(str(bundle / "model"))
            source_env.save(str(bundle / "vec_normalize.pkl"))
            manifest = {
                "schema_version": 2,
                "kind": "recurrent-ppo-behavior-cloning",
                "action_contract": {
                    "raw_minimum": -1.0, "raw_maximum": 1.0,
                    "max_delta": job.max_delta, "minimum_injection": 0.0,
                    "maximum_injection": job.max_injection,
                    "initial_injection": job.initial_injection, "cumulative": True,
                },
                "policy_observation_contract": {"max_injection": job.max_injection},
                "reward": {"id": job.reward.factory.reward_id,
                           "parameters": job.reward.as_dict()},
                "observation_preprocessing": {
                    "kind": "vec-normalize", "artifact": "vec_normalize.pkl",
                    "norm_obs": True,
                },
                "policy": {
                    "initial_action_std": 0.08,
                    "initial_action_std_source": principal.CEM_ACTION_STD_SOURCE,
                },
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            try:
                for name, components in ablations.VARIANTS.items():
                    if not components:
                        continue
                    with self.subTest(name=name):
                        algorithm = replace(
                            job.algorithm,
                            pretrained_policy=bundle,
                            pretrained_components=components,
                            n_envs=1, n_steps=2, batch_size=2, device="cpu",
                        )
                        train_env = _training_vec_normalize(
                            DummyVecEnv([lambda: _PolicySpaceEnv(obs_size)]),
                            algorithm, job.observation, pretrained_bundle=bundle,
                        )
                        try:
                            target = build_model(torch.device("cpu"), 42, train_env,
                                                 observation_size=obs_size)
                            actor_before = target.policy.action_net.bias.detach().clone()
                            std_before = target.policy.log_std.detach().clone()
                            critic_before = target.policy.value_net.bias.detach().clone()
                            metadata = _initialize_pretrained_policy(
                                target, algorithm, bundle=bundle, job=job,
                            )
                            torch.testing.assert_close(target.policy.action_net.bias,
                                                       actor_before)
                            torch.testing.assert_close(target.policy.value_net.bias,
                                                       critic_before)
                            self.assertEqual(target.policy.optimizer.state, {})
                            if "action_std" in components:
                                torch.testing.assert_close(target.policy.log_std,
                                                           source.policy.log_std)
                                self.assertEqual(metadata["initial_action_std"], 0.08)
                                self.assertEqual(metadata["initial_action_std_source"],
                                                 principal.CEM_ACTION_STD_SOURCE)
                            else:
                                torch.testing.assert_close(target.policy.log_std,
                                                           std_before)
                                self.assertIsNone(metadata["initial_action_std"])
                            if "observation_normalization" in components:
                                self.assertEqual(float(train_env.obs_rms.count),
                                                 float(source_env.obs_rms.count))
                            else:
                                self.assertLess(float(train_env.obs_rms.count),
                                                float(source_env.obs_rms.count))
                        finally:
                            train_env.close()
            finally:
                source_env.close()

    def test_teacher_median_mismatch_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            experiment = ablations.build_experiment(17, bundle)
            manifest = {"policy": {
                "initial_action_std_source": principal.CEM_ACTION_STD_SOURCE,
            }}
            with patch.object(principal, "_bc_is_complete", return_value=True), \
                 patch.object(principal, "_read_json", return_value=manifest), \
                 patch.object(principal, "_initial_action_std", return_value=0.08), \
                 patch("sb3_contrib.RecurrentPPO.load") as load:
                load.return_value.policy.log_std.detach.return_value.cpu.return_value.numpy.return_value = np.array([np.log(0.2)])
                with self.assertRaisesRegex(RuntimeError, "disagrees with the teacher median"):
                    ablations._preflight(17, bundle, {}, experiment)

    def test_dry_run_lists_five_seeds_and_does_not_execute(self) -> None:
        output = io.StringIO()
        with patch.object(ablations, "_bc_inputs", side_effect=lambda seed: (Path(f"/bc/{seed}"), {})), \
             patch.object(ablations, "_preflight") as preflight, \
             patch.object(ablations, "execute_experiment") as execute, \
             redirect_stdout(output):
            result = ablations.main(["run", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().count('"seed":'), 5)
        self.assertEqual(output.getvalue().count('"variant":'), 20)
        preflight.assert_not_called()
        execute.assert_not_called()

    def test_selected_variant_resumes_with_seed_matched_bundle(self) -> None:
        provenance = {"teacher_first_action_final_cem_std_median": 0.08}
        with patch.object(ablations, "_bc_inputs", return_value=(Path("/bc/123"), {})), \
             patch.object(ablations, "_preflight", return_value=provenance) as preflight, \
             patch.object(ablations, "execute_experiment") as execute, \
             redirect_stdout(io.StringIO()):
            result = ablations.main([
                "run", "--seed", "123", "--variant", "offline-normalization",
                "--workers", "2", "--resume",
            ])
        self.assertEqual(result, 0)
        experiment = execute.call_args.args[0]
        self.assertEqual({job.training_seed for job in experiment.jobs}, {123})
        self.assertEqual({job.algorithm.pretrained_policy for job in experiment.jobs
                          if job.name == "offline-normalization"},
                         {Path("/bc/123")})
        self.assertEqual(execute.call_args.kwargs["selected"], {"offline-normalization"})
        self.assertTrue(execute.call_args.kwargs["resume"])
        self.assertEqual(execute.call_args.kwargs["workers"], 2)
        runner = execute.call_args.kwargs["registry"].runner_for("control")
        self.assertTrue(runner.defer_evaluation)
        self.assertFalse(runner.render_plots)
        preflight.assert_called_once()

    def test_long_baseline_runs_for_all_seeds_without_bc_preflight(self) -> None:
        with patch.object(ablations, "_bc_inputs", side_effect=lambda seed: (Path(f"/bc/{seed}"), {})), \
             patch.object(ablations, "_require_long_baseline_checkpoint") as checkpoint, \
             patch.object(ablations, "_preflight") as preflight, \
             patch.object(ablations, "execute_experiment") as execute, \
             redirect_stdout(io.StringIO()):
            result = ablations.main(["run", "--variant", "baseline-4m"])
        self.assertEqual(result, 0)
        preflight.assert_not_called()
        self.assertEqual(checkpoint.call_count, len(principal.SEEDS))
        self.assertEqual(execute.call_count, len(principal.SEEDS))
        for seed, call in zip(principal.SEEDS, execute.call_args_list):
            job = call.args[0].job("baseline-4m")
            self.assertEqual(job.training_seed, seed)
            self.assertEqual(job.total_timesteps, 3_400_000)
            self.assertEqual(call.kwargs["registry"].runner_for("control").training_target(job),
                             4_000_000)
            self.assertTrue(call.kwargs["resume"])
            self.assertIsNone(job.algorithm.pretrained_policy)
            self.assertEqual(call.kwargs["selected"], {"baseline-4m"})

    def test_long_baseline_keeps_original_hash_and_full_directory_for_every_seed(self) -> None:
        for seed in principal.SEEDS:
            experiment = ablations.build_experiment(seed, Path(f"/bc/{seed}"))
            job = experiment.job("baseline-4m")
            original = replace(
                principal.build_control_experiment(seed, Path(f"/bc/{seed}")).job("baseline"),
                name="baseline-4m", total_timesteps=3_400_000,
                hooks=ControlHooksSpec(), output=OutputSpec(group="control/ablations/v89/baseline-4m"),
            )
            original_experiment = replace(experiment, jobs=(original,))
            self.assertEqual(config_hash(job), config_hash(original))
            self.assertEqual(principal.expected_run_directory(experiment, job.name),
                             principal.expected_run_directory(original_experiment, original.name))
            self.assertIn("/baseline-4m/", str(principal.expected_run_directory(experiment, job.name)))

    def test_long_baseline_dry_run_shows_extended_target_without_changing_identity(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = ablations.main(["describe", "--seed", "17", "--variant", "baseline-4m"])
        self.assertEqual(result, 0)
        job = json.loads(output.getvalue())["jobs"][0]
        self.assertEqual(job["total_timesteps"], 4_000_000)
        original = ablations.build_experiment(17, Path("/bc/17"))
        self.assertEqual(job["output"], str(principal.expected_run_directory(original, "baseline-4m")))

    def test_long_baseline_missing_checkpoint_never_starts_fresh_training(self) -> None:
        with patch.object(ablations, "_bc_inputs", return_value=(Path("/bc/17"), {})), \
             patch("packages.trainer.control.runner.ControlRunner.can_resume", return_value=False), \
             patch.object(ablations, "execute_experiment") as execute:
            result = ablations.main(["run", "--variant", "baseline-4m"])
        self.assertEqual(result, 2)
        execute.assert_not_called()

    def test_all_long_baseline_checkpoints_are_checked_before_any_training(self) -> None:
        with patch.object(ablations, "_bc_inputs", side_effect=lambda seed: (Path(f"/bc/{seed}"), {})), \
             patch.object(ablations, "_require_long_baseline_checkpoint", side_effect=[None, RuntimeError("missing checkpoint")]), \
             patch.object(ablations, "execute_experiment") as execute:
            result = ablations.main(["run", "--variant", "baseline-4m"])
        self.assertEqual(result, 2)
        execute.assert_not_called()

    def test_all_selected_bundles_are_checked_before_training(self) -> None:
        with patch.object(ablations, "_bc_inputs", side_effect=lambda seed: (Path(f"/bc/{seed}"), {})), \
             patch.object(ablations, "_require_long_baseline_checkpoint"), \
             patch.object(ablations, "_preflight", side_effect=[
                 {"teacher_first_action_final_cem_std_median": 0.08},
                 RuntimeError("missing bundle"),
             ]), \
             patch.object(ablations, "execute_experiment") as execute, \
             redirect_stdout(io.StringIO()):
            result = ablations.main(["run", "--seed", "17", "--seed", "42"])
        self.assertEqual(result, 2)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
