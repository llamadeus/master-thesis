# Pretraining for Efficient Reinforcement Learning-Based Chlorination Control in Drinking Water Distribution Systems

This repository provides the paper, source code, and reproduction tools for the master's thesis _Pretraining for Efficient Reinforcement Learning-Based Chlorination Control in Drinking Water Distribution Systems_. The thesis investigates how model predictive control (MPC)-guided pretraining affects the online sample efficiency and final physical control performance of a recurrent chlorination policy in the simulated Hanoi drinking-water distribution network.

The implementation covers both training approaches examined in the thesis. The baseline approach trains a policy from scratch, while the pretrained approach initializes the recurrent actor through behavior cloning of MPC teacher actions before online fine-tuning. Both approaches use `RecurrentPPO` under the same nominal budget of one million online interactions.

The submitted paper is available at [`output/thesis.pdf`](output/thesis.pdf), with the Typst source in `paper/` and the data underlying its figures and tables in `paper/dataset/` and `paper/results/`. Simulation trajectories, trained models, and raw evaluation outputs are not included in the submission.

The original contributions are licensed under [MIT](LICENSE), as specified in the [License](#license) section. Attribution and licensing details for third-party material are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Reproducing the experiments

The reproduction procedure covers data preparation, model training, policy evaluation, and export of the data used by the paper. It begins with the shared hydraulic scenarios and offline trajectories. The principal pipeline then trains the forecasting models, generates MPC teacher actions, initializes the recurrent actor through behavior cloning, and trains and evaluates both policy approaches. The export scripts convert the recorded measurements into the JSON and CSV inputs used by the paper. The remaining sections describe the additional variants, dataset illustrations, network-delay analysis, and parameter-selection experiments.

Run all commands from the repository root. Simulation trajectories, trained models, and raw evaluation outputs are stored under `data/`, which is excluded by `.gitignore`.

### 1. Prepare the software environment

The experimental software environment uses Python 3.13.3.

On macOS or Linux, create a virtual environment and install the dependencies:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use:

```powershell
py -3.13 -m venv .venv-windows
.\.venv-windows\Scripts\python.exe -m pip install -r requirements.txt
```

The examples below use `.venv/bin/python` and Bash line continuations. On Windows, replace the interpreter with `.\.venv-windows\Scripts\python.exe`. Enter each command on one line or use PowerShell backticks for line continuation.

To use CUDA on Linux or Windows, first install a PyTorch 2.11.0 wheel compatible with the GPU and driver. Then install the remaining requirements. For example, the Windows CUDA 13.0 installation uses:

```powershell
.\.venv-windows\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-windows\Scripts\python.exe -m pip install -r requirements.txt
```

The [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2110) list the available wheels and their installation commands.

Fixed seeds control scenario generation, contamination schedules, and repeated training runs. Numerical reproducibility across hardware and software platforms is subject to the limitations described in the [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html). Model parameters and numerical results may therefore differ across platforms. Runtime measurements also depend on the hardware and execution settings.

### 2. Generate the shared dataset

The dataset contains 100 hydraulic base scenarios for the Hanoi network. Scenarios 0 to 79 form the training partition, scenarios 80 to 89 the validation partition, and scenarios 90 to 99 the test partition. The required chemistry template is included at `external/Environment/hanoi_arsenic_contamination.msx`.

Prepare the hydraulic base scenarios:

```bash
.venv/bin/python tools/dataset/generate_dataset.py
```

The script retrieves the LeakDB source scenarios through EPyT-Flow and stores the prepared scenarios in `data/scenarios/hanoi/`. Initial preparation therefore requires access to the external scenario source. The default configuration prepares all 100 study scenarios. To select individual scenarios, repeat `--scenario-id`, for example `--scenario-id 0 --scenario-id 85`.

Each offline trajectory combines one hydraulic scenario, one fixed arsenic schedule, and one prescribed chlorine profile. Before training, generate the complete offline dataset:

```bash
.venv/bin/python tools/dataset/materialize_forecasting_trajectories.py --workers 4
```

This command generates 2,500 trajectories across the three partitions and records the preparation time under `data/materialization/`. The `--workers` argument specifies the number of worker processes and should be set according to the available CPU resources.

Reproducing the computational-cost table requires a measurement of complete offline dataset generation. Run the command once in a fresh `data/` directory, before any other command generates forecasting trajectories. The complete paper export requires a single completed preparation report covering fresh generation of all 2,500 trajectories. Subsequent executions create additional reports and may reuse existing trajectories, so their timings do not represent the complete generation cost.

### 3. Run the principal comparison

The principal comparison comprises five matched pairs of training runs with seeds `17`, `42`, `123`, `1337`, and `71026`. For each seed, the pipeline executes the following stages in dependency order.

| Stage name         | Purpose                                                                                                          | Required earlier stages      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `forecasting`      | Train five scenario-wise cross-fitted chlorine forecasters.                                                      | Shared offline trajectories. |
| `mpc`              | Generate teacher actions for training with the fold-specific forecasters and for validation with their ensemble. | `forecasting`.               |
| `behavior-cloning` | Train the recurrent actor to reproduce the teacher actions and save its initialization bundle.                   | `mpc`.                       |
| `baseline`         | Train the baseline policy from scratch.                                                                          | Shared hydraulic scenarios.  |
| `fine-tuning`      | Fine-tune the policy initialized through behavior cloning.                                                       | `behavior-cloning`.          |

Both approaches complete the common online training budget. For each approach and seed, the checkpoint with the highest mean validation return is selected and evaluated on the separate test cases. A complete pipeline execution produces the validation measurements used for the learning curves and the test measurements used for the final physical control comparison.

#### Inspect the configuration before running

Display the pipeline configuration for one seed:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py describe --seed 17
```

To inspect the planned run without starting training or writing pipeline files, use:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --seed 17 --dry-run
```

#### Run all five seeds

Omit `--seed` to run the complete principal comparison:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --workers 4
```

The pipeline processes the seeds sequentially. The worker count determines the parallelism of data preparation, teacher generation, and control evaluation. `RecurrentPPO` training uses the four environments specified in the experiment configuration.

The `--validation-workers` argument specifies the number of worker processes for EPANET validation of the MPC teacher demonstrations and defaults to four. The `--device` argument selects the execution device for teacher generation and behavior cloning and defaults to `auto`. Use `--device cpu` or `--device cuda` to select a device explicitly. Forecasting and PPO training use the device settings defined in their experiment configurations.

#### Run selected seeds

Run the complete pipeline for one seed:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --seed 17 --workers 4
```

Repeat `--seed` to run several seeds:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --seed 17 --seed 42 --workers 4
```

The complete paper export requires all five study seeds. A single-seed execution generates the corresponding data and models and allows verification of the procedure before running the full comparison.

#### Run individual stages

Use `--stage` to select individual stages. The argument can be repeated. For example, train the forecasting models for one seed:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --seed 17 --stage forecasting --workers 4
```

After forecasting is complete, generate the teacher datasets and train the actor through behavior cloning:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --seed 17 --stage mpc --stage behavior-cloning --workers 4
```

Then train and evaluate the baseline and pretrained policies:

```bash
.venv/bin/python training/run_principal_v89_pipeline.py run --seed 17 --stage baseline --stage fine-tuning --workers 4
```

Selected stages execute in pipeline order and require their preceding dependencies to be complete for the same seed. Executing all five stages separately also produces the inputs required by the paper exporter.

The pipeline resumes training from compatible forecasting and PPO checkpoints and reuses completed teacher datasets and behavior-cloning bundles. Interrupted runs can therefore be resumed by repeating the relevant stage command. The complete paper exporter also validates the stage timing records and can reject interrupted or repeated stages. Reproducing the computational-cost table requires a fresh, uninterrupted execution of each stage.

The pipeline stores configuration plans, execution status, and stage timing records in `data/pipelines/hanoi-principal-v89/seed-<seed>/`. Trained models and evaluation outputs are stored under `data/weights/`. The displayed configuration plans specify the directories associated with each run.

### 4. Export the principal results

After the principal pipeline has completed for all five seeds, validate the saved experiment outputs:

```bash
.venv/bin/python tools/analysis/export_principal_v89_paper_results.py --results-root data --check-only
```

Then export the data read by the paper:

```bash
.venv/bin/python tools/analysis/export_principal_v89_paper_results.py --results-root data
```

The exporter processes the recorded validation returns, selected checkpoints, test evaluations, forecast errors, and preparation times to generate the following files in `paper/results/`. Export uses the existing measurements without repeating training or simulation.

| Output                    | Data read by the paper                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `learning-plot-data.json` | Validation learning curves for both approaches and all five seeds.                             |
| `control-data.json`       | Selected-policy physical metrics, paired comparisons, and the selected illustrative test case. |
| `forecast-plot-data.json` | Forecast errors across seeds, folds, and prediction horizons.                                  |
| `computation_costs.csv`   | Shared preparation and per-seed stage runtimes.                                                |
| `package_metadata.csv`    | Metadata identifying the exported experiments.                                                 |

To prepare the exploratory analysis of chlorine excess after clearance, use the saved principal test evaluations:

```bash
.venv/bin/python tools/analysis/export_principal_v89_post_cluster.py --results-root data
```

The script writes `paper/results/post-cluster-data.json`. The export scripts default to `data/`. For experiment outputs stored elsewhere, set `--results-root` to the directory containing `pipelines/`, `weights/`, `datasets/`, and `materialization/`. Exported run paths are relative to this directory.

### 5. Additional variant comparisons

The thesis compares two additional variants. The variant without actor weights receives the observation-normalization state and initial action standard deviation from behavior cloning, while its actor remains ordinarily initialized. It therefore requires the completed behavior-cloning bundles:

```bash
.venv/bin/python training/run_control_v89_ablations.py run --variant without-actor-weights --workers 4
```

As in the principal pipeline, omitting `--seed` runs the variant for all five study seeds. Add `--seed 17`, for example, to select a single seed.

The extended-budget baseline resumes training from earlier checkpoints to reach four million nominal interactions. These checkpoints are not included in the submission and are not generated by the principal pipeline above. Reproducing this variant therefore requires the earlier experiment outputs. The data for both submitted validation comparisons are included in `paper/results/variant-learning-plot-data.json`. Regenerating this file also requires those outputs.

### 6. Regenerate the dataset illustrations

The Dataset chapter illustrates selected scenarios and simulation runs. The following commands reproduce these selections and export the figure data to `paper/dataset/`.

Export the example demand pattern and the selected offline simulation:

```bash
.venv/bin/python tools/analysis/dataset/export_demand_patterns.py scenario-0 --week 2
.venv/bin/python tools/analysis/dataset/export_simulation_run.py scenario-85 --seed 6104 --chlorine-seed 7104 --arsenic-index 4 --chlorine-index 3
```

The simulation command uses the arsenic and chlorine seeds assigned to the validation partition and selects the schedule-profile pair illustrated in the paper. In this command, `--seed` specifies the arsenic schedule set rather than the policy-training seed.

Generate the three arsenic illustrations from the fixed control-validation cases simulated without chlorine:

```bash
.venv/bin/python tools/analysis/dataset/export_arsenic_examples.py --workers 4 --mild-run 46 --strong-run 19 --overlap-run 23
```

Then generate the event statistics and reference-window illustration:

```bash
.venv/bin/python tools/analysis/dataset/export_arsenic_statistics.py --workers 4
.venv/bin/python tools/analysis/dataset/export_reference_window_example.py
```

These commands write `demand-patterns.json`, `simulation-run.json`, `arsenic-examples.json`, `arsenic-statistics.json`, and `reference-window-example.json`. Run the statistics export before the reference-window export, which reads the event catalog in `arsenic-statistics.json`. The arsenic examples and statistics reuse the saved zero-chlorine trajectories under `data/runs/`.

### 7. Regenerate the network-delay analysis

The network-delay analysis examines the timing of chlorine responses using the hydraulic training scenarios. It supports the choice of forecasting history and horizon described in the Appendix and can be run independently of policy training.

Run the complete analysis and export the data read by the paper:

```bash
.venv/bin/python tools/analysis/analyze_network_delays.py --workers 4
.venv/bin/python tools/analysis/export_network_delay_paper_summary.py
```

The first command stores the analysis outputs in `data/analysis/network-delays/low-source-outflow/`, and the second generates `paper/results/network-delay-summary.json`. The export requires the complete analysis with the default scenario and start-time coverage.

### 8. Regenerate the illustrative control trajectories

The case illustrated in the Results chapter is specified by the `selected` entry in `paper/results/control-data.json`. It uses `scenario-90`, arsenic schedule index 8, and the policies trained with seed 17.

Before simulating this case, validate the saved experiment inputs:

```bash
.venv/bin/python tools/analysis/mpc/run_principal_v89_case.py --data-root data --seed 17 --scenario scenario-90 --schedule-index 8 --check-inputs
```

Then generate the matched controller rollouts in a new or empty output directory:

```bash
.venv/bin/python tools/analysis/mpc/run_principal_v89_case.py --data-root data --seed 17 --scenario scenario-90 --schedule-index 8 --output data/analysis/principal-v89-paper-case
```

This command performs additional simulations, including an MPC rollout with repeated replanning. After the rollouts are complete, export their time series:

```bash
.venv/bin/python tools/analysis/export_principal_v89_case_trajectories.py --rollout-dir data/analysis/principal-v89-paper-case
```

The exporter writes `paper/results/case-trajectory-data.json` and verifies that the scenario and schedule match the case specified by the principal export. After exporting new principal results, regenerate this file before compiling the paper to maintain consistency between the aggregate results and the illustrated case.

### 9. Repeat the parameter-selection experiments

The principal pipeline uses the configuration selected for the study. The preceding parameter search can be reproduced independently. To execute the four search rounds documented in the Appendix, use:

```bash
.venv/bin/python training/experiments/control_v89_hpo.py run --workers 4
```

To run one round, select it with `--round`, for example:

```bash
.venv/bin/python training/experiments/control_v89_hpo.py --round 3 run --workers 4
```

Rounds 1 to 3 compare PPO parameter settings, while round 4 compares reward parameter settings. All rounds use training seed 42 and store their outputs separately from the principal comparison. The parameter-selection tables are maintained in the Appendix source and are not regenerated by the principal results exporter.

### 10. Compile the paper and run targeted checks

The paper reads the included JSON and CSV inputs directly and can therefore be compiled without rerunning the experiments.

The submitted PDF was compiled with a locally built version of Typst 0.15.1, build `094b9634`. This version supports the figure-boundary setting used by the paper. With this compiler available as `typst`, compile the paper with:

```bash
typst compile --input figure-boundary=true paper/main.typ thesis.pdf
```

The compiler executable is not included in the repository. When using Typst without the figure-boundary extension, omit the input argument. This leaves the conditional boundary rule in `paper/main.typ` disabled:

```bash
typst compile paper/main.typ thesis.pdf
```

Using another compiler build or disabling this rule may change figure placement, layout, and pagination. The first compilation also requires access to the Typst packages used by the paper unless they have already been downloaded.

The included tests cover the final reward, principal pipeline, teacher generation, behavior cloning, and calculations used in the paper exports. For example, execute the following unit tests:

```bash
.venv/bin/python -m unittest tests.test_reward_v89 tests.test_principal_v89_pipeline tests.test_paper_results_projection tests.test_v89_ablation_paper_export
```

The unit tests verify individual calculations and components of the training and evaluation procedure. Validation of newly generated experiment results additionally requires completion of the corresponding training and simulation runs and inspection of their outputs.

Tests that read saved principal experiment outputs use `data/` by default. Set `THESIS_RESULTS_ROOT` to use a different directory for these tests.

## License

The original contributions by Amadeus Beckmann are licensed under the [MIT License](LICENSE). These include the code, thesis text, documentation, and original figures.

The Hanoi arsenic-contamination model at `external/Environment/hanoi_arsenic_contamination.msx` is also licensed under MIT. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records its attribution.
