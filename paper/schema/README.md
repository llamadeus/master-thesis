# Results data contract

This directory defines the detailed Results artifact contract and the compact publication inputs. The paper reads only the publication inputs described below.

The contract is versioned as `1.1.0` and consists of:

- `datapackage.json`, the machine-readable Frictionless Data package schema for the CSV files
- `metric-catalog.json`, the canonical metric names, directions, units, components, and derivations
- this document, which defines cross-table rules that cannot be expressed by column types alone

No result values are stored in this directory. Only compact publication inputs are written to `paper/results/`. Detailed schema-compliant CSV packages belong in a separate artifact directory.

## Scope

The schema covers every result family specified by the Methodology and Evaluation chapters:

1. validation reward learning curves at 0 and every 50,000 interactions through 1,000,000 nominal online interactions
2. the six primary physical metrics on the final test checkpoint
3. paired differences between the baseline and pretrained policies across five training seeds
4. policy-independent zero-injection references and metric applicability
5. main-window, tail-window, regime, scenario, schedule, seed, and node-level breakdowns
6. selected time-resolved trajectories for explanatory figures
7. reusable offline preparation costs, PPO costs, and evaluation costs
8. forecast MAE and RMSE for five forecasters per training seed, ten test scenarios, and sixteen forecast steps

The schema deliberately excludes Discussion-level interpretation. It supplies facts such as values, uncertainty, direction, and applicability. Claims about causes, trade-offs, importance, or generalizability remain authored prose.

## Data flow

```text
experiment
  -> validated detailed measurements (optional CSV artifact package)
  -> schedule/scenario aggregation during export
  -> compact publication inputs in paper/results/
  -> tables, plots, values, and descriptive Results sentences
```

Generated inputs must not be edited by hand. The measured exporters described below regenerate publication inputs from experiment artifacts.

## Compact publication inputs

| File | Data used by the paper |
| --- | --- |
| `package_metadata.csv` | Package identity and synthetic/experimental provenance |
| `computation_costs.csv` | Individual stage runtimes, device and worker information |
| `learning-plot-data.json` | All 210 policy/seed/checkpoint returns and nominal/actual interactions |
| `control-data.json` | 1,300 final policy/seed/scenario/metric-component means, applicability counts, 100 zero-reference burdens, and the 13 displayed selected-case values |
| `forecast-plot-data.json` | All 25 model errors for the full horizon and each of 16 forecast steps (425 rows) |
| `post-cluster-data.json` | Reference-aligned contributions to normal chlorine excess; retained policy-aligned curves and cluster counts |

The control and forecast JSON contracts carry `format_version = 1` and `data_kind`. Learning data also carries `data_kind`. The paper checks that its inputs share the metadata's data kind.

For control, `rows` retain `approach`, `seed`, `scenario`, `metric`, `component`, `value`, `applicable_count`, and `not_applicable_count`. Applicable schedule values are averaged within each scenario. An entirely non-applicable scenario has a null value and explicit counts. Typst then gives applicable scenarios equal weight within each seed and calculates seed means, sample standard deviations, paired differences, and confidence intervals. Fractions are already complete rollout fractions before schedule averaging, so this projection does not pool different schedules' denominators. The `selected` object records the fixed case IDs, selection reason, and displayed metric values. `reference_values` preserves the case-level zero-injection burdens used in the chapter.

For forecasting, each row identifies `training_seed`, `fold_id`, and `horizon_step`, with `mae` and `rmse`. Step zero denotes the full horizon. Error sums and observation counts are first pooled over the chosen steps within a scenario. Scenario MAE and RMSE are then averaged with equal scenario weights. Full-horizon RMSE is computed independently from the step-wise errors, not by averaging step-wise RMSE. Typst still computes fold means within seeds and statistics across seeds.

The exploratory post-cluster input (format version 2) is regenerated from saved validation-selected final test trajectories by `tools/analysis/export_principal_v89_post_cluster.py`. Every cluster is aligned to its persistent transported-arsenic clearance, constructed from that policy's trajectory and the original source schedule using the existing threshold criterion. Rollouts without reference-event windows or without policy-specific clusters are excluded. Offsets 0 through 24 include clearance itself and up to 12 hours of follow-up, ending before the next original source injection or at episode end. At each offset, the fractions of monitored nodes strictly above 0.4 mg/L chlorine are averaged over available clusters within rollouts, then schedules within scenarios, scenarios within seeds, and seeds. `observation_count` is the unweighted number of available cluster fractions across all rollouts and seeds; `rollout_count` separately counts contributing rollouts. Observed zero fractions count as samples. Absent follow-up contributes neither a zero fraction nor an observation; an offset with no observations has a null mean and zero count. The input retains eligibility and cluster counts by policy and seed.

The paper uses `excess_partition`, which decomposes the primary `normal_chlorine_excess_fraction` metric into contributions inside and outside common 12-hour intervals after zero-chlorine reference-cluster clearance. `excess_partition_alignment` is `reference_cluster_clearances`. Each interval includes clearance and excludes its 12-hour endpoint, ending before the next source injection. Both contributions use only reference-normal samples in the main window, so follow-up is truncated at the main-window boundary. Reference-event windows, warm-up, and the clearance tail are excluded. Each eligible above-threshold node--time observation is counted once. For each rollout, both counts are divided by the same reference-normal observation denominator. The resulting contributions are averaged over schedules within scenarios, scenarios within seeds, and seeds, matching the primary table. `inside_metric_fraction` and `outside_metric_fraction` sum to `total_metric_fraction`; the reported shares divide these contributions by their sum. Raw counts are also retained, separately by seed. Rollouts without reference clusters can contribute only to the outside component. The retained policy-aligned curves are no longer displayed in the paper.

The matched-trajectory figure reads `case-trajectory-data.json`, which contains the selected case's controller trajectories, source events, and zero-chlorine reference windows. It is generated separately by `tools/analysis/export_principal_v89_case_trajectories.py` from completed matched rollouts. The optional detailed CSV export retains node and rollout time series for further analysis.


## Detailed artifact tables

| File | Row grain | Purpose |
| --- | --- | --- |
| `package_metadata.csv` | one row per package | Distinguishes synthetic and experimental data and records provenance |
| `checkpoints.csv` | one row per trained-policy checkpoint or zero-injection reference | Identifies approach, seed, pairing, and nominal and actual interaction counts |
| `evaluation_cases.csv` | one row per scenario and arsenic-schedule combination | Defines validation and test cases and their policy-independent eligibility counts |
| `source_events.csv` | one row per source event in an evaluation case | Supports event-aligned plots and schedule checks |
| `reference_windows.csv` | one row per zero-reference event cluster | Defines the shared event and post-source intervals applied to both policies |
| `rollouts.csv` | one row per checkpoint and evaluation case | Links a policy artifact to one deterministic evaluation replay |
| `rollout_metrics.csv` | one row per rollout, metric, and component | Stores the sufficient statistics used for learning curves and physical comparisons |
| `node_metrics.csv` | one row per rollout, node, metric, and component | Supports spatial summaries and reveals aggregate results dominated by particular nodes |
| `rollout_timeseries.csv` | one row per selected rollout and timestep | Supports aligned injection, reward, event-window, and network-summary plots |
| `node_timeseries.csv` | one row per selected rollout, timestep, and node | Supports explanatory chlorine and transported-arsenic trajectories |
| `computation_costs.csv` | one row per measured stage execution | Separates reusable offline preparation, repeated PPO work, and evaluation overhead |
| `forecast_error_statistics.csv` | one row per training seed, fold, test scenario, and horizon step | Stores sufficient statistics for scenario-equal MAE and RMSE |

## Identifiers and joins

Identifiers are opaque strings. Their text must not be parsed to recover scientific variables.

- `checkpoint_id` identifies one approach, seed, and online-interaction checkpoint.
- `case_id` identifies one evaluation split, hydraulic scenario, and fixed arsenic schedule.
- `rollout_id` identifies one checkpoint evaluated on one case.
- `pairing_key` links the baseline and pretrained checkpoint trained with the same seed and matched training conditions.
- `reference_window_id` identifies one policy-independent reference-event cluster within a case.

The explicit columns remain authoritative even if identifiers contain readable fragments.

## Expected principal-study coverage

The package validator must enforce the following study design in addition to the column schema:

- training seeds are `17`, `42`, `123`, `1337`, and `71026`
- validation contains ten hydraulic scenarios and five schedules per scenario, giving 50 cases
- test contains ten hydraulic scenarios and ten schedules per scenario, giving 100 cases
- baseline and pretrained validation rollouts exist at nominal checkpoints 0 and every 50,000 interactions through 1,000,000
- baseline and pretrained test rollouts evaluate the checkpoint selected by the highest mean validation return within the full nominal training budget
- every baseline checkpoint has exactly one pretrained checkpoint with the same split-independent pairing key, seed, and nominal checkpoint
- every case has exactly one zero-injection reference rollout, which is reused across policies, seeds, and checkpoints
- validation checkpoints contain 50 completed rollouts per approach and seed
- the selected test checkpoint contains 100 completed rollouts per approach and seed

`nominal_online_interactions` and `actual_online_interactions` are separate because the Methodology chapter describes a nominal one-million-step budget while the rollout implementation may finish the current PPO batch. Each nominal threshold uses the first completed PPO update at or above that threshold. Thresholds remain absolute, and plots use actual interaction counts. After training through the full budget, final testing evaluates the checkpoint with the highest mean validation return; the earliest checkpoint breaks ties.

## Episode and reference-window invariants

Each evaluation case has 1,200 half-hour samples:

- 96 warm-up steps
- 1,008 main-window steps
- 96 clearance-tail steps

The warm-up is never scored. Return includes the complete main window and tail, including reference-normal tail samples, without resetting policy or reward state. Physical metrics retain their individual gates. Reference-event windows are constructed once from the zero-injection replay and reused unchanged for every policy rollout of that case. The following equalities must hold:

```text
main_normal_steps + main_event_steps = main_steps
0 <= main_post_source_steps <= main_event_steps
0 <= tail_post_source_steps <= tail_event_steps <= tail_steps
```

Reference windows use inclusive starts and exclusive ends. Source events use the same convention. A source event may exist without producing a reference-event window. Multiple source events may belong to one reference window when persistent clearance does not occur between them.

## Metric rows

The authoritative metric definitions are in `metric-catalog.json`. Every completed baseline and pretrained rollout must contain the following rows:

| Metric | Required components | Primary component |
| --- | --- | --- |
| `chlorine_shortfall_fraction` | `main_normal` | `main_normal` |
| `normal_chlorine_excess_fraction` | `main_normal` | `main_normal` |
| `arsenic_burden` | `main`, `tail`, `total` | `total` |
| `chlorine_ceiling_violation_fraction` | `main`, `tail`, `total` | `total` |
| `sustained_chlorine_excess` | `main`, `tail`, `total` | `total` |
| `chlorine_use` | `main_normal`, `main_event`, `tail_event`, `total` | `total` |
| `evaluation_return` | `total` | `total` |

The zero-injection reference must contain at least the `main`, `tail`, and `total` rows for `arsenic_burden`. Additional zero-reference metrics may be retained for diagnostics, but they do not turn the zero-injection profile into a trained-policy baseline.

### Fraction metrics

Fraction rows retain `numerator_count` and `denominator_count`. The reported value is derived rather than independently trusted:

```text
value = numerator_count / denominator_count
```

If the denominator is positive, `applicable` is true and `value` must be present. If the denominator is zero, `applicable` is false and `value` must be empty. A positive denominator with a zero numerator is an applicable value of zero.

For the total chlorine-ceiling fraction, main and tail counts are pooled:

```text
total numerator   = main numerator + tail numerator
total denominator = main denominator + tail denominator
total value       = total numerator / total denominator
```

Main and tail fractions must never be averaged or added.

### Additive metrics

Additive rows are always applicable. If their gate selects no timesteps, their value is zero. Component checks are:

```text
arsenic_burden.total
  = arsenic_burden.main + arsenic_burden.tail

sustained_chlorine_excess.total
  = sustained_chlorine_excess.main + sustained_chlorine_excess.tail

chlorine_use.total
  = chlorine_use.main_normal
  + chlorine_use.main_event
  + chlorine_use.tail_event
```

`evaluation_return.total` is the undiscounted sum of original, unnormalized rewards over scored samples after warm-up. It is the only control metric for which a higher value is favorable. Lower values favor pretraining for each of the six primary physical metrics.

### Node decomposition

`node_metrics.csv` covers the five node-decomposable physical metrics. Chlorine use and evaluation return are not assigned to nodes. For any rollout and component:

- node numerators and denominators must sum to the corresponding rollout counts
- node arsenic burdens must sum to the rollout arsenic burden
- node sustained-excess values must sum to the rollout sustained excess

This table is required for the final test checkpoint. It may also be emitted for validation checkpoints when useful.

## Hierarchical aggregation

The analysis must preserve the experimental unit defined in the Evaluation chapter.

For each policy, training seed, metric, and checkpoint:

1. calculate the metric once per rollout
2. omit only non-applicable fraction rows
3. average applicable schedules within each hydraulic scenario
4. average the applicable scenario summaries with equal scenario weight
5. retain the resulting single seed-level value

Across the five seeds, report the arithmetic mean and sample standard deviation. The paired effect is calculated at seed level:

```text
paired difference = pretrained - baseline
```

Report the mean paired difference and its two-sided 95% Student-t confidence interval with four degrees of freedom. Negative differences favor pretraining for all six primary physical metrics. Do not treat evaluation cases as independent training replicates.

Every aggregate must carry the number of applicable rollout values, non-applicable rollout values, applicable scenarios, and seeds. This information is derived from the source rows and must not be supplied as an unverified constant.

## Learning curves

Learning curves use `evaluation_return.total` from validation rollouts. Schedules are averaged within scenarios, scenarios receive equal weight within a seed, and the five seed summaries determine the displayed mean and standard deviation.

The schema records both approaches at zero interactions. The pretrained zero checkpoint represents the behavior-cloned initialization before PPO updates. The baseline zero checkpoint represents the ordinarily initialized policy before PPO updates.

The schema supports only descriptive checkpoint comparisons. It does not infer a continuous convergence time or a sample-efficiency threshold between evaluated checkpoints.

## Forecast aggregation

`forecast_error_statistics.csv` stores sums and counts instead of pre-averaged errors. This prevents accidental averaging of RMSE values at the wrong level.

For each fold, scenario, and horizon step:

```text
MAE  = absolute_error_sum_mg_l / observation_count
RMSE = sqrt(squared_error_sum_mg2_l2 / observation_count)
```

The ten scenario values then receive equal weight within each fold. Full-horizon errors are calculated by first pooling the sixteen step-level sums and counts within each fold and scenario, then applying the formulas above. Each of five training seeds has five fold-specific forecasters. Average their model errors within each seed, then report the mean and sample standard deviation across the five seed means. Preserve training_seed and fold_id separately. These are errors of individual models, not ensemble-prediction errors.

Forecast results are kept separate from the control-policy comparison. They describe predictive accuracy on unseen hydraulic scenarios and do not establish MPC-teacher or closed-loop control quality.

## Computational cost

Every cost row states whether it is reusable once, repeated per training seed, repeated per forecasting fold, or repeated per evaluation. Offline trajectory generation, forecaster training, MPC teacher generation, behavior cloning, baseline PPO, pretrained PPO, control evaluation, and forecast evaluation remain separate stages.

Total-cost views are derived from these rows. The analysis must not multiply reusable artifacts by the number of policy seeds, and it must not present reduced online interactions as reduced total computation without including offline preparation.

MPC training and validation generation use the direct component timers from the latest completed generation. Historical aggregate timers, interrupted attempts, and resume-only checks are excluded. The `mpc_teacher_fold_generation` rows retain the five fold measurements per seed, including their replay checks. They are components of `mpc_teacher_training_generation`, which also includes merging, and must be excluded from additive totals.

## Time-resolved supporting evidence

Full raw trajectories can remain in experiment artifacts referenced by `raw_artifact_uri`. The two time-series CSVs contain only rollouts selected for Results figures or adverse-tail inspection. `timeseries_role` and `timeseries_selection_reason` make the selection rule visible.

For every selected rollout:

- `rollout_timeseries.csv` contains all 1,200 episode steps
- `node_timeseries.csv` contains all monitored nodes at every included step
- elapsed time advances in 0.5-hour increments
- window, regime, source, and post-source flags agree with the case-level reference windows
- aggregate chlorine and transported-arsenic values agree with the node-level rows

Selection is explanatory rather than a new source of statistical replication.

## Synthetic and experimental packages

Synthetic fixtures set `data_kind` to `synthetic`. Experimental packages set it to `experimental`. No downstream table, figure, or sentence generator may silently combine the two kinds.

The experimental exporter must satisfy the key, cardinality, component, applicability, and aggregation invariants in this contract.

Before final-paper generation, the build must reject `data_kind = synthetic` unless an explicit draft mode is enabled.

## Questions the schema can answer

The normalized tables support the reader-facing questions anticipated by the Results outline:

- Did pretraining alter initial and subsequent validation return?
- Did the difference persist, narrow, or reverse across the fixed checkpoints?
- How did the policies compare on every primary physical metric at the final budget?
- Were apparent aggregate differences consistent across training seeds?
- Which cases were not applicable, and was applicability identical for paired policies?
- Did normal-operation shortfalls improve at the cost of normal-operation excesses?
- Was lower arsenic burden accompanied by ceiling violations or sustained chlorine excess?
- How much of an event-response difference occurred in the main window or clearance tail?
- How did policy burdens compare with the untreated zero-injection reference?
- Was lower chlorine use accompanied by preserved physical control quality?
- Were aggregate differences concentrated in particular scenarios, schedules, or nodes?
- What do representative matched trajectories show at the times responsible for an aggregate result?
- How large were reusable offline costs relative to PPO training and evaluation?
- How did forecast error change over the sixteen-step horizon and across folds and training seeds?

These questions define the boundary of the automated Results pipeline. Explanations of why a difference occurred and what it means belong in the Discussion chapter.

## Measured evaluation export

`tools/analysis/export_evaluation_results.py` exports protocol-v2 runner outputs without starting training. Use `--control-run APPROACH SEED DIRECTORY` for each of the ten policy runs and `--forecast-run DIRECTORY` for each of the 25 forecasting runs, with `--output-dir` pointing to a separate measured-results directory. It writes case-level physical values and applicability counts, learning curves with nominal and actual interactions, and forecast sufficient statistics with both seed and fold. The native Typst/Primaviz figures in `paper/chapters/07_results/evaluation-plots.typ` read `forecast-plot-data.json` and `learning-plot-data.json` from `paper/results/`. These files retain precomputed model/horizon errors and individual seed returns. Typst computes seed means and sample standard deviations and draws both figures directly. Detailed CSVs and summary files stay in the separate measured-results directory. Copy the measured JSON inputs into `paper/results/` when ready. `--placeholder` writes empty inputs without fabricating observations. No SVG plots are generated. Legacy results without the new protocol and aggregation metadata are rejected.

Raw As(III) clearance is checked for every fixed validation and test case within the existing 25 days. At least one tail report must begin a suffix whose summed nodal concentration stays at or below 0.01 mg/L through the final report. The final report alone can satisfy this check. Failure keeps the evaluation incomplete and retains the case identity. Transported-arsenic correction and event windows remain separate from this prerequisite.
