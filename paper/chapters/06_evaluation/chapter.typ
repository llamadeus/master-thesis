#import "../../numbers.typ": *
#import "@preview/acrostiche:0.7.0": acr
#import "@preview/akatable:0.1.0": academic-table
#import "../../utils.typ": continued-tables


= Evaluation <chapter:evaluation>

This chapter presents the evaluation of the two control approaches introduced in the previous chapter in order to address the research question investigated in this thesis.

For this comparison, we refer to the approach trained without pretraining as the baseline approach and to the approach initialized through pretraining as the pretrained approach. Both follow the common online training conditions.

The evaluation uses reward learning curves to assess learning progress during online training and six primary physical metrics to compare the control quality of the policies selected by validation return after the common training budget has been completed. Computational cost is assessed separately, including offline preparation, because fewer online interactions do not necessarily imply less computation overall. We first establish common evaluation cases, episode windows, and policy selection, then define how learning progress, final control quality, and computational cost are evaluated. Additional variant comparisons and a supporting evaluation of the forecasting model complement these primary analyses.


== Evaluation Protocol

A fair comparison requires both policies to encounter the same evaluation conditions. The protocol establishes which parts of each episode contribute to scoring and how the policies are evaluated across fixed cases and repeated training runs.


=== Episode Structure <section:evaluation:episodes>

Each evaluation rollout combines one trained policy with one hydraulic scenario and one fixed arsenic schedule. The 25-day episodes defined in @section:dataset:simulation-setup are divided into a two-day warm-up, a 21-day main window, and a two-day clearance tail. For the remainder of this chapter, we denote the timesteps in the main window and clearance tail by $cal(T)_"M"$ and $cal(T)_"T"$, respectively. We write $cal(T)_w$ for either set with $w in {"M","T"}$.

The warm-up gives the policy time to establish chlorine concentrations throughout the network and is excluded from scoring. The main window provides an extended observation period with and without contamination. Its duration is a practical choice rather than a requirement of the evaluation method. The clearance tail allows us to evaluate the policy response while remaining arsenic clears from the network. The evaluation return, defined in @section:evaluation:learning-curves, sums rewards over the main window and complete clearance tail, excluding the warm-up. @fig:episode-timeline illustrates this division.

The clearance tail captures responses to contamination events occurring near the end of the main window. Although all arsenic inputs stop before the end of the 21-day main window, transported arsenite can still remain in the network and affect chlorine levels and, consequently, the control behavior of the policy. The two additional arsenic-injection-free days allow these effects to be evaluated until a common clearance boundary.

Before policy evaluation, all fixed validation and test cases were checked using reference runs with a zero-chlorine injection profile. The clearance check applies a threshold of #mg-l[0.01] to the sum of raw #acr("As(III)") concentrations across all monitored nodes. This numerical value is taken from the #acr("WHO")'s provisional guideline for arsenic in drinking-water @who2026guidelines. Its application to a concentration sum serves as an operational clearance criterion rather than a measure of compliance with that guideline. In every case, the raw concentration sum reached or fell below this threshold during the clearance tail and remained at or below it until the end of the episode. These checks confirmed that the two-day clearance tail was sufficient under this criterion. The criterion concerns the monitored nodes within the observed horizon and does not imply the complete absence of #acr("As(III)") from the network.

#figure(
  include "episode-timeline.typ",
  placement: none,
  caption: [Episode windows in a 25-day evaluation rollout. The arsenic-injection-free warm-up is excluded from scoring. Arsenic events can only start in the main window. Validation return covers the main window and full clearance tail. Physical control metrics are evaluated as indicated.],
) <fig:episode-timeline>


=== Policy Comparison and Evaluation Procedure <section:evaluation:policy-selection>

Both policies are evaluated on the same fixed sets from @section:dataset:splits. Each set comprises ten scenarios, with five arsenic schedules per validation scenario and ten per test scenario. The 50 validation cases track progress during training and determine which policy checkpoint is selected. The 100 test cases provide the physical comparison of the selected policies. Within each case, hydraulics, demand timing, initial conditions, and the arsenic schedule are identical for the two policies. For each case, a single reference run without chlorine injection is evaluated to quantify arsenic contamination in the absence of dosing and compare it with contamination under each policy.

Five training runs per approach use the seeds #identifier[17], #identifier[42], #identifier[123], #identifier[1337], and #identifier[71026]. Model fitting is repeated for each training seed, with shared offline trajectories and fixed evaluation cases. Corresponding runs encounter the same sequence of hydraulic scenarios and contamination schedules during training, following the sampling procedure described in @section:methodology:training-scenario-and-contamination-sampling. Pairing training runs by their seed allows for comparison under matched environmental conditions.

Both approaches complete the nominal one-million-interaction budget. For each approach and training seed, the checkpoint with the highest mean validation return is then selected from the 21 checkpoints defined in @section:evaluation:learning-curves, including the zero-interaction checkpoint and the checkpoint at the nominal one-million-interaction threshold. The mean gives equal weight to scenarios after averaging their schedule-level returns. If several checkpoints attain the same maximum, the earliest is selected. Both approaches use the same candidate checkpoints, validation cases, and selection criterion.

During evaluation, action selection is deterministic, policy memory is reset at the start of each rollout, the observation-normalization statistics saved with the evaluated checkpoint remain fixed, and reward normalization is disabled. Evaluation follows the policy-input restrictions defined in @section:methodology:policy-observations, with arsenic measurements used only for scoring.


== Learning Curves and Sample Efficiency <section:evaluation:learning-curves>

Training efficiency is assessed through reward learning curves, which track control performance under the common objective during online training. They show whether the policies differ in performance before online training and how their relative performance evolves during training.

The learning curves compare validation return before the first #acr("PPO") update and at nominal thresholds of #num[50000], #num[100000], #num[150000], and every further #num[50000] interactions up to #num[1000000]. At each threshold, we evaluate the first completed #acr("PPO") update whose cumulative interaction count reaches or exceeds it. This rule applies to both approaches and all five training seeds. Actual interaction counts are recorded for each checkpoint. At zero interactions, the pretrained policy has completed behavior cloning, and neither policy has begun online training. The initial comparison measures the control behavior supplied by pretraining, while subsequent checkpoints show how the difference develops over time.

Each checkpoint is evaluated on the same 50 validation cases using the procedure above. Interaction counts include all parallel training environments and exclude evaluation interactions. Using the reward function defined in @section:methodology:reward-function, the evaluation return for each rollout is the undiscounted sum of the original, unnormalized rewards over the main window and clearance tail,

$ R_("eval") = sum_(t in cal(T)_("eval")) r_t. $ <eq:evaluation-return>

Here $cal(T)_("eval") = cal(T)_"M" union cal(T)_"T"$, using the window sets defined in @section:evaluation:episodes. The reward definition and scoring duration are identical for both policies. Higher return indicates better performance under this objective. On the validation cases used to track learning progress, we refer to this quantity as the validation return.

At each checkpoint, returns are averaged across schedules within each scenario and then across scenarios for each seed. The mean and sample standard deviation are calculated across the five seeds. Pretrained-minus-baseline return differences are calculated within matched seed pairs and then averaged.

Because performance is evaluated only at discrete checkpoints, changes between checkpoints cannot be located more precisely, and apparent plateaus do not establish convergence beyond the evaluated budget. The learning curves are interpreted together with the final physical metrics to assess whether faster progress under the training objective is accompanied by acceptable control quality.


== Control Performance Metrics

The evaluation uses six primary control metrics over the main and tail windows introduced in @section:evaluation:episodes.

Two metrics assess chlorine compliance during normal operation. Arsenic burden, mean arsenic clearance lead, and chlorine ceiling violations characterize event response, while chlorine use measures chemical expenditure during both normal operation and event response. @tab:evaluation-metric-windows provides an overview of the evaluation windows and reference regimes used by the six primary metrics.

#{
  set figure(placement: none)
  set par(justify: false)

  academic-table(
    [Evaluation windows and reference regimes used by the six primary control metrics. Clearance refers to persistent clearance in the zero-chlorine reference. The regime follows the definition introduced in @section:evaluation:reference-windows.],
    header: ([*Metric*], [*Main window*], [*Tail window*], [*Regime*]),
    (
      [Chlorine shortfalls],
      [Yes],
      [No],
      [Reference-normal],

      [Chlorine excesses during normal operation],
      [Yes],
      [No],
      [Reference-normal],

      [Arsenic burden],
      [Yes],
      [Until clearance],
      [Reference-event],

      [Mean arsenic clearance lead],
      [Yes],
      [Until clearance],
      [Reference-event],

      [Chlorine ceiling violation],
      [Yes],
      [Until clearance],
      [Reference-event],

      [Chlorine use],
      [Yes],
      [Until clearance],
      [Both regimes],
    ),
    format: "booktabs",
    columns: (auto, auto, auto, 1fr),
    align: (left, center, center, center),
    label: <tab:evaluation-metric-windows>,
  )
}


=== Reference-Event Windows <section:evaluation:reference-windows>

The two policies must be scored over the same periods even if they clear arsenic at different rates. Reference-event windows are therefore determined from the zero-chlorine replay of each case and applied unchanged to both policies at every checkpoint. A window begins when the transported arsenic aggregate first exceeds the clearance threshold and ends with persistent clearance which is caused by outflow and dilution. The zero-chlorine replay provides the conservative upper bound on #acr("As(III)") persistence, so every policy has cleared the event by this shared endpoint at the latest. Times outside these windows form the reference-normal regime.

The arsenic measure used to construct the windows is separate from the raw-concentration clearance check in @section:evaluation:episodes and accounts for the local mixing behavior described in @section:dataset:simulation-setup. In this simulation model, chlorine reacts with arsenite only within pipes, so increasing chlorine injection cannot neutralize newly injected arsenite at the source node of the event itself. Including this immediate local contribution would therefore add a burden that the controller cannot reduce through chlorination. The evaluation excludes this contribution while continuing to account for arsenite transported through the network. At each monitored node $v$ and timestep $t$, the estimated contribution of newly injected arsenic is subtracted from the reported #acr("As(III)") concentration, and the remainder is clipped at zero. The calculation is specified in @section:appendix:transported-arsenic. Denoting this transported concentration by $a_(v,t)$ gives the aggregate

$ A_(Sigma) (t) = sum_(v=1)^N a_(v,t). $ <eq:evaluation:transported-arsenic>

Clearance occurs when $A_(Sigma) (t) <= epsilon$, with $epsilon$ = #mg-l[0.01]. This is the same operational threshold used for the clearance check in @section:evaluation:episodes.

Source events are grouped into a contamination cluster whenever the zero-chlorine trajectory does not clear persistently between them. A reference-event window begins at the first threshold exceedance and ends at persistent clearance, which may precede source shutdown. The interval includes its start, but excludes its endpoint.

Clearance is persistent if the aggregate remains below or at the threshold until the next original source injection or the end of evaluation. Consequently, reference-event windows may differ from the original source-event intervals due to transport delays in the network. An event produces no reference-event window if the transported arsenic aggregate remains at or below the clearance threshold, as can occur when the arsenic event is too weak or when injected arsenic does not flow into adjacent nodes. @fig:reference-window-example illustrates how transported #acr("As(III)") can persist after a single arsenic source event has ended.

During the clearance tail defined in @section:evaluation:episodes, only samples within these reference-event windows contribute to the physical control metrics. The subsequent reference-normal part of the tail contributes only to the evaluation return defined in @section:evaluation:learning-curves.

#figure(
  include "reference-window-example.typ",
  placement: none,
  caption: [Zero-chlorine reference trajectory for one control-validation event in `scenario-85` with arsenic schedule `arsenic-2`. The plot starts four hours before injection and ends four hours after persistent clearance. Gray and orange lines show the sums of local source and transported #acr("As(III)") concentrations across monitored nodes, respectively, in #mg-l(). The gray and orange spans above the curves mark the #num[39.5]-hour source interval and the #num[8.5]-hour interval from source shutdown to persistent clearance. Orange shading marks the reference-event window, which closes at persistent clearance, with $A_(Sigma) (t)$ remaining at or below $epsilon$ = #mg-l[0.01] until the next source injection or episode end. Values are reported every #minute[30].],
) <fig:reference-window-example>


=== Common Notation

We use $"N"$ and $"E"$ for the reference-normal and reference-event regimes defined in @section:evaluation:reference-windows. They are represented by

$
  I^"E" (t) & = cases(
                1 quad & "if" t "lies in a reference-event window",
                0 & "otherwise,",
              ) \
  I^"N" (t) & = 1-I^"E" (t).
$

Let $N$ be the number of monitored nodes, indexed by $v = 1, dots, N$. The chlorine concentration at node $v$ after control step $t$ is $c_(v,t)$. The chlorine injection rate is denoted by $u_t$. The common reporting and control interval is $Delta t$ = #minute[30] (#hour[0.5]). It is expressed in hours for concentration-time metrics and clearance differences, and in minutes when multiplied by the injection rate in #mg-min().

The chlorine limits follow the operational bounds established in @section:methodology:operational-objectives. The lower and upper chlorine bounds for normal operation are $L$ = #mg-l[0.2] and $U$ = #mg-l[0.4], respectively. The absolute chlorine ceiling is $C_(max)$ = #mg-l[5.0], chosen based on the WHO's health-based guideline value for free chlorine in drinking water @who2026guidelines. Values on a bound are compliant, so only strict shortfalls or exceedances count as violations.

Using the evaluation-window timestep sets $cal(T)_w$ introduced in @section:evaluation:episodes, the number of eligible node--time observations for window $w$ and regime $r in {"N","E"}$ is

$ N_w^r = N sum_(t in cal(T)_w) I^r (t). $

Additive metrics are defined as zero when no timesteps are eligible for scoring. A fraction with a positive denominator is zero when its numerator is zero. If no eligible observations are present and its denominator is therefore zero, the fraction is recorded as not applicable.

@tab:evaluation-notation summarizes the recurring symbols and units used in the control metrics.

#{
  show: continued-tables(3)
  set par(justify: false)

  academic-table(
    [Notation used in the control-performance evaluation. Counts, indicators, and fractions are dimensionless.],
    header: ([*Symbol*], [*Meaning*], [*Unit*]),
    (
      [$v$],
      [Monitored-node index],
      [--],

      [$w$],
      [Evaluation-window index, $"M"$ or $"T"$],
      [--],

      [$r$],
      [Reference-regime index, $"N"$ or $"E"$],
      [--],

      [$cal(T)_w$],
      [Timesteps in evaluation window $w$],
      [--],

      [$N$],
      [Number of monitored nodes],
      [--],

      [$c_(v,t)$],
      [Nodal chlorine concentration],
      [#mg-l()],

      [$a_(v,t)$],
      [Nodal transported #acr("As(III)") concentration],
      [#mg-l()],

      [$A_(Sigma) (t)$],
      [Sum of transported #acr("As(III)") concentrations],
      [#mg-l()],

      [$u_t$],
      [Chlorine injection rate],
      [#mg-min()],

      [$L$, $U$],
      [Lower and upper normal chlorine bounds],
      [#mg-l()],

      [$C_(max)$],
      [Absolute chlorine ceiling],
      [#mg-l()],

      [$epsilon$],
      [Clearance threshold for $A_(Sigma) (t)$],
      [#mg-l()],

      [$Delta t$],
      [Common reporting and control interval],
      [#minute() or #hour()],

      [$I^"N" (t)$, $I^"E" (t)$],
      [Reference-normal and reference-event indicators],
      [--],

      [$N_w^r$],
      [Eligible node--time count in window $w$ and regime $r$],
      [--],

      [$f_("low")^"N"$, $f_("high")^"N"$],
      [Normal-operation shortfall and excess fractions],
      [--],

      [$B^"E"$],
      [Integrated transported arsenic burden],
      [#mg-h-l()],

      [$Delta_("clear")$],
      [Mean arsenic clearance lead relative to the zero-chlorine reference],
      [#hour()],

      [$f_("cap")^"E"$],
      [Event-time ceiling-violation fraction],
      [--],

      [$U_("eval")$],
      [Evaluated chlorine use],
      [#milligram()],
    ),
    format: "booktabs",
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    label: <tab:evaluation-notation>,
  )
}


=== Chlorine Shortfalls and Excesses <section:evaluation:chlorine-shortfalls-and-excesses>

The two compliance metrics are measured only in the reference-normal regime of the main window. The fractions of eligible observations below the lower chlorine bound and above the normal upper bound are defined as

$
   f_("low")^"N" & = frac(
                     sum_(t in cal(T)_"M") sum_(v=1)^N I^"N" (t) thin bold(1)[c_(v,t)<L],
                     N_"M"^"N",
                   ), \
  f_("high")^"N" & = frac(
                     sum_(t in cal(T)_"M") sum_(v=1)^N I^"N" (t) thin bold(1)[c_(v,t)>U],
                     N_"M"^"N",
                   ).
$ <eq:evaluation-normal-chlorine-shortfall>

Lower values indicate better compliance. Keeping the two fractions separate distinguishes insufficient residual chlorine from excessive concentrations during normal operation.

An exploratory analysis was added after the primary evaluation to examine when reference-normal chlorine excess occurs. For each reference-event window defined in @section:evaluation:reference-windows, we consider the first #hour[12] after reference clearance. This period is shortened if a new source injection begins or the main window ends. The same periods apply to both policies.

We separate the exceedances contributing to $f_("high")^"N"$ into two groups: those recorded during these post-clearance periods and those recorded at other reference-normal times in the main window. For each rollout, each group's exceedance count is divided by the total number of reference-normal node--time observations, $N_"M"^"N"$. These contributions are then averaged using the weighting in @section:evaluation:statistical-analysis. Their sum equals mean $f_("high")^"N"$. Dividing each mean contribution by this total gives the proportion of chlorine excess occurring in that group.


=== Arsenic Burden

Arsenic burden combines the magnitude and duration of transported arsenic concentrations in the reference-event regime. For each window $w in {"M","T"}$, the component is

$
  B_w^"E" = Delta t sum_(t in cal(T)_w) I^"E" (t) thin A_(Sigma) (t),
$

where $A_(Sigma) (t)$ is the sum of transported #acr("As(III)") concentrations across all monitored nodes at timestep $t$, as defined in @eq:evaluation:transported-arsenic.

The primary metric accumulates both components,

$ B^"E" = B_"M"^"E" + B_"T"^"E". $ <eq:evaluation-arsenic-burden-total>

The tail component therefore contains only the part of a reference event whose clearance extends beyond the main window. Lower values indicate less monitored arsenic burden. The metric is a sum of concentration--time burdens across monitored nodes and is measured in #mg-h-l(). It does not measure physical arsenic mass or population exposure.

Arsenic burden is also evaluated for the zero-chlorine injection profile. This provides an upper bound on untreated transported arsenic burden for each scenario and arsenic schedule combination, allowing us to compare policy performance with the absence of chlorine injection.


=== Mean Arsenic Clearance Lead <section:evaluation:arsenic-clearance-lead>

Mean arsenic clearance lead complements integrated arsenic burden by measuring how much earlier a policy reaches persistent clearance than the zero-chlorine reference.

Within a rollout of the policy being evaluated, let $cal(W)$ be the set of reference-event windows from @section:evaluation:reference-windows in which $A_(Sigma) (t)$ exceeds $epsilon$ at least once. Each $W in cal(W)$ contains the step indices of one such window.

For each $W$, the clearance step $t_("clear") (W)$ is the first reported timestep after the final sample with $A_(Sigma) (t) > epsilon$,

$
  t_("clear") (W) = max {t in W : A_(Sigma) (t) > epsilon} + 1.
$

For the zero-chlorine reference policy $pi_"ref"$, $t_("clear")^(pi_"ref") (W)$ is the endpoint of the same reference-event window.

The primary metric for $|cal(W)| > 0$ is the arithmetic mean of the clearance-step differences, multiplied by $Delta t$ expressed in hours,

$
  Delta_("clear") = frac(Delta t, |cal(W)|) sum_(W in cal(W)) lr((t_("clear")^(pi_"ref") (W) - t_("clear") (W)), size: #125%).
$ <eq:evaluation-arsenic-clearance-lead>

Positive values indicate earlier clearance on average than the reference, and zero indicates no average timing difference.


=== Chlorine Ceiling Violation

Violations of the absolute chlorine ceiling are evaluated in the reference-event regime across both windows. For each window, the number of violations is

$ K_("cap",w)^"E" = sum_(t in cal(T)_w) sum_(v=1)^N I^"E" (t) thin bold(1)[c_(v,t)>C_(max)]. $

The primary violation fraction pools the eligible node--time observations from the main and tail windows,

$
  f_("cap")^"E" = frac(
    K_("cap","M")^"E" + K_("cap","T")^"E",
    N_"M"^"E" + N_"T"^"E",
  ).
$

The two window fractions are not averaged. Pooling their numerators and denominators gives every eligible node--time observation the same weight. Lower values indicate fewer ceiling violations. Concentrations between the normal upper bound $U$ and $C_(max)$ do not count as violations of this metric.


=== Chlorine Use

Chlorine use is evaluated throughout the main window, while in the tail, it is only measured during a reference event. The components for the eligible window--regime pairs $(w,r) in {("M","N"), ("M","E"), ("T","E")}$ are defined by

$
  U_w^r = Delta t sum_(t in cal(T)_w) I^r (t) thin u_t.
$

The primary result is their sum,

$ U_("eval") = U_"M"^"N" + U_"M"^"E" + U_"T"^"E". $

The rate uses #mg-min() as established in @section:methodology:control-action-and-actuator-constraints. Chlorine use is calculated in #milligram() and reported in tonnes (#tonne()). Injection in the tail after the final reference event has ended does not contribute to the metric. Chlorine use is interpreted alongside the concentration metrics to determine whether lower chemical use preserves control quality.


=== Cross-Window Aggregation

Arsenic burden and chlorine ceiling violations are first calculated separately for the main and tail windows and then combined into one primary result. Arsenic burden accumulates its two window components. The ceiling-violation fraction pools the corresponding numerators and denominators rather than averaging window-specific fractions. Chlorine use adds its main-normal, main-event, and tail-event components. These individual window components remain available as a transparent supplementary breakdown.


== Statistical Analysis <section:evaluation:statistical-analysis>

Each primary metric is calculated once per rollout. For fraction-based metrics, a rollout with no eligible observations is retained as not applicable and does not enter the corresponding numerical average. Within each training run, rollout values are averaged across schedules within each scenario, excluding non-applicable values. The available scenario means are then averaged with equal weight. This produces one summary per metric and training seed. The numbers of applicable rollouts and the paired exclusions are reported alongside the results.

The five training seeds provide the replicates for estimating variability. Evaluation cases remain repeated measurements of their respective trained policy. For each metric, reporting includes the mean and sample standard deviation across seeds, together with the mean paired difference between the pretrained and baseline approaches and its #pct[95] Student-$t$ confidence interval with four degrees of freedom. The difference is calculated as pretrained minus baseline. Positive differences favor pretraining for arsenic clearance lead. Negative differences favor pretraining for the other five primary metrics.

Individual paired differences are reported alongside the aggregate estimates to show the underlying variation. The confidence intervals quantify uncertainty in the estimated mean paired differences for the fixed Hanoi scenarios, evaluation catalogs, and training protocol. The learning curves in @section:evaluation:learning-curves and supplementary numerical summaries use the same scenario weighting. Return standard deviations are rounded to two significant figures and means to the corresponding precision. Other return values use up to three significant figures. All calculations use unrounded values.

Arsenic burden is also compared for each test case after averaging across the five training seeds. Each policy's mean arsenic burden is expressed as a percentage reduction relative to the mean zero-chlorine reference burden over the same test cases.

Paired comparisons of mean arsenic clearance lead use the intersection of the two policy-specific eligible-window sets $cal(W)$ defined in @section:evaluation:arsenic-clearance-lead. Both policies must therefore exceed the threshold within each included reference-event window. Therefore, a rollout pair without any common eligible windows is not applicable. Policy-specific no-exceedance counts and paired exclusions are reported alongside the results.

For a selected test case, nodal chlorine concentrations are summarized at each timestep by their mean and population standard deviation. Transported arsenic peaks are determined within individual reference-event windows, and case-level arsenic burdens are compared across the five training seeds and against the zero-chlorine reference.


== Training Time and Resource Requirements

Time and resource measurements account for both online training and the offline preparation required by pretraining. Computational cost complements the interaction-based comparison in @section:evaluation:learning-curves. Wall-clock times are measured separately for simulation-data generation, forecasting-model training, #acr("MPC")-teacher demonstration generation, behavior cloning, #acr("PPO") training, and control evaluation. Hardware, worker and device counts, and parallelism are recorded to put these metrics into perspective.

Runtimes repeated for each training seed are summarized by their mean and sample standard deviation across the five seeds. Forecasting-model training times are first summed across the five folds within each seed. Shared data-generation costs are counted once and excluded from per-seed totals. Overall costs sum the stage runtimes across all seeds and add shared data generation once. Online interaction counts and training runtimes cover the complete training budget, including any training after the selected checkpoint.


== Additional Variant Comparisons <section:evaluation:additional-variants>

The variants defined in @section:methodology:additional-variants are evaluated through validation return using the same five training seeds, 50 fixed validation cases, and evaluation procedure as the principal comparison. Checkpoints follow the same nominal #num[50000]-interaction spacing. The extended-budget baseline has 81 checkpoints from zero to #num[4000000] nominal interactions and completes #num[4005888] actual interactions. The variant without actor weights uses the 21 checkpoints through one million nominal interactions.

Endpoint comparisons use each variant's return at its full online budget and the pretrained approach's return at the one-million-interaction endpoint. Differences are calculated as variant minus pretrained within matched seed pairs and summarized using the statistical procedure defined above. To compare learning progress, we retrospectively identify the first observed checkpoint at which a variant's mean validation return reaches or exceeds the reference approach's mean return at one million interactions. The reference is the pretrained approach for the extended-budget baseline and the ordinary baseline for the variant without actor weights.

The supplementary runtime comparison reports #acr("PPO") training time for both variants and checkpoint-validation time separately for the extended-budget baseline.


== Supporting Evaluation of the Forecasting Model

The chlorine forecaster is evaluated separately on unseen hydraulic scenarios. For each of the five training seeds, each of the five fold-specific models used to generate the #acr("MPC") teacher labels is applied to all valid forecast origins in the 250 offline trajectories of the ten forecasting test scenarios defined in @section:dataset:splits. Every example uses the 48-step history, planned 16-step injection sequence, and calendar features specified in @section:methodology:forecasting-data.

Predicted and simulated chlorine concentrations are compared after reversing the target normalization. Accuracy is measured using #acr("MAE") and #acr("RMSE"), both expressed in #mg-l(). Reporting both metrics in physical concentration units allows their magnitudes to be interpreted directly.

For each of the 16 forecast steps, both errors are aggregated across all valid forecast origins, trajectories, and monitored nodes within each test scenario. Applying the same aggregation across all forecast steps provides the #acr("MAE") and #acr("RMSE") over the complete eight-hour horizon. For both the step-wise and full-horizon metrics, the ten scenario values are averaged with equal weight for each model. Results are calculated separately for all 25 models. For each training seed, the five model errors are averaged. The overall summary then reports the mean and sample standard deviation across the five seed means.

Together, the step-wise and full-horizon errors answer the question of how accurately the fold-specific models predict chlorine concentrations over the next eight hours on unseen scenarios under prescribed injection profiles. However, a low prediction error alone does not establish the control quality of the demonstrations generated by the #acr("MPC") teacher.
