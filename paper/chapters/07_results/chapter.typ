#import "../../numbers.typ": *
#import "../../utils.typ": continued-tables
#import "@preview/acrostiche:0.7.0": acr, display-short
#import "@preview/akatable:0.1.0": academic-table
#import "@preview/zero:0.7.0": format-table
#import "data.typ": *
#import "variant-validation.typ": (
  extended-baseline-first-crossing, extended-baseline-return, extended-baseline-selected-return,
  extended-baseline-training-cost, pretrained-selected-return-mean, pretrained-selected-return-sd,
  without-actor-first-crossing, without-actor-return, without-actor-selected-return, without-actor-training-cost,
)
#import "case-trajectories.typ": case-trajectory-figure
#import "evaluation-plots.typ": case-burden-figure, control-comparison-figure, forecast-horizon-figure, learning-figure


= Results <chapter:results>

This chapter reports the results obtained with the evaluation protocol. The principal comparison is between the baseline approach and pretrained approach under the same online interaction budget. Validation returns describe both policies before and during #acr("PPO") training, while six physical metrics summarize the test performance of the selected policies. Computational costs, additional variant comparisons, one matched test case, and forecasting errors complement this comparison. The chapter describes the observed values and their variation across training seeds. Their interpretation follows in @chapter:discussion.

In the final test evaluations, we refer to the checkpoints with the highest mean validation return for each approach and training seed as the selected baseline policy and selected pretrained policy. Selection considers the checkpoints evaluated within the nominal one-million-interaction budget and follows @section:evaluation:policy-selection.


== Online Sample Efficiency

Validation returns were evaluated before the first #acr("PPO") update and at every nominal #num[50000]-interaction threshold up to one million, following @section:evaluation:learning-curves. Both approaches reached #num(actual-interactions(1000000)) actual training interactions at the final checkpoint. @fig:validation-return shows two panels with the complete training range and the interval beginning at the nominal #num[500000]-interaction checkpoint. As defined in @eq:evaluation-return, the validation return includes the main window and complete clearance tail and excludes the warm-up.

#let initial-return = learning-summaries.first()
#let early-return = learning-summaries.find(summary => summary.nominal == 100000)
#let midpoint-return = learning-summaries.find(summary => summary.nominal == 500000)
#let late-return = learning-summaries.find(summary => summary.nominal == 900000)
#let final-return = learning-summaries.last()

At zero online interactions, the baseline policy had its ordinary initialization and the pretrained policy had the initialization obtained through behavior cloning. The baseline mean validation return and sample standard deviation were #return-pm(initial-return.at("baseline-mean"), initial-return.at("baseline-sd")). The corresponding pretrained values were #return-pm(initial-return.at("pretrained-mean"), initial-return.at("pretrained-sd")). The mean pretrained-minus-baseline difference was approximately #return-num(initial-return.at("difference-mean")). With random untrained initialization, baseline seed #identifier[71026] started near #fixed(learning-data.rows.find(row => row.approach == "baseline" and row.training_seed == 71026 and row.nominal_interactions == 0).evaluation_return / 1000000, digits: 1) million, below the lower bound shown in panel (a).


The pretrained policy had a higher mean return at every checkpoint, while the mean gap narrowed overall despite fluctuations. The mean paired difference decreased from #return-num(early-return.at("difference-mean")) at #num[100000] nominal interactions to #return-num(midpoint-return.at("difference-mean")) at #num[500000]. The large negative baseline returns at the start dominate the vertical scale of panel (a), whereas panel (b) shows the later reduction in the gap on a separate scale.

By #num[900000] nominal interactions, the difference between the mean returns had decreased to #return-num(late-return.at("difference-mean")). After that, the pretrained mean stayed near #return-mean(late-return.at("pretrained-mean"), late-return.at("pretrained-sd")), while the baseline mean rose from #return-mean(late-return.at("baseline-mean"), late-return.at("baseline-sd")) to #return-mean(final-return.at("baseline-mean"), final-return.at("baseline-sd")). At the nominal one-million-interaction checkpoint, the baseline mean return and sample standard deviation were #return-pm(final-return.at("baseline-mean"), final-return.at("baseline-sd")), compared with #return-pm(final-return.at("pretrained-mean"), final-return.at("pretrained-sd")) for the pretrained policy. The mean paired difference was #return-num(final-return.at("difference-mean")).

The return difference at this budget endpoint was positive in #num(final-return.differences.filter(value => value != none and value > 0).len()) matched seed pairs, with individual differences ranging from #return-num(calc.min(..final-return.differences)) to #return-num(calc.max(..final-return.differences)). @tab:appendix-validation-return in @section:appendix:numerical-results lists the mean returns, sample standard deviations, and paired differences at all 21 checkpoints, together with the nominal and actual interaction counts.

At the nominal #num[300000]-interaction checkpoint, the pretrained mean validation return first reached or exceeded the baseline mean at the one-million-interaction endpoint. It remained above this level at all subsequent checkpoints. Across paired seeds, the first pretrained checkpoint reaching or exceeding the respective baseline endpoint return occurred between #num[100000] and #num[350000] nominal interactions.

For the baseline approach, one checkpoint selected by the highest mean validation return occurred at #num[950000] and four at #num[1000000] nominal online interactions. For the pretrained approach, the checkpoints were selected at #num[500000], #num[550000], #num[650000], #num[800000], and #num[950000]. @tab:validation-selected-checkpoints in @section:appendix:numerical-results lists the individual selection counts and corresponding mean validation returns.

#figure(
  learning-figure(),
  placement: none,
  caption: [Validation returns over the full training range (a) and from #num[500000] to one million nominal online interactions (b), shown against nominal interaction thresholds. Each checkpoint is evaluated on the same #num[50] validation cases. The panels use separate vertical scales. Thick lines show mean validation returns across five training seeds, shaded bands show one sample standard deviation, and thin lines show individual seeds. Panel (a) uses a vertical range based on the mean validation returns, with seed trajectories and standard deviation bands clipped at its boundaries.],
) <fig:validation-return>


== Final Physical Control Performance

This section compares the selected baseline policy and selected pretrained policy regarding normal operation, contamination response and clearance, and chlorine use. Overall, the selected pretrained policy had lower arsenic burden and greater clearance lead, alongside higher normal-operation chlorine excess and chlorine use. @tab:final-control-metrics summarizes the six physical metrics across five training seeds, with paired differences and #pct[95] confidence intervals, following @section:evaluation:statistical-analysis. @tab:seed-differences in @section:appendix:numerical-results lists the paired differences for the individual training seeds.

#let shortfall = metric-summary("chlorine_shortfall_fraction")
#let normal-excess = metric-summary("normal_chlorine_excess_fraction")
#let arsenic-burden = metric-summary("arsenic_burden")
#let clearance-lead = metric-summary("arsenic_clearance_lead")
#let ceiling-violation = metric-summary("chlorine_ceiling_violation_fraction")
#let chlorine-use = metric-summary("chlorine_use")

#{
  set par(justify: false)

  academic-table(
    [Test control metrics for the selected baseline and selected pretrained policies. The normal-operation chlorine shortfall and excess fractions are $f_("low")^"N"$ and $f_("high")^"N"$, respectively. The remaining metrics are transported arsenic burden $B^"E"$, mean arsenic clearance lead $Delta_("clear")$ relative to the zero-chlorine reference, the event-time ceiling-violation fraction $f_("cap")^"E"$, and chlorine use $U_("eval")$. Policy values are means with sample standard deviations across five training seeds. Differences are selected pretrained policy minus selected baseline policy, with fraction differences and their confidence intervals expressed in percentage points. The last column counts applicable evaluations across the five seeds per approach.],
    header: (
      [*Metric*],
      [*Baseline*],
      [*Pretrained*],
      [*Difference*],
      [*#pct[95] #display-short("CI")*],
      [*Rollouts*],
    ),
    (
      (shortfall, [$f_("low")^"N"$ (#pct())], 4),
      (normal-excess, [$f_("high")^"N"$ (#pct())], 3),
      (arsenic-burden, [$B^"E"$ (#mg-h-l())], 1),
      (clearance-lead, [$Delta_("clear")$ (#hour())], 2),
      (ceiling-violation, [$f_("cap")^"E"$ (#pct())], 3),
      (chlorine-use, [$U_("eval")$ (#tonne())], 3),
    )
      .map(((summary, metric, places)) => (
        metric,
        [#num(table-value(
          summary.at("baseline-mean") * summary.definition.scale,
          sd: summary.at("baseline-sd") * summary.definition.scale,
          digits: places,
        ))],
        [#num(table-value(
          summary.at("pretrained-mean") * summary.definition.scale,
          sd: summary.at("pretrained-sd") * summary.definition.scale,
          digits: places,
        ))],
        [#table-value(summary.at("difference-mean") * summary.definition.scale, digits: summary.definition.digits)],
        [#table-interval(summary)],
        [#table-value(summary.applicable, digits: 0)],
      ))
      .flatten(),
    format: "booktabs",
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, center, right),
    label: <tab:final-control-metrics>,
  )
}


=== Normal Operation

Both normal-operation fractions were applicable to all #num(shortfall.applicable) rollouts per approach. Mean reference-normal chlorine shortfall was small for both policies, at #format-metric-pct-like-pm(shortfall.definition, shortfall.at("baseline-mean"), shortfall.at("baseline-sd")) for the selected baseline policy and #format-metric-pct-like-pm(shortfall.definition, shortfall.at("pretrained-mean"), shortfall.at("pretrained-sd")) for the selected pretrained policy. The paired #pct[95] confidence interval spanned zero, leaving the direction of the mean difference uncertain.

Reference-normal chlorine excess increased by #format-metric(normal-excess.definition, normal-excess.at("difference-mean")) percentage points under the selected pretrained policy. Although small in absolute terms, this increase occurred in all five matched seed pairs, and its #pct[95] confidence interval excluded zero.

#let baseline-post-cluster-excess = post-cluster-excess("baseline")
#let pretrained-post-cluster-excess = post-cluster-excess("pretrained")
#assert(calc.abs(baseline-post-cluster-excess.total_metric_fraction - normal-excess.at("baseline-mean")) < 1e-12)
#assert(calc.abs(pretrained-post-cluster-excess.total_metric_fraction - normal-excess.at("pretrained-mean")) < 1e-12)

In the exploratory decomposition defined in @section:evaluation:chlorine-shortfalls-and-excesses, the first #hour[12] after reference clearance accounted for #fixed-pct(100 * baseline-post-cluster-excess.inside_fraction, digits: 2) of the selected baseline policy's mean chlorine excess and #fixed-pct(100 * pretrained-post-cluster-excess.inside_fraction, digits: 2) of the selected pretrained policy's mean.

The difference in total excess was almost entirely accounted for by the post-clearance intervals, whose contribution to mean $f_("high")^"N"$ was #fixed(100 * (pretrained-post-cluster-excess.inside_metric_fraction - baseline-post-cluster-excess.inside_metric_fraction), digits: 3) percentage points higher under the selected pretrained policy. In the remaining reference-normal periods, the contributions were #fixed-pct(100 * baseline-post-cluster-excess.outside_metric_fraction, digits: 5) for the selected baseline policy and #fixed-pct(100 * pretrained-post-cluster-excess.outside_metric_fraction, digits: 5) for the selected pretrained policy, a difference of #fixed(100 * (pretrained-post-cluster-excess.outside_metric_fraction - baseline-post-cluster-excess.outside_metric_fraction), digits: 5) percentage points.


=== Contamination Response and Clearance

Mean transported arsenic burden was #fixed-pct(100 * (1 - arsenic-burden.at("pretrained-mean") / arsenic-burden.at("baseline-mean")), digits: 1) lower under the selected pretrained policy, with lower burdens in all five matched seed pairs. The paired difference was #format-metric(arsenic-burden.definition, arsenic-burden.at("difference-mean")) #mg-h-l(), with a #pct[95] confidence interval from #format-metric(arsenic-burden.definition, arsenic-burden.at("ci-low")) to #format-metric(arsenic-burden.definition, arsenic-burden.at("ci-high")) #mg-h-l().

Across the test cases, the policy-independent zero-chlorine reference had a mean case-level burden of #fixed(zero-reference-burden, digits: 1) #mg-h-l(). Relative to this reference mean, transported arsenic burden was reduced by #fixed-pct(100 * (1 - arsenic-burden.at("baseline-mean") / zero-reference-burden), digits: 1) under the selected baseline policy and #fixed-pct(100 * (1 - arsenic-burden.at("pretrained-mean") / zero-reference-burden), digits: 1) under the selected pretrained policy. Nearly all of this burden occurred in the main window.

Mean arsenic clearance lead was #duration-text(clearance-lead.at("baseline-mean") * 3600) for the selected baseline policy and #duration-text(clearance-lead.at("pretrained-mean") * 3600) for the selected pretrained policy. The paired mean difference corresponded to persistent clearance #duration-text(clearance-lead.at("difference-mean") * 3600) earlier under the selected pretrained policy. These time values are rounded to the nearest minute. Clearance lead was higher in all five matched seed pairs, but the #pct[95] confidence interval for the paired difference ranged from #duration-text(clearance-lead.at("ci-low") * 3600) to #duration-text(clearance-lead.at("ci-high") * 3600), indicating uncertainty in the size of this improvement.

This clearance comparison was conditional on #num(clearance-lead.applicable) eligible paired rollouts per approach. Of the #num[25] excluded rollouts, #num[20] had no reference-event windows because the zero-chlorine reference never exceeded #mg-l[0.01]. The remaining five concerned `scenario-96` with arsenic schedule `arsenic-5`, where at least one policy never exceeded the threshold. Neither policy exceeded it for seeds #identifier[17], #identifier[42], #identifier[123], and #identifier[71026]. For seed #identifier[1337], the selected baseline policy stayed at or below the threshold while the selected pretrained policy exceeded it, leaving no eligible paired clearance-time comparison.

The event-time chlorine ceiling-violation fraction was zero for both policies in all applicable event windows. The ceiling-violation fraction excluded only the #num[20] rollouts without reference-event samples and was therefore applicable to #num(ceiling-violation.applicable) rollouts per approach, following @section:evaluation:reference-windows.

@fig:test-case-burden compares performance across test cases and training seeds. Averaged across the five seeds, the selected pretrained policy's burden was lower in #num(case-burden-points.filter(row => row.pretrained < row.baseline).len()) of the 100 fixed test cases, equal in #num(case-burden-points.filter(row => row.pretrained == row.baseline).len()), and higher in #num(case-burden-points.filter(row => row.pretrained > row.baseline).len()). Across all five matched seed pairs, lower mean burden coincided with earlier persistent clearance on average over the eligible events.

#figure(
  control-comparison-figure(case-burden-points, joint-control-points),
  placement: none,
  caption: [Physical test performance of the selected baseline and selected pretrained policies. In (a), each of the 100 points pairs the selected baseline and selected pretrained arsenic burdens for one test case, averaged across five training seeds. Both axes use the same scale. The dotted line marks equal burdens. Points below it indicate lower burdens for the selected pretrained policy, and points above it indicate higher burdens. In (b), each of the ten points summarizes arsenic burden across all 100 test cases and mean arsenic clearance lead relative to the zero-chlorine reference across the applicable cases for one policy and training seed. Colors identify the policies, labels identify the seeds, and lines connect matched seed pairs.],
) <fig:test-case-burden>


=== Chlorine Use

The selected pretrained policy used #fixed-pct(100 * (chlorine-use.at("pretrained-mean") / chlorine-use.at("baseline-mean") - 1), digits: 1) more chlorine on average, alongside the #fixed-pct(100 * (1 - arsenic-burden.at("pretrained-mean") / arsenic-burden.at("baseline-mean")), digits: 1) reduction in arsenic burden. Chlorine use was higher in all five matched seed pairs, and the paired #pct[95] confidence interval excluded zero. This additional chlorine represents greater treatment effort, with about #fixed-pct(100 * (chlorine-use-components.pretrained.at("main-event") - chlorine-use-components.baseline.at("main-event")) / chlorine-use.at("difference-mean"), digits: 0) of the increase occurring during reference-event periods in the main window.

In the main window, reference-normal operation accounted for about #tonne(fixed-string(chlorine-use-components.baseline.at("main-normal") / 1000000000, digits: 2)) for the selected baseline policy and #tonne(fixed-string(chlorine-use-components.pretrained.at("main-normal") / 1000000000, digits: 2)) for the selected pretrained policy. The corresponding reference-event components were about #tonne(fixed-string(chlorine-use-components.baseline.at("main-event") / 1000000000, digits: 2)) and #tonne(fixed-string(chlorine-use-components.pretrained.at("main-event") / 1000000000, digits: 2)).


== Computational Cost

Summing the recorded training-pipeline stage runtimes across the five seeds gives #duration-text(baseline-total-seconds) for the baseline approach and #duration-text(pretrained-total-seconds) for the pretrained approach, including checkpoint validation and test evaluation. The pretrained pipeline took about #fixed(pretrained-total-seconds / baseline-total-seconds, digits: 1) times as long. Its shorter #acr("PPO") phase did not offset the offline preparation cost. @tab:computational-costs summarizes the runtimes of data generation, model training, and evaluation.

#let baseline-eval = duration-text(baseline-control-evaluation-cost.mean)
#let pretrained-eval = duration-text(pretrained-control-evaluation-cost.mean)

For the baseline approach, the mean total runtime per seed was #duration-text(baseline-seed-cost.mean) and #duration-text(pretrained-seed-cost.mean) for the pretrained approach. Across the five training seeds, mean #acr("PPO") training time was #duration-text(baseline-ppo-cost.mean) for the baseline approach and #duration-text(pretrained-ppo-cost.mean) for the pretrained approach. Validation and test evaluation added #if baseline-eval == pretrained-eval [#baseline-eval per seed for both approaches] else [#baseline-eval per baseline seed and #pretrained-eval per pretrained seed].

Simulating the offline trajectories used to train the forecasters and to generate the #acr("MPC") teacher dataset took #duration-text(offline-trajectories-cost.total) with #num(offline-trajectories-cost.workers) CPU workers. The trajectories were generated once and reused across the five pretrained runs. Their generation time is counted once in the global pretrained total and excluded from the per-seed means.

The seed-specific offline preparation of the forecasting models, the #acr("MPC") teacher demonstrations, and behavior cloning required a mean of #duration-text(offline-preparation-per-seed-seconds) per pretrained seed. Training the five fold-specific forecasting models required a mean of #duration-text(forecaster-cost.total / training-seeds.len()) per seed, or #duration-text(forecaster-cost.mean) per fold-specific forecaster. Generating the #acr("MPC") demonstrations required a mean of #duration-text(mpc-cost.mean) per seed. The mean times were #duration-text(mpc-training-cost.mean) for the behavior cloning training dataset and #duration-text(mpc-validation-cost.mean) for the separate validation dataset. Mean generation time per training fold was #duration-text(mpc-fold-cost.mean). Mean behavior cloning time was #duration-text(behavior-cloning-cost.mean) per seed.

The wall-clock times refer to the training computer described in @section:methodology:implementation-and-reproducibility and the stage-specific worker counts in @tab:computational-costs. The learning stages used the GPU through CUDA. Trajectory generation and #acr("MPC") demonstration generation used the CPU.

#{
  show: continued-tables(4)
  set par(justify: false)
  show table: format-table(none, none, auto, auto)

  academic-table(
    [Wall-clock costs per training seed, reported as mean #sym.plus.minus sample standard deviation across the five seeds in hours. Forecaster training includes all five folds per seed. Control evaluation includes checkpoint validation and test evaluation. Totals sum the recorded pipeline stages for each seed before calculating the mean and standard deviation. Shared trajectory generation is shown separately from the per-seed totals.],
    header: ([*Stage*], [*Device / workers*], [*Baseline (#hour())*], [*Pretrained (#hour())*]),
    (
      [Shared offline trajectory generation (once)],
      [#cost-device(offline-trajectories-cost) / #offline-trajectories-cost.workers],
      [--],
      [#table-value(offline-trajectories-cost.total / 3600, digits: 3)],

      [Forecaster training],
      [#cost-device(forecaster-cost) / #forecaster-cost.workers],
      [--],
      [#table-value(forecaster-seed-cost.mean / 3600, sd: forecaster-seed-cost.sd / 3600, digits: 3)],

      [#acr("MPC") demonstrations (training)],
      [CPU / #mpc-training-cost.workers],
      [--],
      [#table-value(mpc-training-cost.mean / 3600, sd: mpc-training-cost.sd / 3600, digits: 3)],

      [#acr("MPC") demonstrations (validation)],
      [CPU / #mpc-validation-cost.workers],
      [--],
      [#table-value(mpc-validation-cost.mean / 3600, sd: mpc-validation-cost.sd / 3600, digits: 3)],

      [Behavior cloning],
      [#cost-device(behavior-cloning-cost) / #behavior-cloning-cost.workers],
      [--],
      [#table-value(behavior-cloning-cost.mean / 3600, sd: behavior-cloning-cost.sd / 3600, digits: 3)],

      [#acr("PPO") training],
      [#cost-device(baseline-ppo-cost) / #baseline-ppo-cost.workers],
      [#table-value(baseline-ppo-cost.mean / 3600, sd: baseline-ppo-cost.sd / 3600, digits: 3)],
      [#table-value(pretrained-ppo-cost.mean / 3600, sd: pretrained-ppo-cost.sd / 3600, digits: 3)],

      [Control evaluation],
      [#cost-device(baseline-control-evaluation-cost) / #baseline-control-evaluation-cost.workers],
      [#table-value(
        baseline-control-evaluation-cost.mean / 3600,
        sd: baseline-control-evaluation-cost.sd / 3600,
        digits: 3,
      )],
      [#table-value(
        pretrained-control-evaluation-cost.mean / 3600,
        sd: pretrained-control-evaluation-cost.sd / 3600,
        digits: 3,
      )],

      table.hline(stroke: 0.75pt),
      [Total per seed],
      [],
      [#table-value(baseline-seed-cost.mean / 3600, sd: baseline-seed-cost.sd / 3600, digits: 3)],
      [#table-value(pretrained-seed-cost.mean / 3600, sd: pretrained-seed-cost.sd / 3600, digits: 3)],
    ),
    format: "booktabs",
    columns: (1fr, auto, auto, auto),
    align: (left, center, right, right),
    label: <tab:computational-costs>,
  )
}


== Additional Variant Comparisons <section:results:additional-variants>

The two variants defined in @section:methodology:additional-variants were compared through validation return following @section:evaluation:additional-variants. @tab:variant-validation-returns summarizes their endpoint comparisons with the pretrained approach. Checkpoint selection follows @section:evaluation:policy-selection.

At #num(extended-baseline-first-crossing) nominal interactions, the extended-budget baseline mean first reached or exceeded the pretrained mean at the one-million-interaction endpoint. It later fell below this reference again before ending slightly above it at #num(extended-baseline-return.nominal) interactions. At this endpoint, the paired difference was positive in #num(extended-baseline-return.differences.filter(value => value > 0).len()) of the five seed pairs, and its #pct[95] confidence interval spanned zero.

The selected policies of the extended-budget baseline had a mean validation return of #return-mean(extended-baseline-selected-return.mean, extended-baseline-selected-return.sd), compared with #return-mean(pretrained-selected-return-mean, pretrained-selected-return-sd) for the selected pretrained policies. The checkpoints were selected between #num(calc.min(..extended-baseline-selected-return.nominal-checkpoints)) and #num(calc.max(..extended-baseline-selected-return.nominal-checkpoints)) nominal interactions for the extended-budget baseline and between #num[500000] and #num[950000] for the pretrained approach. The mean paired difference between the selected policies was #box(return-num(extended-baseline-selected-return.difference-mean)), with a paired #pct[95] confidence interval from #return-num(extended-baseline-selected-return.ci-low) to #return-num(extended-baseline-selected-return.ci-high).

Without actor weights, the mean validation return exceeded the ordinary baseline mean from #num[50000] through #num[950000] nominal interactions, but remained below the pretrained mean at every checkpoint. Its mean first reached or exceeded the ordinary baseline's one-million endpoint return at #num(without-actor-first-crossing) nominal interactions. At the one-million endpoint, its mean was again slightly below the ordinary baseline's. Relative to the pretrained approach, its endpoint return was lower in all five matched seed pairs, with a paired #pct[95] confidence interval below zero.

The selected policies without actor weights likewise had a mean validation return of #return-mean(without-actor-selected-return.mean, without-actor-selected-return.sd), compared with #return-mean(pretrained-selected-return-mean, pretrained-selected-return-sd) for the selected pretrained policies. This selected-policy difference was negative in all five matched seed pairs, with a paired #pct[95] confidence interval below zero.

Mean #acr("PPO") training time per seed was #duration-text(extended-baseline-training-cost.mean) #sym.plus.minus #duration-text(extended-baseline-training-cost.sd, with-about: false) for the extended-budget baseline and #duration-text(without-actor-training-cost.mean) #sym.plus.minus #duration-text(without-actor-training-cost.sd, with-about: false) for the variant without actor weights. Checkpoint validation for the extended-budget baseline took a mean of #duration-text(extended-baseline-training-cost.validation-mean) per seed.

#{
  set figure(placement: none)
  set par(justify: false)

  academic-table(
    [Endpoint validation returns for the additional variants. The 'Interactions' column gives the nominal online training interaction counts. Returns are means with sample standard deviations across five training seeds. Differences are variant minus pretrained at the pretrained approach's one-million-interaction endpoint, with paired #pct[95] confidence intervals.],
    header: (
      [*Variant*],
      [*Interactions*],
      [*Validation return*],
      [*Difference*],
      [*#pct[95] #display-short("CI")*],
    ),
    (
      ([Extended-budget baseline], extended-baseline-return),
      ([Without actor weights], without-actor-return),
    )
      .map(((name, summary)) => (
        [#name],
        [#num(summary.nominal)],
        [#return-pm(summary.mean, summary.sd)],
        [#return-num(summary.difference-mean)],
        [#return-num(summary.ci-low) to #return-num(summary.ci-high)],
      ))
      .flatten(),
    format: "booktabs",
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, center),
    label: <tab:variant-validation-returns>,
  )
}


== Supporting Case-Level Evidence

Test scenario #raw(selected-case-scenario) with arsenic schedule #raw(selected-case-schedule) and training seed #identifier(case-trajectory-data.seed) were selected to illustrate a pronounced reduction in transported arsenic burden under the selected pretrained policy alongside differences in the responses to individual contamination events. The plotted policy rollouts are part of the aggregate test results above. The #num(selected-case-days)-day episode contains four arsenic source-injection events at different nodes. Two of these overlap in time, resulting in three reference-event windows, as defined in @section:evaluation:reference-windows. @fig:matched-case-trajectories shows the complete episode, including chlorine control between the contamination events.

The largest separation occurs in the third reference-event window, where mean nodal chlorine concentrations under the selected pretrained policy rise above the normal upper bound, while those under the selected baseline policy fall towards the normal lower bound. The transported #acr("As(III)") concentration sum peaks at #mg-l(fixed-string(selected-case-window-peak(case-trajectory-data.reference_transported_arsenic_sum_mg_l, 2), digits: 2)) in the zero-chlorine reference, compared with #mg-l(fixed-string(selected-case-window-peak(case-trajectory-data.controllers.ppo_best.transported_arsenic_sum_mg_l, 2), digits: 2)) for the selected baseline policy and #mg-l(fixed-string(selected-case-window-peak(case-trajectory-data.controllers.finetuned_best.transported_arsenic_sum_mg_l, 2), digits: 2)) for the selected pretrained policy. #{
  let event = selected-case-clearance-events.at(2)
  let reference-time = event.reference_clearance_h * 3600
  let baseline-time = event.baseline_clearance_h * 3600
  let pretrained-time = event.pretrained_clearance_h * 3600
  let lead = baseline-time - pretrained-time

  let comparison = if lead > 0 { [#duration-text(lead) earlier] } else if lead < 0 {
    [#duration-text(-lead) later]
  } else { [at the same time] }

  if reference-time == baseline-time {
    [In this window, both the zero-chlorine reference and the selected baseline policy reached persistent clearance at #duration-text(reference-time) of simulation time, while the selected pretrained policy reached it #comparison.]
  } else {
    [In this window, the zero-chlorine reference reached persistent clearance at #duration-text(reference-time) of simulation time. The selected baseline policy reached clearance at #duration-text(baseline-time), while the selected pretrained policy reached it #comparison.]
  }
}

For the complete case, transported arsenic burden was #format-metric(selected-case-values.at(2).definition, selected-case-seed-burden.reference) #mg-h-l() in the zero-chlorine reference, while it was #format-metric(selected-case-values.at(2).definition, selected-case-seed-burden.baseline) #mg-h-l() for the selected baseline policy and #format-metric(selected-case-values.at(2).definition, selected-case-seed-burden.pretrained) #mg-h-l() for the selected pretrained policy. @fig:selected-case-burden places this comparison in the context of all #num(selected-case-burdens.len()) training seeds for the same test case. #if selected-case-lower-seeds.len() > 0 [Arsenic burden was lower for the selected pretrained policy than for the selected baseline policy for seeds #selected-case-lower-seeds.map(seed => identifier(seed)).join([, ], last: if selected-case-lower-seeds.len() > 2 [, and ] else [ and ]).] #if selected-case-higher-seeds.len() > 0 [It was higher for seeds #selected-case-higher-seeds.map(seed => identifier(seed)).join([, ], last: if selected-case-higher-seeds.len() > 2 [, and ] else [ and ]).] Each group compares the two policy burdens with the same zero-chlorine reference.

#figure(
  case-trajectory-figure(),
  placement: none,
  caption: [Complete trajectories for the selected illustrative case, #raw(selected-case-scenario) with arsenic schedule #raw(selected-case-schedule), and training seed #identifier(case-trajectory-data.seed). Panels show (a) chlorine injection, (b) mean nodal chlorine concentration with one population standard deviation across the #num(case-trajectory-data.node_count) nodes, and (c) the transported #acr("As(III)") concentration sum (left axis) with total arsenic source injection as dashed steps (right axis). Values are reported every #num(selected-case-interval-minutes) minutes over #num(selected-case-days) days. Lower band edges in (b) are clipped at zero. Horizontal dotted lines mark the normal chlorine bounds of #mg-l[0.2] and #mg-l[0.4]. Vertical dashed lines mark the end of the warm-up at simulation day #num[2] and the start of the clearance tail at simulation day #num[23]. Shaded intervals mark the reference-event windows.],
) <fig:matched-case-trajectories>

#figure(
  case-burden-figure(selected-case-burdens),
  placement: none,
  caption: [Transported arsenic burden for the selected illustrative case, test scenario #raw(selected-case-scenario) with arsenic schedule #raw(selected-case-schedule), for each training seed. Separate bars show the actual burdens under the selected baseline policy, selected pretrained policy, and zero-chlorine reference.],
) <fig:selected-case-burden>


== Supporting Forecasting Results

Forecasting accuracy was evaluated on the ten forecasting test scenarios, with equal weight assigned to each scenario. For each training seed, #acr("MAE") and #acr("RMSE") were averaged across the five fold-specific models. The reported values summarize these five seed means through their mean and sample standard deviation.

Over the full forecasting horizon of eight hours, #acr("MAE") was #fixed-pm(forecast-mae-mean, forecast-mae-sd, digits: 4, places: 5) #mg-l() and #acr("RMSE") was #fixed-pm(forecast-rmse-mean, forecast-rmse-sd, digits: 4, places: 5) #mg-l(). The full-horizon errors for all 25 models are shown by training seed in @fig:forecast-full-horizon and listed by seed and fold in @tab:appendix-forecast-model-errors, both in @section:appendix:numerical-results.

Both mean errors increased at every step of the forecast horizon, as shown in @fig:forecast-horizon-errors. At the first step, #num[30] minutes ahead, mean #acr("MAE") was #fixed-like-pm(forecast-step-summaries.first().at("mae-mean"), forecast-step-summaries.first().at("mae-sd"), digits: 4, places: 5) #mg-l() and mean #acr("RMSE") was #fixed-like-pm(forecast-step-summaries.first().at("rmse-mean"), forecast-step-summaries.first().at("rmse-sd"), digits: 4, places: 5) #mg-l(). At the final step, #num[8] hours ahead, the corresponding errors were #fixed-like-pm(forecast-step-summaries.last().at("mae-mean"), forecast-step-summaries.last().at("mae-sd"), digits: 4, places: 5) #mg-l() and #fixed-like-pm(forecast-step-summaries.last().at("rmse-mean"), forecast-step-summaries.last().at("rmse-sd"), digits: 4, places: 5) #mg-l(). Values for each forecast step are listed in @tab:appendix-forecast-horizon-errors in @section:appendix:numerical-results.

#figure(
  forecast-horizon-figure(forecast-step-summaries),
  caption: [Chlorine-forecast errors over the eight-hour horizon, with (a) #acr("MAE") and (b) #acr("RMSE"). Errors are evaluated on the ten forecasting test scenarios under prescribed injection profiles. Lines show the mean and shaded bands show one sample standard deviation across the five seed means.],
) <fig:forecast-horizon-errors>
