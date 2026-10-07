#import "../../numbers.typ": *
#import "@preview/acrostiche:0.7.0": acr
#import "@preview/akatable:0.1.0": academic-table
#import "../07_results/data.typ": fixed-string

#let delay-data = json("../../results/network-delay-summary.json")
#let delay-counts = delay-data.counts
#let rounded(value) = num(if calc.round(value, digits: 2) == 100 { "100" } else { fixed-string(value, digits: 2) })
#let percent(count, total) = rounded(100 * count / total)
#assert(delay-counts.trials == 1680 and delay-counts.responsive == 51879)
#assert(delay-data.history.len() == 6 and delay-data.horizons.len() == 7)

= Network-Delay Diagnostic for Temporal-Window Selection <section:appendix:network-delay>

The diagnostic examines how long the effect of a single chlorine action remains visible. It compares two simulations of the same hydraulic scenario. Both use a constant injection rate of #mg-min[100000], except that one raises the rate to #mg-min[125000] for a single 30-minute step. The resulting difference in chlorine concentration is recorded for #hour[48] at each of the 31 monitored junctions. No arsenic is injected.

The start times cover every hour of the day and every phase of the weekly demand cycle. For each combination of weekday and hour, the diagnostic considers three occurrences during days 2 to 22 of each of the 80 forecasting-training scenarios. Within each scenario, it selects the occurrence with the lowest source outflow over the following #hour[24]. It then selects the ten scenarios with the lowest outflow. This procedure yields 1680 pulse trials across 168 weekday-hour combinations. A scenario can be selected for more than one combination, so the trials cover 15 distinct scenarios.

At each junction, a positive difference greater than #mg-l[0.0001] counts as a visible pulse effect. This threshold is an analysis rule, not a measured sensor limit. An effect is visible in #num[51879] of the #num[52080] junction-pulse cases. The other 201 cases have no sample above the threshold. The same cases are used to check the history and horizon chosen in @section:methodology:temporal-window-selection.

For the history comparison, the diagnostic identifies the last sample above the threshold at each junction. A case has settled within a candidate window if this sample occurs by the end of that window. The mean response seen is the average, across visible cases, of the share of each case's 48-hour above-threshold response recorded within the window. @tab:methodology-history-window-coverage also reports the number of visible junction-pulse cases with a later sample above the threshold and the share of pulse trials in which all 31 junctions have settled. Junctions without a visible effect count as settled for this last measure.

At #hour[12], 168 visible junction-pulse cases still have a later effect, and all junctions have settled in #pct[90.83] of pulse trials. At #hour[20], at least one junction has a later effect in 13 of the 1680 pulse trials. At #hour[24], only one trial still has a later effect, with its last visible sample at #hour[29.5]. The mean response seen rounds to #pct[100] because very little of this case's effect remains after #hour[24]. The reduction in late effects and coverage of a complete daily cycle support #hour[24] as a practical history length under the study's feasibility constraints. They do not establish that this history length gives the best forecasts.

The horizon comparison examines when the same pulse effect begins, reaches its peak, and settles. For each visible junction-pulse case, the effect begins at the first sample above the threshold. The peak is the largest positive concentration difference during the 48-hour observation. The effect has settled by a candidate horizon if no later sample within those #hour[48] exceeds the threshold. @tab:methodology-chlorine-response-timing reports the share of cases that have reached each point by the candidate horizon.

By the selected eight-hour horizon, the effect has begun in #pct[99.85] of visible cases, reached its peak in #pct[99.70], and settled in #pct[98.41]. At four hours, #pct[90.30] have settled. Longer horizons cover only small additional shares of cases while requiring longer forecasts and evaluation of more predicted steps for each #acr("MPC") plan. These observations support eight hours as a practical choice under the study's feasibility constraints. They do not establish an empirically optimal horizon, since the diagnostic does not measure forecasting or control performance.

#{
  set par(justify: false)
  set figure(placement: none)

  academic-table(
    [Response timing by candidate history length. The mean response seen is the average, across visible cases, of the share of each case's 48-hour response above #mg-l[0.0001] recorded within the window. A trial is fully settled when none of its 31 junctions has a later sample above this threshold within the 48-hour observation. The mean and trial share are percentages. Values are rounded to at most two decimal places.],
    header: (
      [*History (#hour())*],
      [*Mean response seen (#pct())*],
      [*Cases still visible*],
      [*Trials fully settled (#pct())*],
    ),
    delay-data
      .history
      .map(row => (
        [#row.hours],
        [#rounded(row.mean_response_seen_pct)],
        [#num(row.still_visible_cases)],
        [#percent(row.all_junctions_settled_trials, delay-counts.trials)],
      ))
      .flatten(),
    format: "booktabs",
    columns: (auto, 1fr, 1fr, 1fr),
    align: (right, right, right, right),
    label: <tab:methodology-history-window-coverage>,
  )
}

#{
  set par(justify: false)
  set figure(placement: none)

  academic-table(
    [Timing of the pulse response. Values are percentages of #num[51879] visible junction-pulse cases, rounded to at most two decimal places. A pulse effect is visible above #mg-l[0.0001] and has settled by a horizon if no later sample within the 48-hour observation exceeds this threshold.],
    header: (
      [*Horizon (#hour())*],
      [*Effect begun (#pct())*],
      [*Peak reached (#pct())*],
      [*Effect settled (#pct())*],
    ),
    delay-data
      .horizons
      .map(row => (
        [#row.hours],
        [#percent(row.begun, delay-counts.responsive)],
        [#percent(row.peak_reached, delay-counts.responsive)],
        [#percent(row.settled, delay-counts.responsive)],
      ))
      .flatten(),
    format: "booktabs",
    columns: (auto, 1fr, 1fr, 1fr),
    align: (right, right, right, right),
    label: <tab:methodology-chlorine-response-timing>,
  )
}
