#import "../../numbers.typ": *
#import "../../utils.typ": chapter-ref, continued-tables
#import "@preview/acrostiche:0.7.0": acr
#import "@preview/akatable:0.1.0": academic-table
#import "@preview/zero:0.7.0": format-table
#import "../07_results/evaluation-plots.typ": forecast-figure
#import "../07_results/data.typ": (
  checkpoints, forecast-complete, forecast-error, forecast-seed-summaries, forecast-step-summaries, learning-summaries,
  metric-summary, return-pm-string, return-string, selected-validation-checkpoints, table-value, training-seeds,
)

= Numerical Results <section:appendix:numerical-results>

This appendix provides the numerical values underlying the learning curves, selected policy checkpoints, physical control comparison, and forecasting results reported in #chapter-ref(<chapter:results>). The tables follow the aggregation defined in #chapter-ref(<chapter:evaluation>).


== Validation Return and Selected Policy Checkpoints

@tab:appendix-validation-return lists mean validation returns and sample standard deviations across the five training seeds for the baseline approach and pretrained approach. Mean paired differences are calculated as pretrained minus baseline. Alongside each nominal interaction threshold, the table reports the actual interaction count of the evaluated checkpoint.

For each approach and training seed, the checkpoint with the highest mean validation return is selected following @section:evaluation:policy-selection. @tab:validation-selected-checkpoints lists the selected nominal interaction counts and corresponding mean validation returns.


#{
  show: continued-tables(5)
  set par(justify: false)
  show table: format-table(auto, auto, auto, auto, auto)
  show regex("[0-9]{5,}"): it => zero.num(it.text)

  academic-table(
    [Validation return at all nominal checkpoints. The sample standard deviation is calculated across five seeds. Paired differences are calculated as pretrained minus baseline. Rounding follows @section:evaluation:statistical-analysis.],
    header: (
      table.cell(colspan: 2, align: center)[*Interactions*],
      table.cell(rowspan: 2, align: right + horizon)[*Baseline*],
      table.cell(rowspan: 2, align: right + horizon)[*Pretrained*],
      table.cell(rowspan: 2, align: right + horizon)[*Difference*],
      table.hline(start: 0, end: 2, stroke: 0.5pt),
      [*Nominal*],
      [*Actual*],
    ),
    learning-summaries
      .map(summary => {
        let actual = checkpoints
          .filter(row => (
            row.at("approach") != "zero_injection" and int(row.at("nominal_online_interactions")) == summary.nominal
          ))
          .map(row => int(row.at("actual_online_interactions")))
          .sorted()
          .dedup()
        assert(actual.len() == 1, message: "Report differing actual interaction counts separately")
        (
          [#summary.nominal],
          [#actual.first()],
          [#return-pm-string(summary.at("baseline-mean"), summary.at("baseline-sd"))],
          [#return-pm-string(summary.at("pretrained-mean"), summary.at("pretrained-sd"))],
          [#return-string(summary.at("difference-mean"))],
        )
      })
      .flatten(),
    format: "booktabs",
    columns: (auto, 1fr, 1.5fr, 1.2fr, 1fr),
    column-gutter: (0em, 1em, 0em, 0em),
    align: (right, right, right, right, right),
    label: <tab:appendix-validation-return>,
  )
}

#{
  set par(justify: false)
  set figure(placement: none)
  show table: format-table(none, auto, auto, auto, auto)

  academic-table(
    [Nominal online interaction counts and mean validation returns at the selected checkpoints for each training seed. Returns follow the scenario weighting defined in @section:evaluation:policy-selection and the rounding in @section:evaluation:statistical-analysis.],
    header: (
      table.cell(rowspan: 2, align: left + horizon)[*Training seed*],
      table.cell(colspan: 2, align: center)[*Baseline*],
      table.cell(colspan: 2, align: center)[*Pretrained*],
      table.hline(start: 1, end: 3, stroke: 0.5pt),
      table.hline(start: 3, end: 5, stroke: 0.5pt),
      [*Interactions*],
      [*Return*],
      [*Interactions*],
      [*Return*],
    ),
    selected-validation-checkpoints
      .map(row => (
        [#row.seed],
        [#table-value(row.baseline.nominal_interactions, digits: 0)],
        [#return-string(row.baseline.evaluation_return)],
        [#table-value(row.pretrained.nominal_interactions, digits: 0)],
        [#return-string(row.pretrained.evaluation_return)],
      ))
      .flatten(),
    format: "booktabs",
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    column-gutter: (1em, 0em, 1.4em, 0em),
    align: (left, right, right, right, right),
    label: <tab:validation-selected-checkpoints>,
  )
}


== Paired Physical Metric Differences

@tab:seed-differences lists the individual paired differences underlying the aggregate test results in @tab:final-control-metrics. For each physical metric, the selected baseline policy's value is subtracted from the selected pretrained policy's value within the same training seed.

#let shortfall = metric-summary("chlorine_shortfall_fraction")
#let normal-excess = metric-summary("normal_chlorine_excess_fraction")
#let arsenic-burden = metric-summary("arsenic_burden")
#let clearance-lead = metric-summary("arsenic_clearance_lead")
#let ceiling-violation = metric-summary("chlorine_ceiling_violation_fraction")
#let chlorine-use = metric-summary("chlorine_use")

#{
  show: continued-tables(6)
  set par(justify: false)

  academic-table(
    [Paired test-metric differences for the individual training seeds, calculated as the selected pretrained policy's value minus the selected baseline policy's value in the units shown. The normal-operation chlorine shortfall and excess fractions are $f_("low")^"N"$ and $f_("high")^"N"$, respectively. The remaining metrics are transported arsenic burden $B^"E"$, mean arsenic clearance lead $Delta_("clear")$ relative to the zero-chlorine reference, the event-time ceiling-violation fraction $f_("cap")^"E"$, and chlorine use $U_("eval")$. pp = percentage points.],
    header: ([*Metric*], [*Seed 17*], [*Seed 42*], [*Seed 123*], [*Seed 1337*], [*Seed 71026*]),
    (
      [$f_("low")^"N"$ (pp)],
      [#table-value(shortfall.differences.at(0) * shortfall.definition.scale, digits: shortfall.definition.digits)],
      [#table-value(shortfall.differences.at(1) * shortfall.definition.scale, digits: shortfall.definition.digits)],
      [#table-value(shortfall.differences.at(2) * shortfall.definition.scale, digits: shortfall.definition.digits)],
      [#table-value(shortfall.differences.at(3) * shortfall.definition.scale, digits: shortfall.definition.digits)],
      [#table-value(shortfall.differences.at(4) * shortfall.definition.scale, digits: shortfall.definition.digits)],

      [$f_("high")^"N"$ (pp)],
      [#table-value(
        normal-excess.differences.at(0) * normal-excess.definition.scale,
        digits: normal-excess.definition.digits,
      )],
      [#table-value(
        normal-excess.differences.at(1) * normal-excess.definition.scale,
        digits: normal-excess.definition.digits,
      )],
      [#table-value(
        normal-excess.differences.at(2) * normal-excess.definition.scale,
        digits: normal-excess.definition.digits,
      )],
      [#table-value(
        normal-excess.differences.at(3) * normal-excess.definition.scale,
        digits: normal-excess.definition.digits,
      )],
      [#table-value(
        normal-excess.differences.at(4) * normal-excess.definition.scale,
        digits: normal-excess.definition.digits,
      )],

      [$B^"E"$ (#mg-h-l())],
      [#table-value(
        arsenic-burden.differences.at(0) * arsenic-burden.definition.scale,
        digits: arsenic-burden.definition.digits,
      )],
      [#table-value(
        arsenic-burden.differences.at(1) * arsenic-burden.definition.scale,
        digits: arsenic-burden.definition.digits,
      )],
      [#table-value(
        arsenic-burden.differences.at(2) * arsenic-burden.definition.scale,
        digits: arsenic-burden.definition.digits,
      )],
      [#table-value(
        arsenic-burden.differences.at(3) * arsenic-burden.definition.scale,
        digits: arsenic-burden.definition.digits,
      )],
      [#table-value(
        arsenic-burden.differences.at(4) * arsenic-burden.definition.scale,
        digits: arsenic-burden.definition.digits,
      )],

      [$Delta_("clear")$ (#hour())],
      [#table-value(
        clearance-lead.differences.at(0) * clearance-lead.definition.scale,
        digits: clearance-lead.definition.digits,
      )],
      [#table-value(
        clearance-lead.differences.at(1) * clearance-lead.definition.scale,
        digits: clearance-lead.definition.digits,
      )],
      [#table-value(
        clearance-lead.differences.at(2) * clearance-lead.definition.scale,
        digits: clearance-lead.definition.digits,
      )],
      [#table-value(
        clearance-lead.differences.at(3) * clearance-lead.definition.scale,
        digits: clearance-lead.definition.digits,
      )],
      [#table-value(
        clearance-lead.differences.at(4) * clearance-lead.definition.scale,
        digits: clearance-lead.definition.digits,
      )],

      [$f_("cap")^"E"$ (pp)],
      [#table-value(
        ceiling-violation.differences.at(0) * ceiling-violation.definition.scale,
        digits: ceiling-violation.definition.digits,
      )],
      [#table-value(
        ceiling-violation.differences.at(1) * ceiling-violation.definition.scale,
        digits: ceiling-violation.definition.digits,
      )],
      [#table-value(
        ceiling-violation.differences.at(2) * ceiling-violation.definition.scale,
        digits: ceiling-violation.definition.digits,
      )],
      [#table-value(
        ceiling-violation.differences.at(3) * ceiling-violation.definition.scale,
        digits: ceiling-violation.definition.digits,
      )],
      [#table-value(
        ceiling-violation.differences.at(4) * ceiling-violation.definition.scale,
        digits: ceiling-violation.definition.digits,
      )],

      [$U_("eval")$ (#tonne())],
      [#table-value(
        chlorine-use.differences.at(0) * chlorine-use.definition.scale,
        digits: chlorine-use.definition.digits,
      )],
      [#table-value(
        chlorine-use.differences.at(1) * chlorine-use.definition.scale,
        digits: chlorine-use.definition.digits,
      )],
      [#table-value(
        chlorine-use.differences.at(2) * chlorine-use.definition.scale,
        digits: chlorine-use.definition.digits,
      )],
      [#table-value(
        chlorine-use.differences.at(3) * chlorine-use.definition.scale,
        digits: chlorine-use.definition.digits,
      )],
      [#table-value(
        chlorine-use.differences.at(4) * chlorine-use.definition.scale,
        digits: chlorine-use.definition.digits,
      )],
    ),
    format: "booktabs",
    columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right, right),
    label: <tab:seed-differences>,
  )
}


== Forecast Errors

@fig:forecast-full-horizon and @tab:appendix-forecast-model-errors report #acr("MAE") and #acr("RMSE") over the complete eight-hour horizon for all 25 fold-specific models. Each model's errors give equal weight to the ten forecasting test scenarios. In the table, the mean row for each training seed averages the errors of its five models.

The full-horizon errors are complemented by @tab:appendix-forecast-horizon-errors, which lists the values shown in @fig:forecast-horizon-errors for each half-hour horizon step. At each step, the five fold-specific model errors are averaged within each training seed. The reported mean and sample standard deviation are then calculated across the five seed means.

#figure(
  forecast-figure(),
  placement: none,
  caption: [Chlorine-forecast errors over the full eight-hour horizon by training seed, with (a) #acr("MAE") and (b) #acr("RMSE"). Errors are evaluated on the ten forecasting test scenarios under prescribed injection profiles. Colors identify the five folds consistently across both panels. Black horizontal bars mark seed means.],
) <fig:forecast-full-horizon>

#{
  show: continued-tables(4)
  set par(justify: false)
  show table: format-table(none, none, auto, auto)

  academic-table(
    [Forecast errors over the full eight-hour horizon by training seed and fold. Errors are evaluated on the ten forecasting test scenarios under prescribed injection profiles, expressed in #mg-l(), and rounded to five decimal places.],
    header: ([*Training seed*], [*Fold*], [*#acr("MAE") (#mg-l())*], [*#acr("RMSE") (#mg-l())*]),
    if forecast-complete {
      training-seeds
        .enumerate()
        .map(((index, seed)) => {
          let model-rows = range(5)
            .map(fold => {
              let errors = forecast-error(seed, fold)
              (
                [#identifier(seed)],
                [#(fold + 1)],
                [#table-value(errors.mae, digits: 5)],
                [#table-value(errors.rmse, digits: 5)],
              )
            })
            .flatten()
          let seed-mean = forecast-seed-summaries.at(index)
          (
            model-rows
              + (
                [#identifier(seed)],
                [Mean],
                [#table-value(seed-mean.mae, digits: 5)],
                [#table-value(seed-mean.rmse, digits: 5)],
              )
          )
        })
        .flatten()
    } else { () },
    format: "booktabs",
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, center, right, right),
    label: <tab:appendix-forecast-model-errors>,
  )
}

#pagebreak(weak: true)
#{
  show: continued-tables(4)
  set par(justify: false)
  show table: format-table(auto, auto, auto, auto)

  academic-table(
    [Forecast errors at each half-hour horizon step in #mg-l(), evaluated on the ten forecasting test scenarios under prescribed injection profiles. Means and sample standard deviations across seed means are rounded to five decimal places.],
    header: ([*Step*], [*Horizon (#hour())*], [*#acr("MAE") (#mg-l())*], [*#acr("RMSE") (#mg-l())*]),
    forecast-step-summaries
      .map(summary => (
        [#summary.step],
        [#table-value(summary.step / 2, digits: 1)],
        [#table-value(summary.at("mae-mean"), sd: summary.at("mae-sd"), digits: 5)],
        [#table-value(summary.at("rmse-mean"), sd: summary.at("rmse-sd"), digits: 5)],
      ))
      .flatten(),
    format: "booktabs",
    columns: (auto, 1fr, 1fr, 1fr),
    align: (right, right, right, right),
    label: <tab:appendix-forecast-horizon-errors>,
  )
}
