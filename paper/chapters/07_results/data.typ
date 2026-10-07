// Publication inputs are projected once by the exporter. Typst retains seed
// statistics, paired intervals, and figure rendering.
#let metadata = csv("../../results/package_metadata.csv", row-type: dictionary).first()
#let costs = csv("../../results/computation_costs.csv", row-type: dictionary)
#let learning-data = json("../../results/learning-plot-data.json")
#let forecast-data = json("../../results/forecast-plot-data.json")
#let control-data = json("../../results/control-data.json")
#let case-trajectory-data = json("../../results/case-trajectory-data.json")
#let post-cluster-data = json("../../results/post-cluster-data.json")
#assert(control-data.format_version == 1 and forecast-data.format_version == 1)
#assert(post-cluster-data.format_version == 2)
#assert(post-cluster-data.max_step == 24 and post-cluster-data.timestep_seconds == 1800)
#assert(
  (learning-data, forecast-data, control-data, post-cluster-data).all(data => data.data_kind == metadata.data_kind),
  message: "Publication inputs must have the same data kind",
)
#let post-cluster-excess(approach) = {
  assert(post-cluster-data.excess_partition_scope == "main_reference_normal_node_time_exceedances")
  assert(post-cluster-data.excess_partition_alignment == "reference_cluster_clearances")
  assert(post-cluster-data.excess_partition_aggregation == "schedule_scenario_seed_metric_contributions")
  post-cluster-data.excess_partition.find(row => row.approach == approach)
}
#let checkpoints = learning-data.rows.map(row => (
  approach: row.approach,
  nominal_online_interactions: str(row.nominal_interactions),
  actual_online_interactions: str(row.actual_interactions),
))
#let forecast-rows = forecast-data.rows

#let training-seeds = ("17", "42", "123", "1337", "71026")
#let nominal-checkpoints = range(0, 1000001, step: 50000)
#let t-critical-95-df4 = 2.7764451051977987

#let best-validation-checkpoint(approach, seed) = {
  let candidates = learning-data
    .rows
    .filter(row => row.approach == approach and str(row.training_seed) == seed)
    .sorted(key: row => row.nominal_interactions)
  assert(candidates.map(row => row.nominal_interactions) == nominal-checkpoints)
  let highest-return = calc.max(..candidates.map(row => row.evaluation_return))
  candidates.find(row => row.evaluation_return == highest-return)
}
#let selected-validation-checkpoints = training-seeds.map(seed => (
  seed: seed,
  baseline: best-validation-checkpoint("baseline", seed),
  pretrained: best-validation-checkpoint("pretrained", seed),
))

#let mean(values) = {
  let values = values.filter(value => value != none)
  if values.len() == 0 { return none }
  values.sum() / values.len()
}

#let sample-sd(values) = {
  let values = values.filter(value => value != none)
  if values.len() < 2 { return none }
  let center = mean(values)
  calc.sqrt(values.map(value => calc.pow(value - center, 2)).sum() / (values.len() - 1))
}

#import "../../numbers.typ": hour, mg-h-l, num, pct, tonne

#let fixed-string(value, digits: 2) = {
  if value == none { return "n/a" }
  let threshold = 0.5 / calc.pow(10, digits)
  let normalized = if calc.abs(value) < threshold { 0.0 } else { value }
  let rounded = calc.round(normalized, digits: digits)
  if digits <= 0 {
    str(int(rounded))
  } else {
    let parts = str(rounded).split(".")
    let whole = parts.first()
    let fraction = if parts.len() == 1 { "" } else { parts.last() }
    let zeros = range(0, calc.max(0, digits - fraction.len())).map(_ => "0").join()
    whole + "." + fraction + zeros
  }
}

#let fixed(value, digits: 2) = if value == none { [n/a] } else { num(fixed-string(value, digits: digits)) }
#let fixed-pct(value, digits: 2) = if value == none { [n/a] } else { pct(fixed-string(value, digits: digits)) }
#let pm-digits(sd, default: 2, trim-zero: true) = {
  if sd == 0 { return default }
  let scaled = calc.abs(sd)
  let places = 1
  while scaled >= 10 {
    scaled /= 10
    places -= 1
  }
  while scaled < 1 {
    scaled *= 10
    places += 1
  }
  let rounded = calc.round(sd, digits: places)
  if rounded >= calc.pow(10, 2 - places) {
    places -= 1
    rounded = calc.round(sd, digits: places)
  }
  // Omit a trailing zero when it does not carry useful precision.
  if trim-zero and places > 0 and calc.round(rounded, digits: places - 1) == rounded {
    places -= 1
  }
  places
}
#let fixed-like-pm(value, sd, digits: 2, places: auto) = fixed(
  value,
  digits: if places == auto { pm-digits(sd, default: digits) } else { places },
)
// Zero's table parser requires plain numeral text before formatting/alignment.
#let fixed-pm-string(value, sd, digits: 2, places: auto) = if value == none or sd == none { "n/a" } else {
  let places = if places == auto { pm-digits(sd, default: digits) } else { places }
  fixed-string(value, digits: places) + "+-" + fixed-string(sd, digits: places)
}
#let fixed-pm(value, sd, digits: 2, places: auto) = if value == none or sd == none { [n/a] } else {
  let places = if places == auto { pm-digits(sd, default: digits) } else { places }
  // Zero leaves integer uncertainties ungrouped; format both operands when
  // reporting whole-number returns so the same grouping rule applies to each.
  if places <= 0 {
    (
      num(fixed-string(value, digits: places))
        + h(0.16em)
        + sym.plus.minus
        + h(0.16em)
        + num(fixed-string(sd, digits: places))
    )
  } else {
    num(fixed-string(value, digits: places) + "+-" + fixed-string(sd, digits: places))
  }
}

// Return means follow the two-significant-figure SD. Other return values
// retain up to three significant figures, including small paired differences.
#let return-digits(value) = if value == none or value == 0 { 0 } else {
  calc.min(0, 2 - int(calc.floor(calc.log(calc.abs(value), base: 10))))
}
#let return-string(value) = fixed-string(value, digits: return-digits(value))
#let return-num(value) = fixed(value, digits: return-digits(value))
#let return-mean(value, sd) = fixed(value, digits: pm-digits(sd, default: 0, trim-zero: false))
#let return-pm-string(value, sd) = fixed-pm-string(
  value,
  sd,
  places: pm-digits(sd, default: 0, trim-zero: false),
)
#let return-pm(value, sd) = fixed-pm(
  value,
  sd,
  places: pm-digits(sd, default: 0, trim-zero: false),
)

#let hours(seconds) = fixed(seconds / 3600, digits: 3)
#let duration-text(seconds, with-about: true) = {
  let total-minutes = int(calc.round(seconds / 60))
  let hours = int(calc.floor(total-minutes / 60))
  let minutes = calc.rem(total-minutes, 60)
  let parts = ()
  if hours > 0 {
    parts.push([#num(hours) #(if hours == 1 { "hour" } else { "hours" })])
  }
  if minutes > 0 or hours == 0 {
    parts.push([#num(minutes) #(if minutes == 1 { "minute" } else { "minutes" })])
  }
  if calc.abs(total-minutes * 60 - seconds) > 0.000001 and with-about {
    [about #parts.join([ and ])]
  } else {
    [#parts.join([ and ])]
  }
}

#let cost-device(cost) = if cost.device == "cuda" { [GPU] } else if cost.device == "CPU" {
  [CPU]
} else { [Not recorded] }


#let matching-measurements(approach, metric, component, split: none, seed: none, nominal: none) = {
  assert(split == "test" and (nominal == none or nominal == 1000000))
  control-data.rows.filter(row => (
    row.approach == approach
      and row.metric == metric
      and row.component == component
      and (seed == none or row.seed == seed)
  ))
}

#let seed-summary(approach, seed, nominal, split, metric, component) = {
  if split == "validation" {
    assert(metric == "evaluation_return" and component == "total")
    let rows = learning-data.rows.filter(row => (
      row.approach == approach and str(row.training_seed) == seed and row.nominal_interactions == nominal
    ))
    assert(rows.len() == 1, message: "Missing or duplicate learning-curve point")
    rows.first().evaluation_return
  } else {
    mean(
      matching-measurements(approach, metric, component, split: split, seed: seed, nominal: nominal).map(row => {
        row.value
      }),
    )
  }
}

#let summarize-pair(nominal, split, metric, component) = {
  let baseline = training-seeds.map(seed => seed-summary(
    "baseline",
    seed,
    nominal,
    split,
    metric,
    component,
  ))
  let pretrained = training-seeds.map(seed => seed-summary(
    "pretrained",
    seed,
    nominal,
    split,
    metric,
    component,
  ))
  let differences = range(training-seeds.len()).map(index => {
    if pretrained.at(index) == none or baseline.at(index) == none { none } else {
      pretrained.at(index) - baseline.at(index)
    }
  })
  let difference-mean = mean(differences)
  let half-width = if differences.all(value => value != none) {
    t-critical-95-df4 * sample-sd(differences) / calc.sqrt(differences.len())
  }
  (
    baseline-values: baseline,
    pretrained-values: pretrained,
    differences: differences,
    baseline-mean: mean(baseline),
    baseline-sd: sample-sd(baseline),
    pretrained-mean: mean(pretrained),
    pretrained-sd: sample-sd(pretrained),
    difference-mean: difference-mean,
    ci-low: if half-width != none { difference-mean - half-width },
    ci-high: if half-width != none { difference-mean + half-width },
  )
}

#let learning-summaries = (
  nominal-checkpoints
    .filter(nominal => checkpoints.any(row => row.at("nominal_online_interactions") == str(nominal)))
    .map(nominal => {
      let summary = summarize-pair(nominal, "validation", "evaluation_return", "total")
      summary.insert("nominal", nominal)
      summary
    })
)

#let actual-interactions(nominal) = {
  let values = learning-data
    .rows
    .filter(row => row.nominal_interactions == nominal)
    .map(row => row.actual_interactions)
    .dedup()
  assert(values.len() == 1, message: "Interaction counts must agree across policies and seeds")
  values.first()
}

#let metric-definitions = (
  (
    id: "chlorine_shortfall_fraction",
    component: "main_normal",
    symbol: [$f_("low")^"N"$],
    unit: [#pct()],
    scale: 100.0,
    digits: 4,
  ),
  (
    id: "normal_chlorine_excess_fraction",
    component: "main_normal",
    symbol: [$f_("high")^"N"$],
    unit: [#pct()],
    scale: 100.0,
    digits: 3,
  ),
  (
    id: "arsenic_burden",
    component: "total",
    symbol: [$B^"E"$],
    unit: [#mg-h-l()],
    scale: 1.0,
    digits: 1,
  ),
  (
    id: "chlorine_ceiling_violation_fraction",
    component: "total",
    symbol: [$f_("cap")^"E"$],
    unit: [#pct()],
    scale: 100.0,
    digits: 3,
  ),
  (
    id: "arsenic_clearance_lead",
    component: "total",
    symbol: [$Delta_("clear")$],
    unit: [#hour()],
    scale: 1.0,
    digits: 2,
  ),
  (
    id: "chlorine_use",
    component: "total",
    symbol: [$U_("eval")$],
    unit: [#tonne()],
    scale: 0.000000001,
    digits: 3,
  ),
)

#let physical-summaries = metric-definitions.map(definition => {
  let summary = summarize-pair(1000000, "test", definition.id, definition.component)
  let baseline-rows = matching-measurements(
    "baseline",
    definition.id,
    definition.component,
    split: "test",
    nominal: 1000000,
  )
  summary.insert("definition", definition)
  summary.insert("applicable", baseline-rows.map(row => row.applicable_count).sum())
  summary.insert("not-applicable", baseline-rows.map(row => row.not_applicable_count).sum())
  summary
})

#let metric-summary(id) = physical-summaries.find(summary => summary.definition.id == id)

#let case-burden-points = control-data.case_burdens
#assert(case-burden-points.len() == 100)
#let joint-control-points = (
  ("baseline", "pretrained")
    .map(approach => training-seeds
      .enumerate()
      .map(((i, seed)) => (
        approach: approach,
        seed: seed,
        burden: metric-summary("arsenic_burden").at(approach + "-values").at(i),
        clearance: metric-summary("arsenic_clearance_lead").at(approach + "-values").at(i),
      )))
    .flatten()
)

#let format-metric(definition, value) = fixed(
  if value == none { none } else { value * definition.scale },
  digits: definition.digits,
)
#let format-metric-pct-like-pm(definition, value, sd) = fixed-pct(
  value * definition.scale,
  digits: pm-digits(sd * definition.scale, default: definition.digits),
)

// Table cells contain only numeral strings. Zero formats them at table level.
// Scaling and displayed precision are explicit at each table call site.
#let table-value(value, sd: none, digits: 2, separator: "+-") = {
  if value == none { return "n/a" }
  let values = if sd == none { (value,) } else { (value, sd) }
  values
    .map(v => {
      let rounded = calc.round(if calc.abs(v) < 0.5 / calc.pow(10, digits) { 0.0 } else { v }, digits: digits)
      if digits <= 0 { return str(int(rounded)) }
      let parts = str(rounded).split(".")
      let fraction = if parts.len() > 1 { parts.last() } else { "" }
      parts.first() + "." + fraction + "0" * calc.max(0, digits - fraction.len())
    })
    .join(separator)
}
#let table-interval(summary) = context {
  // A common width keeps "to" centered at the same position in every row.
  let endpoints(item) = (item.at("ci-low"), item.at("ci-high")).map(value => table-value(
    value * item.definition.scale,
    digits: item.definition.digits,
  ))
  let side-width = calc.max(..physical-summaries.map(item => calc.max(
    ..endpoints(item).map(value => measure(value).width),
  )))
  let gutter = 0.35em
  let (low, high) = endpoints(summary)
  box(width: 2 * side-width + measure([to]).width + 2 * gutter, grid(
    columns: (1fr, auto, 1fr),
    column-gutter: gutter,
    align: (right, center, left),
    low, [to], high,
  ))
}

#let confidence-interval-excluding-zero = (
  physical-summaries
    .filter(summary => {
      summary.at("ci-low") != none and (summary.at("ci-low") > 0 or summary.at("ci-high") < 0)
    })
    .len()
)

#let zero-reference-burden = mean(control-data.reference_values)

#let component-mean(approach, metric, component) = mean(training-seeds.map(seed => seed-summary(
  approach,
  seed,
  1000000,
  "test",
  metric,
  component,
)))

#let component-sd(approach, metric, component) = sample-sd(training-seeds.map(seed => seed-summary(
  approach,
  seed,
  1000000,
  "test",
  metric,
  component,
)))

#let event-components = (
  (
    label: [Arsenic burden, baseline],
    main: component-mean("baseline", "arsenic_burden", "main"),
    main-sd: component-sd("baseline", "arsenic_burden", "main"),
    tail: component-mean("baseline", "arsenic_burden", "tail"),
    tail-sd: component-sd("baseline", "arsenic_burden", "tail"),
    total: component-mean("baseline", "arsenic_burden", "total"),
    total-sd: component-sd("baseline", "arsenic_burden", "total"),
  ),
  (
    label: [Arsenic burden, pretrained],
    main: component-mean("pretrained", "arsenic_burden", "main"),
    main-sd: component-sd("pretrained", "arsenic_burden", "main"),
    tail: component-mean("pretrained", "arsenic_burden", "tail"),
    tail-sd: component-sd("pretrained", "arsenic_burden", "tail"),
    total: component-mean("pretrained", "arsenic_burden", "total"),
    total-sd: component-sd("pretrained", "arsenic_burden", "total"),
  ),
)

#let chlorine-use-components = (
  baseline: (
    main-normal: component-mean("baseline", "chlorine_use", "main_normal"),
    main-event: component-mean("baseline", "chlorine_use", "main_event"),
    tail-event: component-mean("baseline", "chlorine_use", "tail_event"),
    total: component-mean("baseline", "chlorine_use", "total"),
  ),
  pretrained: (
    main-normal: component-mean("pretrained", "chlorine_use", "main_normal"),
    main-event: component-mean("pretrained", "chlorine_use", "main_event"),
    tail-event: component-mean("pretrained", "chlorine_use", "tail_event"),
    total: component-mean("pretrained", "chlorine_use", "total"),
  ),
)

#let cost-summary(stage, approach) = {
  let rows = costs.filter(row => row.at("stage") == stage and row.at("approach") == approach)
  let values = rows.map(row => float(row.at("wall_time_seconds")))
  assert(values.len() > 0, message: "No matching cost rows found")
  (
    rows: rows,
    count: rows.len(),
    total: values.sum(),
    mean: mean(values),
    sd: if values.len() > 1 { sample-sd(values) } else { none },
    device: rows.first().at("execution_device"),
    workers: rows.first().at("worker_count"),
  )
}

#let offline-trajectories-cost = cost-summary("offline_trajectory_generation", "shared")
#let hydraulic-profiles-cost = cost-summary("hydraulic_profile_generation", "shared")
#let forecaster-cost = cost-summary("forecaster_training", "pretrained")
#let mpc-training-cost = cost-summary("mpc_teacher_training_generation", "pretrained")
#let mpc-fold-cost = cost-summary("mpc_teacher_fold_generation", "pretrained")
#let mpc-validation-cost = cost-summary("mpc_teacher_validation_generation", "pretrained")
#let mpc-cost = {
  let values = training-seeds.map(seed => (
    float(mpc-training-cost.rows.find(row => row.training_seed == seed).wall_time_seconds)
      + float(mpc-validation-cost.rows.find(row => row.training_seed == seed).wall_time_seconds)
  ))
  (total: values.sum(), mean: mean(values), sd: sample-sd(values))
}
#let behavior-cloning-cost = cost-summary("behavior_cloning", "pretrained")
#let baseline-ppo-cost = cost-summary("ppo_training", "baseline")
#let pretrained-ppo-cost = cost-summary("ppo_training", "pretrained")
#let baseline-control-evaluation-cost = cost-summary("control_evaluation", "baseline")
#let pretrained-control-evaluation-cost = cost-summary("control_evaluation", "pretrained")
#let forecast-evaluation-cost = cost-summary("forecast_evaluation", "pretrained")

#let offline-preparation-seconds = (
  offline-trajectories-cost.total + forecaster-cost.total + mpc-cost.total + behavior-cloning-cost.total
)
#let baseline-total-seconds = baseline-ppo-cost.total + baseline-control-evaluation-cost.total
#let pretrained-total-seconds = (
  offline-preparation-seconds + pretrained-ppo-cost.total + pretrained-control-evaluation-cost.total
)

#let seed-cost-summary(approach, stage: none, exclude-stages: ()) = {
  let values = training-seeds.map(seed => costs
    .filter(row => (
      row.at("approach") == approach
        and row.at("training_seed") == seed
        and (stage == none or row.at("stage") == stage)
        and row.at("stage") != "mpc_teacher_fold_generation"
        and not exclude-stages.contains(row.at("stage"))
    ))
    .map(row => float(row.at("wall_time_seconds")))
    .sum())
  (mean: mean(values), sd: sample-sd(values))
}
#let baseline-seed-cost = seed-cost-summary("baseline")
#let pretrained-seed-cost = seed-cost-summary("pretrained", exclude-stages: ("forecast_evaluation",))
#let forecaster-seed-cost = seed-cost-summary("pretrained", stage: "forecaster_training")
#let forecast-evaluation-seed-cost = seed-cost-summary("pretrained", stage: "forecast_evaluation")
#let offline-preparation-per-seed-seconds = (
  (
    forecaster-cost.total + mpc-cost.total + behavior-cloning-cost.total
  )
    / training-seeds.len()
)

#let forecast-error(seed, fold, step: none) = {
  let selected = forecast-rows.filter(row => (
    str(row.training_seed) == str(seed)
      and row.fold_id == fold
      and row.horizon_step == if step == none { 0 } else { step }
  ))
  assert(selected.len() == 1, message: "Missing or duplicate forecast model/horizon error")
  (mae: selected.first().mae, rmse: selected.first().rmse)
}

#let forecast-complete = forecast-data.data_kind != "pending" and forecast-rows.len() == 425
#let forecast-seed-errors(step: none) = training-seeds.map(seed => {
  let folds = range(0, 5).map(fold => forecast-error(seed, fold, step: step))
  (mae: mean(folds.map(value => value.mae)), rmse: mean(folds.map(value => value.rmse)))
})
#let forecast-seed-summaries = if forecast-complete { forecast-seed-errors() } else { () }
#let forecast-mae-mean = if forecast-complete { mean(forecast-seed-summaries.map(value => value.mae)) }
#let forecast-mae-sd = if forecast-complete { sample-sd(forecast-seed-summaries.map(value => value.mae)) }
#let forecast-rmse-mean = if forecast-complete { mean(forecast-seed-summaries.map(value => value.rmse)) }
#let forecast-rmse-sd = if forecast-complete { sample-sd(forecast-seed-summaries.map(value => value.rmse)) }
#let forecast-step-summaries = if forecast-complete {
  range(1, 17).map(step => {
    let seeds = forecast-seed-errors(step: step)
    (
      step: step,
      mae-mean: mean(seeds.map(value => value.mae)),
      mae-sd: sample-sd(seeds.map(value => value.mae)),
      rmse-mean: mean(seeds.map(value => value.rmse)),
      rmse-sd: sample-sd(seeds.map(value => value.rmse)),
    )
  })
} else { () }

#let selected-cases = control-data.selected.case_ids
#let selected-case-id = selected-cases.first()
#let selected-case-scenario = control-data.selected.scenario_id
#let selected-case-schedule = control-data.selected.schedule_id
#let selected-case-burdens = control-data.selected.burden_by_seed
#let selected-case-reason = control-data.selected.reason
#assert(
  case-trajectory-data.scenario_id == selected-case-scenario
    and case-trajectory-data.schedule_id == selected-case-schedule,
  message: "Case trajectories and control metrics must describe the same illustrative case",
)
#let selected-case-days = case-trajectory-data.time_days.last() - case-trajectory-data.time_days.first()
#let selected-case-events = case-trajectory-data.source_events.sorted(key: event => event.start_time_seconds)
#let selected-case-interval-minutes = calc.round(
  (case-trajectory-data.time_days.at(1) - case-trajectory-data.time_days.first()) * 1440,
  digits: 6,
)
#let selected-case-seed-burden = selected-case-burdens.find(row => str(row.seed) == str(case-trajectory-data.seed))
#let selected-case-lower-seeds = selected-case-burdens.filter(row => row.pretrained < row.baseline).map(row => row.seed)
#let selected-case-higher-seeds = (
  selected-case-burdens.filter(row => row.pretrained > row.baseline).map(row => row.seed)
)
#let selected-case-window-peak(values, index) = {
  let window = case-trajectory-data.reference_windows.at(index)
  calc.max(..values.slice(window.start_index, window.end_index + 1))
}

#let selected-case-value(approach, metric, component) = {
  let rows = control-data.selected.rows.filter(row => (
    row.approach == approach and row.metric == metric and row.component == component
  ))
  assert(rows.len() == 1, message: "Missing or duplicate selected-case metric")
  rows.first().value
}

#let selected-case-values = metric-definitions.map(definition => (
  definition: definition,
  baseline: selected-case-value("baseline", definition.id, definition.component),
  pretrained: selected-case-value("pretrained", definition.id, definition.component),
  reference: if definition.id == "arsenic_burden" {
    selected-case-value("zero_injection", definition.id, definition.component)
  } else {
    none
  },
))

#let clearance-support = control-data.clearance_support
#let selected-case-clearance-events = (
  control-data.selected.clearance_events_by_seed.find(row => str(row.seed) == str(case-trajectory-data.seed)).events
)
