#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": grouped-bar-chart, multi-line-chart, multi-scatter-plot

#let seeds = (17, 42, 123, 1337, 71026)
#let folds = series-colors
#let policy-colors = series-colors.slice(0, 2)
#let label = plot-label
// Match Dataset's preflowed titles so Primaviz measures the rotated width.
#let y-axis-label(body) = rotate(90deg, rotate(-90deg, reflow: true, plot-label(body)))
// Reserve only the space occupied by the native axis labels. This keeps the
// complete figure inside the text width without empty fixed side margins.
#let axis-tick-text(value) = {
  if calc.abs(value) >= 1000000 { str(calc.round(value / 1000000, digits: 1)) + "M" } else if calc.abs(value) >= 10000 {
    str(calc.round(value / 1000, digits: 1)) + "k"
  } else { str(calc.round(value, digits: 8)) }
}
#let chart-side-padding(title, bounds, last-x: []) = (
  axis-padding-left: measure(rotate(-90deg, reflow: true, plot-label(title))).width
    + axis-title-gap
    + y-tick-gap
    + calc.max(..(bounds.low, bounds.high, 0).map(v => measure(plot-label(axis-tick-text(v))).width)),
  axis-padding-right: measure(plot-label(last-x)).width / 2,
)
#let mean(values) = values.sum() / values.len()
#let sample-sd(values) = {
  let mu = mean(values)
  calc.sqrt(values.map(v => calc.pow(v - mu, 2)).sum() / (values.len() - 1))
}
#let seed-mean-marker = legend-line(black, stroke: line-size + black)
#let titled-panel(title, body) = block(width: 100%, stack(
  dir: ttb,
  spacing: title-spacing,
  align(center, panel-title(title)),
  body,
))

// Chart composition uses shared styles. Primaviz draws axes, series, and
// markers. Panels reserve only the shared axis footer, excluding the
// package's unused legend allowance.
#let results-chart-theme = (
  chart-theme
    + (
      axis-padding-left: 48pt,
      axis-padding-right: 10pt,
      label-offset: y-tick-gap,
      legend-gap: 0pt,
    )
)

// Primaviz 0.10.0 has no ribbon series. Match its D3-style axis bounds so
// this mean +/- SD polygon shares the native series' coordinate system.
#let ribbon-bounds(values) = {
  let low = calc.min(..values)
  let high = calc.max(..values)
  let span = if high == low { 1.0 } else { high - low }
  let raw-step = span / 4
  let base = calc.pow(10, calc.floor(calc.log(raw-step, base: 10)))
  let error = raw-step / base
  let step = (
    base
      * if error >= calc.sqrt(50) { 10 } else if error >= calc.sqrt(10) { 5 } else if error >= calc.sqrt(2) { 2 } else {
        1
      }
  )
  (low: calc.floor(low / step) * step, high: calc.ceil(high / step) * step)
}
#let sd-ribbon(points, width, height, bounds, fill, nonnegative: false) = {
  let point(p, sign) = {
    let value = p.mean + sign * p.sd
    if nonnegative { value = calc.max(0, value) }
    (p.x * width, (bounds.high - value) / (bounds.high - bounds.low) * height)
  }
  polygon(
    ..points.map(p => point(p, 1)),
    ..points.rev().map(p => point(p, -1)),
    fill: fill,
    stroke: none,
  )
}

// Pairing and equality guides are content annotations in the native scatter
// coordinate system, using the shared path primitive.
#let scatter-link(from, to, width, height, x-bounds, y-bounds, stroke) = (
  type: "content",
  x: from.at(0),
  y: from.at(1),
  anchor: "top-left",
  body: plot-path(
    (
      (0pt, 0pt),
      (
        (to.at(0) - from.at(0)) / (x-bounds.high - x-bounds.low) * width,
        (from.at(1) - to.at(1)) / (y-bounds.high - y-bounds.low) * height,
      ),
    ),
    stroke,
  ),
)

#let case-comparison-panel(cases, bounds, padding, plot-size) = {
  let upper = bounds.high
  let annotations = (scatter-link((0, 0), (upper, upper), plot-size, plot-size, bounds, bounds, boundary-rule),)
  annotations += cases.map(row => (
    type: "point",
    x: row.baseline,
    y: row.pretrained,
    radius: 1.9pt,
    fill: accent.transparentize(20%),
    stroke: none,
  ))
  let width = plot-size + padding.axis-padding-left + padding.axis-padding-right
  box(width: width, height: plot-size + axis-footer-height)[
    #place(top + left, multi-scatter-plot(
      (
        series: (
          (name: "Test cases", points: cases.map(row => (row.baseline, row.pretrained))),
          (name: "Axis extent", points: ((0, 0), (upper, upper))),
        ),
      ),
      width: width,
      height: plot-size,
      point-size: 0pt,
      show-legend: false,
      x-label: [Selected baseline policy arsenic burden (#mg-h-l())],
      y-label: y-axis-label([Selected pretrained policy arsenic burden (#mg-h-l())]),
      annotations: annotations,
      // Native scatter charts add 10pt to the theme's left padding.
      theme: results-chart-theme
        + padding
        + (
          axis-padding-left: padding.axis-padding-left - 10pt,
          marker-stroke: none,
        ),
    ))
  ]
}

#let joint-control-panel(points, x-bounds, y-bounds, padding, plot-size) = {
  let annotations = ()
  for seed in seeds {
    let pair = points.filter(p => p.seed == str(seed))
    assert(pair.len() == 2)
    annotations.push(scatter-link(
      (pair.first().burden, pair.first().clearance),
      (pair.last().burden, pair.last().clearance),
      plot-size,
      plot-size,
      x-bounds,
      y-bounds,
      0.6pt + muted.transparentize(40%),
    ))
  }
  for p in points {
    let color = policy-colors.at(if p.approach == "baseline" { 0 } else { 1 })
    annotations.push((type: "point", x: p.burden, y: p.clearance, radius: 2.5pt, fill: color, stroke: none))
    // Separate the neighboring pretrained labels without moving points.
    let left-label = p.approach == "pretrained" and p.seed == "71026"
    annotations.push((
      type: "content",
      x: p.burden,
      y: p.clearance,
      anchor: if left-label { "right" } else { "left" },
      dx: if left-label { -5pt } else { 5pt },
      dy: if left-label { 5pt } else { -5pt },
      body: plot-label(p.seed, fill: color, size: annotation-size),
    ))
  }
  let width = plot-size + padding.axis-padding-left + padding.axis-padding-right
  box(width: width, height: plot-size + axis-footer-height)[
    #place(top + left, multi-scatter-plot(
      (
        series: ("baseline", "pretrained").map(approach => (
          name: approach,
          points: points.filter(p => p.approach == approach).map(p => (p.burden, p.clearance)),
        )),
      ),
      width: width,
      height: plot-size,
      point-size: 0pt,
      show-legend: false,
      x-label: [Arsenic burden (#mg-h-l())],
      y-label: y-axis-label([Mean arsenic clearance lead (#hour())]),
      annotations: annotations,
      theme: results-chart-theme
        + padding
        + (
          axis-padding-left: padding.axis-padding-left - 10pt,
          marker-stroke: none,
        ),
    ))
  ]
}

#let control-comparison-figure(cases, points) = {
  assert(points.len() == 10)
  let case-bounds = (
    low: 0,
    high: ribbon-bounds((0,) + cases.map(row => row.baseline) + cases.map(row => row.pretrained)).high,
  )
  let x-bounds = ribbon-bounds(points.map(p => p.burden))
  let y-bounds = ribbon-bounds(points.map(p => p.clearance))
  set par(justify: false)
  // Preserve the existing panel-b reference without a second figure number or
  // caption. The native figure element acts as a labelled subfigure only.
  show figure.where(kind: "control-comparison-panel"): it => it.body
  block(width: 100%, context layout(size => {
    let case-padding = chart-side-padding(
      [Selected pretrained policy arsenic burden (#mg-h-l())],
      case-bounds,
      last-x: axis-tick-text(case-bounds.high),
    )
    let joint-padding = chart-side-padding([Mean arsenic clearance lead (#hour())], y-bounds, last-x: axis-tick-text(
      x-bounds.high,
    ))
    let padding = (
      axis-padding-left: calc.max(case-padding.axis-padding-left, joint-padding.axis-padding-left),
      axis-padding-right: calc.max(case-padding.axis-padding-right, joint-padding.axis-padding-right),
    )
    let plot-size = (size.width - panel-gutter) / 2 - padding.axis-padding-left - padding.axis-padding-right
    stack(
      dir: ttb,
      spacing: legend-spacing,
      grid(
        columns: (1fr, 1fr),
        column-gutter: panel-gutter,
        titled-panel([(a) Test-case comparison], case-comparison-panel(cases, case-bounds, padding, plot-size)),
        [#figure(
          titled-panel([(b) Seed summaries], joint-control-panel(points, x-bounds, y-bounds, padding, plot-size)),
          kind: "control-comparison-panel",
          outlined: false,
          supplement: [Figure],
          numbering: n => context [#counter(figure.where(kind: image)).at(<fig:test-case-burden>).first()#text("(b)")],
        ) <fig:joint-control-metrics>],
      ),
      legend-row(
        legend-item(circle(radius: 1.9pt, fill: accent.transparentize(20%), stroke: none), [Test case (a)]),
        legend-item(legend-line(muted, stroke: boundary-rule), [Equal burden (a)]),
        ..("Selected baseline policy", "Selected pretrained policy")
          .enumerate()
          .map(((i, name)) => legend-item(circle(radius: 2.5pt, fill: policy-colors.at(i), stroke: none), [#name (b)])),
        legend-item(legend-line(muted, stroke: 0.6pt + muted.transparentize(40%)), [Matched seed (b)]),
      ),
    )
  }))
}

#let case-burden-figure(burdens) = {
  assert(burdens.map(row => int(row.seed)) == seeds)
  assert(burdens.all(row => (
    row.baseline >= 0 and row.pretrained >= 0 and row.baseline <= row.reference and row.pretrained <= row.reference
  )))
  let reference-color = reference-bar-fill
  let series = (
    (name: "Selected baseline policy", values: burdens.map(row => row.baseline)),
    (name: "Selected pretrained policy", values: burdens.map(row => row.pretrained)),
    (name: "Zero-chlorine reference", values: burdens.map(row => row.reference)),
  )
  set par(justify: false)
  block(width: 100%, stack(
    dir: ttb,
    spacing: legend-spacing,
    context layout(size => {
      let y-title = [Arsenic burden (#mg-h-l())]
      let bounds = ribbon-bounds((0,) + burdens.map(row => row.reference))
      let theme = results-chart-theme + chart-side-padding(y-title, bounds)
      box(width: size.width, height: 145pt + axis-footer-height)[
        #place(top + left, grouped-bar-chart(
          (labels: burdens.map(row => identifier(row.seed)), series: series),
          width: size.width,
          height: 145pt,
          show-legend: false,
          x-label: [Training seed],
          y-label: y-axis-label(y-title),
          theme: theme
            + (
              palette: (policy-colors.at(0), policy-colors.at(1), reference-color),
            ),
        ))
      ]
    }),
    legend-row(
      legend-item(legend-patch(policy-colors.at(0)), label([Selected baseline policy])),
      legend-item(legend-patch(policy-colors.at(1)), label([Selected pretrained policy])),
      legend-item(legend-patch(reference-color), label([Zero-chlorine reference])),
    ),
  ))
}

#let forecast-models(data) = {
  assert(data.format_version == 1)
  if data.data_kind == "pending" {
    assert(data.rows.len() == 0)
    return ()
  }
  assert(data.rows.len() == 425, message: "Forecast figure requires all 25 models and horizons")
  seeds
    .map(seed => range(5).map(fold => {
      let rows = data.rows.filter(r => r.training_seed == seed and r.fold_id == fold)
      assert(rows.map(r => r.horizon_step).sorted() == range(17))
      let full = rows.find(r => r.horizon_step == 0)
      (seed: seed, fold: fold, mae: full.mae, rmse: full.rmse)
    }))
    .flatten()
}

// Displace only colliding markers, then center each connected cluster on its
// seed guide. Work in physical units so separation matches marker diameter.
#let forecast-swarm(models, metric, bounds, height) = {
  let placed = ()
  let separation = 4.8
  for model in models.sorted(key: m => m.at(metric)) {
    let py = (bounds.high - model.at(metric)) / (bounds.high - bounds.low) * height / 1pt
    let candidates = (0,)
    for point in placed {
      let dy = calc.abs(py - point.y)
      if dy < separation {
        let dx = calc.sqrt(separation * separation - dy * dy)
        candidates.push(point.x - dx)
        candidates.push(point.x + dx)
      }
    }
    let px = candidates
      .sorted(key: x => calc.abs(x))
      .find(candidate => placed.all(point => (
        calc.pow(candidate - point.x, 2) + calc.pow(py - point.y, 2) >= separation * separation - 0.000001
      )))
    placed.push((x: px, y: py, model: model))
  }
  let groups = ()
  for point in placed {
    if groups.len() == 0 or calc.abs(point.y - groups.last().last().y) >= separation {
      groups.push((point,))
    } else {
      let group = groups.pop()
      groups.push(group + (point,))
    }
  }
  groups
    .map(group => {
      let center = mean(group.map(point => point.x))
      group.map(point => (dx: (point.x - center) * 1pt, model: point.model))
    })
    .flatten()
}

#let forecast-figure(data: json("../../results/forecast-plot-data.json")) = {
  let models = forecast-models(data)
  let titles = ([(a) MAE], [(b) RMSE])
  assert(models.len() == 25)
  set par(justify: false)
  block(width: 100%, stack(
    dir: ttb,
    spacing: legend-spacing,
    grid(
      columns: (1fr, 1fr), column-gutter: panel-gutter,
      ..("mae", "rmse")
        .enumerate()
        .map(((index, metric)) => {
          // Padding categories leave room for the outer folds. The real model
          // series set the native axis range, with no connecting lines.
          let series = range(5).map(fold => {
            let values = seeds.map(seed => models.find(m => m.seed == seed and m.fold == fold).at(metric))
            (name: [Fold #(fold + 1)], values: (values.first(),) + values + (values.last(),))
          })
          titled-panel(titles.at(index), context layout(size => {
            let bounds = ribbon-bounds(models.map(model => model.at(metric)))
            let theme = results-chart-theme + chart-side-padding([#mg-l()], bounds)
            let plot-width = size.width - theme.axis-padding-left - theme.axis-padding-right
            let annotations = ()
            for (i, seed) in seeds.enumerate() {
              let selected = models.filter(model => model.seed == seed)
              annotations.push((type: "v-line", value: i + 1, color: grid-color, stroke: grid-line-size))
              annotations.push((
                type: "errorbar",
                orientation: "h",
                y: mean(selected.map(model => model.at(metric))),
                low: i + 1 - 5pt / plot-width * 6,
                high: i + 1 + 5pt / plot-width * 6,
                cap-width: 0pt,
                stroke: line-size,
                color: black,
              ))
              for point in forecast-swarm(selected, metric, bounds, 120pt) {
                annotations.push((
                  type: "point",
                  x: i + 1 + point.dx / plot-width * 6,
                  y: point.model.at(metric),
                  radius: 2pt,
                  fill: folds.at(point.model.fold),
                  stroke: none,
                ))
              }
            }
            // Exclude Primaviz's unused 50pt legend allowance from the panel.
            box(width: size.width, height: 120pt + axis-footer-height)[
              #place(top + left, multi-line-chart(
                (labels: ([],) + seeds.map(seed => identifier(str(seed))) + ([],), series: series),
                width: size.width,
                height: 120pt,
                line-width: 0pt,
                show-points: false,
                show-legend: false,
                x-label: [Training seed],
                y-label: y-axis-label([#mg-l()]),
                annotations: annotations,
                theme: theme + (palette: folds),
              ))
            ]
          }))
        }),
    ),
    legend-row(
      ..range(5).map(f => legend-item(
        circle(radius: 2pt, fill: folds.at(f), stroke: none),
        [Fold #(f + 1)],
      )),
      legend-item(seed-mean-marker, [Seed mean]),
    ),
  ))
}

#let learning-series(data) = {
  if data.data_kind == "pending" {
    assert(data.rows.len() == 0)
    return ()
  }
  assert(data.rows.len() == 210, message: "Learning curves require both policies and all five seeds")
  ("baseline", "pretrained").map(approach => {
    let runs = seeds.map(seed => {
      let rows = data
        .rows
        .filter(r => r.approach == approach and r.training_seed == seed)
        .sorted(key: r => r.nominal_interactions)
      assert(rows.map(r => r.nominal_interactions) == range(0, 1000001, step: 50000))
      assert(rows.all(r => r.actual_interactions >= r.nominal_interactions))
      rows
    })
    let positions = runs.first().map(r => r.actual_interactions)
    assert(
      runs.all(run => run.map(r => r.actual_interactions) == positions),
      message: "Do not average learning curves with different actual interaction counts",
    )
    (
      runs: runs,
      summaries: positions
        .enumerate()
        .map(((i, actual)) => {
          let returns = runs.map(run => run.at(i).evaluation_return)
          (
            nominal: runs.first().at(i).nominal_interactions,
            actual: actual,
            mean: mean(returns),
            sd: sample-sd(returns),
          )
        }),
    )
  })
}

#let forecast-horizon-figure(summaries) = {
  let titles = ([(a) MAE], [(b) RMSE])
  assert(summaries.len() == 16)
  set par(justify: false)
  block(width: 100%, grid(
    columns: (1fr, 1fr), column-gutter: panel-gutter,
    ..("mae", "rmse")
      .enumerate()
      .map(((index, metric)) => titled-panel(titles.at(index), context layout(size => {
        let color = policy-colors.at(index)
        let means = summaries.map(s => s.at(metric + "-mean"))
        let sds = summaries.map(s => s.at(metric + "-sd"))
        let extents = (-1, 1).map(sign => (
          name: "SD extent",
          values: means.zip(sds).map(((mu, sd)) => mu + sign * sd),
        ))
        let bounds = ribbon-bounds(extents.map(s => s.values).flatten())
        let theme = (
          results-chart-theme + chart-side-padding([#mg-l()], bounds, last-x: num(str(summaries.last().step / 2)))
        )
        let points = summaries
          .enumerate()
          .map(((i, s)) => (
            x: i / (summaries.len() - 1),
            mean: means.at(i),
            sd: sds.at(i),
          ))
        box(width: size.width, height: 120pt + axis-footer-height)[
          #place(top + left, dx: theme.axis-padding-left, sd-ribbon(
            points,
            size.width - theme.axis-padding-left - theme.axis-padding-right,
            120pt,
            bounds,
            uncertainty-fill(color),
          ))
          #place(top + left, multi-line-chart(
            (
              labels: summaries.map(s => if s.step in (1, 4, 8, 12, 16) { num(str(s.step / 2)) } else { [] }),
              series: ((name: metric, values: means),) + extents,
            ),
            width: size.width,
            height: 120pt,
            show-points: false,
            show-legend: false,
            line-width: line-size,
            x-label: [Forecast horizon (#hour())],
            y-label: y-axis-label([#mg-l()]),
            theme: theme + (palette: (color, white.transparentize(100%), white.transparentize(100%))),
          ))
        ]
      }))),
  ))
}

#let gcd(a, b) = {
  while b != 0 {
    let remainder = calc.rem(a, b)
    a = b
    b = remainder
  }
  a
}

// Primaviz's line charts use equally spaced positions. Subdivide the intervals
// on their common integer grid and interpolate each existing straight segment.
// Every measured checkpoint retains its exact x/y position. The ribbons join
// mean +/- SD at those same checkpoints with straight boundaries.
#let interpolate-grid(rows, positions, x-value, y-value) = positions.map(position => {
  let exact = rows.find(row => x-value(row) == position)
  if exact != none { return y-value(exact) }
  let upper-index = rows.position(row => x-value(row) > position)
  let lower = rows.at(upper-index - 1)
  let upper = rows.at(upper-index)
  let fraction = (position - x-value(lower)) / (x-value(upper) - x-value(lower))
  y-value(lower) + fraction * (y-value(upper) - y-value(lower))
})

#let learning-policy-panel(policies, x-low, height, mean-scale: false) = context layout(size => {
  let visible = policies.map(p => (
    runs: p.runs.map(run => run.filter(r => r.nominal_interactions >= x-low)),
    summaries: p.summaries.filter(s => s.nominal >= x-low),
  ))
  let measured = visible.first().summaries.map(s => s.nominal)
  assert(visible.all(p => p.summaries.map(s => s.nominal) == measured))
  let intervals = range(1, measured.len()).map(i => measured.at(i) - measured.at(i - 1))
  let unit = intervals.fold(intervals.first(), gcd)
  let positions = range(measured.first(), measured.last() + 1, step: unit)
  let labels = positions.map(n => if calc.rem(n, 100000) == 0 {
    if n == 0 { "0" } else { str(int(n / 1000)) + "k" }
  } else { [] })
  let means = visible
    .enumerate()
    .map(((i, p)) => (
      name: ("Baseline approach", "Pretrained approach").at(i),
      values: interpolate-grid(p.summaries, positions, s => s.nominal, s => s.mean),
    ))
  let runs = visible
    .enumerate()
    .map(((i, p)) => p
      .runs
      .enumerate()
      .map(((j, run)) => (
        name: [#("Baseline approach", "Pretrained approach").at(i), seed #seeds.at(j)],
        values: interpolate-grid(run, positions, r => r.nominal_interactions, r => r.evaluation_return),
      )))
    .flatten()
  let sd-extents = visible
    .map(p => (-1, 1).map(sign => (
      name: "SD extent",
      values: interpolate-grid(p.summaries, positions, s => s.nominal, s => s.mean + sign * s.sd),
    )))
    .flatten()
  let range-values = (runs + sd-extents).map(s => s.values).flatten()
  let display-values = if mean-scale { (-3500000, 0) } else { range-values }
  let extents = (
    (name: "Lower display bound", values: (calc.min(..display-values),) * positions.len()),
    (name: "Upper display bound", values: (calc.max(..display-values),) * positions.len()),
  )
  let transparent = white.transparentize(100%)
  let extent-colors = (transparent,) * extents.len()
  let seed-colors = policy-colors.map(color => (color.transparentize(65%),) * seeds.len()).flatten()
  let bounds = if mean-scale { (low: -3500000, high: 0) } else { ribbon-bounds(display-values) }
  // Use the widest title/tick combination from the full training range in
  // both panels so their axes stay aligned despite different vertical scales.
  let full-bounds = ribbon-bounds(
    policies
      .map(p => (
        p.runs.map(run => run.map(r => r.evaluation_return)).flatten()
          + p.summaries.map(s => (s.mean - s.sd, s.mean + s.sd)).flatten()
      ))
      .flatten(),
  )
  let theme = results-chart-theme + chart-side-padding([Validation return], full-bounds, last-x: labels.last())
  let ribbons = visible.map(p => p.summaries.map(s => (
    x: (s.nominal - positions.first()) / (positions.last() - positions.first()),
    mean: s.mean,
    sd: s.sd,
  )))
  box(width: size.width, height: height + axis-footer-height)[
    #let y-title = vertical-axis-title([Validation return])
    #box(width: size.width, height: height, align(left + horizon, y-title))
    #for (i, points) in ribbons.enumerate() {
      let width = size.width - theme.axis-padding-left - theme.axis-padding-right
      let ribbon = sd-ribbon(points, width, height, bounds, uncertainty-fill(policy-colors.at(i)))
      place(top + left, dx: theme.axis-padding-left, if mean-scale {
        box(width: width, height: height, clip: true, ribbon)
      } else { ribbon })
    }
    #if mean-scale {
      // Project seeds directly onto the displayed scale. Clip the paths at
      // the plot boundary while preserving every measured return.
      let width = size.width - theme.axis-padding-left - theme.axis-padding-right
      let span = bounds.high - bounds.low
      place(top + left, dx: theme.axis-padding-left, box(width: width, height: height, clip: true)[
        #for (i, run) in runs.enumerate() {
          let points = run
            .values
            .enumerate()
            .map(((j, value)) => (
              j / (positions.len() - 1) * width,
              (bounds.high - value) / span * height,
            ))
          plot-path(points, seed-line-size + seed-colors.at(i))
        }
      ])
    } else {
      place(top + left, multi-line-chart(
        (labels: labels, series: runs + extents),
        width: size.width,
        height: height,
        show-points: false,
        show-legend: false,
        line-width: seed-line-size,
        x-label: [Nominal online training interactions],
        y-label: none,
        theme: theme + (palette: seed-colors + extent-colors),
      ))
    }
    #place(top + left, multi-line-chart(
      (labels: if mean-scale { labels } else { ([],) * labels.len() }, series: means + extents),
      width: size.width,
      height: height,
      show-points: false,
      show-legend: false,
      line-width: emphasized-line-size,
      x-label: if mean-scale { [Nominal online training interactions] } else { none },
      y-label: none,
      // Eight ticks keep the requested -3.5M to 0 bounds at 0.5M intervals.
      theme: theme
        + (palette: policy-colors + extent-colors)
        + if mean-scale { (tick-count: 8) } else {
          (
            text-color: transparent,
            axis-stroke: 0.45pt + transparent,
          )
        },
    ))
  ]
})

#let learning-figure(data: json("../../results/learning-plot-data.json")) = {
  let policies = learning-series(data)
  assert(policies.len() == 2)
  set par(justify: false)
  block(width: 100%, stack(
    dir: ttb,
    spacing: panel-spacing,
    titled-panel([(a) Full training range], learning-policy-panel(policies, 0, 125pt, mean-scale: true)),
    titled-panel([(b) Late training], learning-policy-panel(policies, 500000, 150pt)),
    legend-row(
      ..("Baseline approach", "Pretrained approach")
        .enumerate()
        .map(((i, name)) => legend-item(
          legend-patch(
            uncertainty-fill(policy-colors.at(i)),
            line-color: policy-colors.at(i),
            line-stroke: emphasized-line-size + policy-colors.at(i),
          ),
          [#name],
        )),
    ),
  ))
}
