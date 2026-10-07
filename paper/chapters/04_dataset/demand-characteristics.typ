#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": multi-line-chart

#let data = json("../../dataset/demand-patterns.json")
#let per-day = int(86400 / data.timestep_seconds)
#let days = data.calendar_days
#let scenario-days = int(data.simulation_duration_seconds / 86400)
#let count = data.nodes.len()
#let origin = data.first_sample_calendar_day
#assert(data.samples_per_node == days * per-day)
#assert(data.quantity == "actual_nodal_demand" and data.unit == "m3/h")

// Restore calendar order from the rotation stored in the simulation inputs.
#let cut = calc.rem((days - origin) * per-day, days * per-day)
#let annual = data.nodes.map(node => {
  assert(node.demand.len() == days * per-day)
  node.demand.slice(cut) + node.demand.slice(0, cut)
})
#let mean(values) = values.sum() / values.len()
#let summarize(values) = {
  let mu = mean(values)
  // Population SD describes all monitored demand junctions, with equal weight.
  let sd = calc.sqrt(values.map(value => calc.pow(value - mu, 2)).sum() / values.len())
  (mean: mu, sd: sd)
}
#let at-step(step) = summarize(annual.map(values => values.at(calc.rem(step, days * per-day))))

// The exporter's CLI selects a complete week, using day zero as Monday.
#let week-start = data.figure_selection.week_start_day
#let day-start = data.figure_selection.day_start_day
#assert(calc.rem(week-start, 7) == 0 and week-start >= 0 and week-start + 7 <= scenario-days)
#assert(day-start == week-start + 3)
#let day-stats = range(per-day + 1).map(i => at-step(day-start * per-day + i))
#let week-stats = range(7 * per-day + 1).map(i => at-step(week-start * per-day + i))
// Use half-hourly values throughout the scenario, including its endpoint.
#let scenario-stats = range(scenario-days * per-day + 1).map(at-step)
#let all-stats = day-stats + week-stats + scenario-stats
// Round the common range to readable ticks without excess empty space.
#let band-max = calc.max(..all-stats.map(s => s.mean + s.sd))
#let tick-magnitude = calc.pow(10, calc.floor(calc.log(band-max / 4, base: 10)))
#let y-step = (1, 2, 2.5, 5, 10).find(step => step * tick-magnitude >= band-max / 4) * tick-magnitude
#let upper = calc.ceil(band-max / y-step) * y-step
#let y-tick-count = int(upper / y-step) + 1
#assert(all-stats.all(s => s.mean - s.sd >= 0), message: "Demand band extends below the common zero axis")

#let day-color = orange
#let week-color = purple
#let band-fill = uncertainty-fill(accent)
#let day-fill = day-color.transparentize(90%)
#let week-fill = week-color.transparentize(90%)

// Primaviz 0.10.0 has no ribbon series. A content annotation supplies the
// polygon between mean +/- SD, using the same explicit plot range and padding.
#let panel(stats, title, axis-title, ticks, height: 78pt, grid-ticks: none, intervals: ()) = context layout(size => {
  let width = size.width
  let tick-label = plot-label
  let y-label-width = calc.max(..range(y-tick-count).map(i => measure(tick-label(str(int(i * y-step)))).width))
  let x-label-height = calc.max(..ticks.map(tick => measure(tick-label(tick.at(1))).height))
  let y-title = rotate(-90deg, reflow: true, tick-label([#m3-h()]))
  // Include all axis labels in the text-width container. Use the same visible
  // gap from tick labels to axis titles on the horizontal and vertical axes.
  let left-pad = measure(y-title).width + axis-title-gap + y-label-width + y-tick-gap
  // One common margin keeps the plotting areas aligned across all panels.
  let right-pad = calc.max(..(24, 7, scenario-days).map(value => measure(tick-label(str(value))).width)) / 2
  let plot-width = width - left-pad - right-pad
  let n = stats.len()
  let point(i, y) = (i / (n - 1) * plot-width, (1 - y / upper) * height)
  let ribbon = polygon(
    ..stats.enumerate().map(((i, s)) => point(i, s.mean + s.sd)),
    ..stats.enumerate().rev().map(((i, s)) => point(i, s.mean - s.sd)),
    fill: band-fill,
    stroke: none,
  )
  let annotations = (
    (type: "content", x: 0, y: upper, anchor: "top-left", body: box(width: plot-width, height: height, ribbon)),
  )
  for interval in intervals {
    let x1 = calc.max(0, interval.start)
    let x2 = calc.min(n - 1, interval.end)
    if x2 > x1 {
      annotations.push((type: "rect", x1: x1, x2: x2, y1: 0, y2: upper, fill: interval.fill))
      // Boundaries distinguish the selected day within the scenario view.
      annotations.push((type: "v-line", value: x1, color: interval.color, stroke: 0.65pt))
      annotations.push((type: "v-line", value: x2, color: interval.color, stroke: 0.65pt))
    }
  }
  for (position, label) in ticks {
    annotations.push((
      type: "content",
      x: position,
      y: 0,
      anchor: "top",
      body: line(end: (0pt, tick-length), stroke: axis-rule),
    ))
    annotations.push((type: "content", x: position, y: 0, anchor: "top", dy: x-tick-offset, body: tick-label(label)))
  }
  annotations.push((
    type: "content",
    x: 0,
    y: upper / 2,
    anchor: "right",
    dx: -(y-label-width + y-tick-gap + axis-title-gap),
    body: y-title,
  ))
  annotations.push((
    type: "content",
    x: (n - 1) / 2,
    y: 0,
    anchor: "top",
    dy: x-tick-offset + x-label-height + axis-title-gap,
    body: tick-label(axis-title),
  ))
  stack(
    dir: ttb,
    spacing: title-spacing,
    grid(
      columns: (left-pad, 1fr, right-pad),
      [], align(center, panel-title(title)), [],
    ),
    // Reserve the actual plot and tick-label height, without the package's
    // unused legend allowance. All content is positioned inside this box.
    box(width: width, height: height + axis-footer-height)[
      // Draw the time grid explicitly. Primaviz's automatic vertical grid
      // follows its y-tick count rather than the time-axis tick positions.
      #let time-grid = if grid-ticks == none { ticks.map(tick => tick.at(0)) } else { grid-ticks }
      #for position in time-grid {
        place(top + left, dx: left-pad + position / (n - 1) * plot-width, line(
          end: (0pt, height),
          stroke: grid-rule,
        ))
      }
      #for i in range(y-tick-count) {
        place(top + left, dx: left-pad, dy: i * height / (y-tick-count - 1), line(
          length: plot-width,
          stroke: grid-rule,
        ))
      }
      #place(top + left, multi-line-chart(
        (
          labels: stats.map(_ => ""),
          series: (
            (name: "Mean", values: stats.map(s => s.mean)),
            // Transparent reference series enforce one common scale for all views.
            (name: "Axis minimum", values: stats.map(_ => 0)),
            (name: "Axis maximum", values: stats.map(_ => upper)),
          ),
        ),
        width: width,
        height: height,
        show-points: false,
        show-legend: false,
        line-width: line-size,
        line-interpolation: "linear",
        annotations: annotations,
        theme: (
          ..chart-theme,
          palette: (accent, white.transparentize(100%), white.transparentize(100%)),
          axis-padding-left: left-pad,
          axis-padding-right: right-pad,
          tick-count: y-tick-count,
          tick-digits: 0,
          number-format: "plain",
        ),
      ))
    ],
  )
})

#let demand-figure = {
  set par(justify: false)
  set text(size: label-size, fill: ink)
  block(width: 100%, stack(
    dir: ttb,
    spacing: panel-spacing,
    panel(
      day-stats,
      [(a) Day #day-start],
      [hour of day],
      range(25).map(hour => (hour * per-day / 24, str(hour))),
      grid-ticks: range(0, 25, step: 3).map(hour => hour * per-day / 24),
    ),
    panel(
      week-stats,
      [(b) Week #data.figure_selection.week_number],
      [day of week],
      range(8).map(day => (day * per-day, str(day))),
      intervals: ((start: 3 * per-day, end: 4 * per-day, fill: day-fill, color: day-color),),
    ),
    panel(
      scenario-stats,
      [(c) Entire scenario (#scenario-days days)],
      [simulation day],
      range(scenario-days + 1).map(day => (day * per-day, str(day))),
      height: 104pt,
      intervals: (
        (start: week-start * per-day, end: (week-start + 7) * per-day, fill: week-fill, color: week-color),
        (start: day-start * per-day, end: (day-start + 1) * per-day, fill: day-fill, color: day-color),
      ),
    ),
    legend-spacing,
    legend-row(
      legend-item(legend-patch(band-fill, line-color: accent), [Mean #sym.plus.minus 1 standard deviation]),
      legend-item(legend-patch(week-fill, stroke: 0.5pt + week-color), [Selected week]),
      legend-item(legend-patch(day-fill, stroke: 0.5pt + day-color), [Selected day]),
    ),
  ))
}

#let demand-caption = [Nodal water demand over (a) day #day-start, (b) week #data.figure_selection.week_number, and (c) the selected #(scenario-days)-day period for #raw(data.scenario_id). Lines show the mean demand across junctions in #m3-h(), with bands indicating one population standard deviation. Demands are computed by the hydraulic model for this base scenario. All panels use 30-minute values. Colored intervals identify the selected week and day.]
