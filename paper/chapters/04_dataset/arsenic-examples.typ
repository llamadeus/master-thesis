#import "@preview/acrostiche:0.7.0": acr
#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": multi-line-chart

#let data = json("../../dataset/arsenic-examples.json")
#assert(data.unit == "mg/L" and data.policy == "zero-injection")
#let colors = (series-colors.at(1), series-colors.at(0)) + series-colors.slice(2)
#let label = plot-label
#let total-injection(example) = {
  let events = example.selection.event_indices.map(i => example.run.events.at(i))
  let window = example.selection.window_seconds
  let times = (
    (window.first(), window.last())
      + events
        .map(event => (event.start_time_seconds, event.end_time_seconds))
        .flatten()
        .filter(time => window.first() <= time and time <= window.last())
  )
  times
    .sorted()
    .dedup()
    .map(time => (
      time: time,
      rate: events
        .filter(event => event.start_time_seconds <= time and time < event.end_time_seconds)
        .map(event => event.strength)
        .fold(0, (sum, rate) => sum + rate),
    ))
}
#let rate-maximum = calc.max(
  ..data.panels.map(example => total-injection(example).map(point => point.rate)).flatten(),
)
// One common rate scale makes injection magnitudes comparable across panels.
#let rate-upper = calc.ceil(rate-maximum / 10000) * 10000
#let rate-ticks = (0, rate-upper / 2, rate-upper)
#let rate-label(value) = label(num(str(int(value))))

#let concentration-scale(example) = {
  let indices = range(example.time_seconds.len()).filter(i => (
    example.selection.window_seconds.first() <= example.time_seconds.at(i)
      and example.time_seconds.at(i) <= example.selection.window_seconds.last()
  ))
  let peak = calc.max(
    ..example
      .selection
      .nodes
      .map(id => {
        let values = example.nodes.find(node => node.id == id).arsenic
        indices.map(i => values.at(i))
      })
      .flatten(),
  )
  let magnitude = calc.pow(10, calc.floor(calc.log(peak / 3, base: 10)))
  let ratio = peak / 3 / magnitude
  let factor = if ratio >= calc.sqrt(50) { 10 } else if ratio >= calc.sqrt(10) { 5 } else if ratio >= calc.sqrt(2) {
    2
  } else { 1 }
  let step = factor * magnitude
  let upper = calc.ceil(peak / step) * step
  let tick-count = int(upper / step) + 1
  let digits = int(calc.max(0, -calc.floor(calc.log(step, base: 10))))
  (step: step, upper: upper, tick-count: tick-count, digits: digits)
}
#let concentration-scales = data.panels.map(concentration-scale)

#let panel(example, title, scale) = context layout(size => {
  let selection = example.selection
  let events = selection.event_indices.map(i => example.run.events.at(i))
  let injection = total-injection(example)
  let indices = range(example.time_seconds.len()).filter(i => {
    let time = example.time_seconds.at(i)
    selection.window_seconds.first() <= time and time <= selection.window_seconds.last()
  })
  let times = indices.map(i => (example.time_seconds.at(i) - selection.origin_seconds) / 3600)
  let series = selection.nodes.map(id => {
    let node = example.nodes.find(node => node.id == id)
    let source = events.any(event => event.node_id == id)
    (name: [Node #id], source: source, values: indices.map(i => node.arsenic.at(i)))
  })
  let upper = scale.upper
  let tick-count = scale.tick-count
  let digits = scale.digits
  let height = 78pt
  let width = size.width
  // Measure all panels together so both axis margins and plot widths match.
  let y-label-width = calc.max(
    ..concentration-scales
      .map(scale => range(scale.tick-count).map(i => {
        measure(label(num(str(calc.round(i * scale.step, digits: scale.digits))))).width
      }))
      .flatten(),
  )
  let y-title = rotate(-90deg, reflow: true, label([#mg-l()]))
  let left-pad = measure(y-title).width + axis-title-gap + y-label-width + y-tick-gap
  let rate-title = vertical-axis-title([#mg-min()])
  let rate-label-width = calc.max(..rate-ticks.map(value => measure(rate-label(value)).width))
  let right-pad = y-tick-gap + rate-label-width + axis-title-gap + measure(rate-title).width
  let plot-width = width - left-pad - right-pad
  let index-at(hour) = (hour - times.first()) / (data.timestep_seconds / 3600)
  let annotations = ()
  for i in range(injection.len() - 1).filter(i => injection.at(i).rate > 0) {
    annotations.push((
      type: "rect",
      x1: index-at((injection.at(i).time - selection.origin_seconds) / 3600),
      x2: index-at((injection.at(i + 1).time - selection.origin_seconds) / 3600),
      y1: 0,
      y2: upper,
      fill: active-injection-fill,
    ))
  }
  let time-step = calc.max(6, calc.ceil(times.last() / 3 / 6) * 6)
  let ticks = range(0, int(times.last()) + 1, step: int(time-step))
  for hour in ticks {
    annotations.push((
      type: "content",
      x: index-at(hour),
      y: 0,
      anchor: "top",
      body: line(end: (0pt, tick-length), stroke: axis-rule),
    ))
    annotations.push((
      type: "content",
      x: index-at(hour),
      y: 0,
      anchor: "top",
      dy: x-tick-offset,
      body: label(str(hour)),
    ))
  }
  let x-label-height = calc.max(..ticks.map(hour => measure(label(str(hour))).height))
  annotations.push((
    type: "content",
    x: (indices.len() - 1) / 2,
    y: 0,
    anchor: "top",
    dy: x-tick-offset + x-label-height + axis-title-gap,
    body: label([time since event start (#hour())]),
  ))
  annotations.push((
    type: "content",
    x: 0,
    y: upper / 2,
    anchor: "right",
    dx: -(y-label-width + y-tick-gap + axis-title-gap),
    body: y-title,
  ))
  stack(
    dir: ttb,
    spacing: title-spacing,
    align(center, panel-title(title)),
    box(width: width, height: height + axis-footer-height)[
      #for hour in ticks {
        place(top + left, dx: left-pad + index-at(hour) / (indices.len() - 1) * plot-width, line(
          end: (0pt, height),
          stroke: grid-rule,
        ))
      }
      #for i in range(tick-count) {
        place(top + left, dx: left-pad, dy: i * height / (tick-count - 1), line(
          length: plot-width,
          stroke: grid-rule,
        ))
      }
      #place(top + left, multi-line-chart(
        (
          labels: indices.map(_ => ""),
          series: (
            (name: "min", values: indices.map(_ => 0)),
            (name: "max", values: indices.map(_ => upper)),
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
          palette: (white.transparentize(100%), white.transparentize(100%)),
          axis-padding-left: left-pad,
          axis-padding-right: right-pad,
          tick-count: tick-count,
          tick-digits: digits,
        ),
      ))
      // Primaviz provides axes and annotations but has one shared line width.
      // Draw continuous, unsmoothed concentration paths with role-specific
      // widths, preserving all original 30-minute samples.
      #for (i, s) in series.enumerate() {
        let points = s
          .values
          .enumerate()
          .map(((j, value)) => (
            left-pad + j / (indices.len() - 1) * plot-width,
            height * (1 - value / upper),
          ))
        place(top + left, curve(
          curve.move(points.first()),
          ..points.slice(1).map(point => curve.line(point)),
          fill: none,
          stroke: concentration-rule(colors.at(i), source: s.source),
        ))
      }
      // Sum active source rates at exact schedule boundaries and draw one
      // rectangular step path on the shared time map.
      #{
        let x-at(time) = (
          left-pad
            + (time - selection.window_seconds.first())
              / (selection.window_seconds.last() - selection.window_seconds.first())
              * plot-width
        )
        let y-at(point) = height * (1 - point.rate / rate-upper)
        let steps = (curve.move((x-at(injection.first().time), y-at(injection.first()))),)
        for i in range(1, injection.len()) {
          let point = injection.at(i)
          steps += (
            curve.line((x-at(point.time), y-at(injection.at(i - 1)))),
            curve.line((x-at(point.time), y-at(point))),
          )
        }
        place(top + left, curve(
          ..steps,
          fill: none,
          stroke: injection-rule(muted),
        ))
      }
      #place(top + left, dx: left-pad + plot-width, line(end: (0pt, height), stroke: axis-rule))
      #for value in rate-ticks {
        let y = height * (1 - value / rate-upper)
        place(top + left, dx: left-pad + plot-width + y-tick-gap, dy: y, box(height: 0pt, align(
          horizon,
          rate-label(value),
        )))
      }
      #place(
        top + left,
        dx: left-pad + plot-width + y-tick-gap + rate-label-width + axis-title-gap,
        dy: (height - measure(rate-title).height) / 2,
        rate-title,
      )
    ],
    legend-spacing,
    legend-column(..series
      .enumerate()
      .map(((i, s)) => legend-item(
        legend-line(colors.at(i), stroke: concentration-rule(colors.at(i), source: s.source)),
        s.name,
      ))),
  )
})

#let arsenic-figure = block(width: 100%)[
  #set par(justify: false)
  #stack(
    dir: ttb,
    spacing: legend-spacing,
    grid(
      columns: (1fr, 1fr, 1fr),
      column-gutter: panel-gutter,
      panel(data.panels.at(0), [(a) Mild event], concentration-scales.at(0)),
      panel(data.panels.at(1), [(b) Strong event], concentration-scales.at(1)),
      panel(data.panels.at(2), [(c) Overlapping events], concentration-scales.at(2)),
    ),
    legend-row(
      legend-item(legend-line(muted, stroke: concentration-rule(muted, source: true)), [Source concentration]),
      legend-item(legend-line(muted, stroke: concentration-rule(muted)), [Downstream concentration]),
      legend-item(legend-line(muted, stroke: injection-rule(muted)), [Total injection rate (right axis)]),
    ),
  )
]

#let case-context(
  example,
) = [#raw(example.run.scenario_id) with #raw("arsenic-" + str(example.run.schedule_index)), at #hour(str(example.selection.origin_seconds / 3600))]
#let arsenic-caption = [Illustrative #acr("As(III)") concentration histories from the fixed control-validation cases with zero chlorine injection and a chlorine-free initial state. (a, b) A relatively mild and a relatively strong event show delayed downstream transport. Source-node peaks lie in the lower and upper quartiles, respectively, of the isolated events eligible for the transport illustration. (c) Two source events overlap in time. The examples use #case-context(data.panels.at(0)) (a), #case-context(data.panels.at(1)) (b), and #case-context(data.panels.at(2)) (c) into the 25-day simulation. Schedule indices start at zero, and the generation seeds are specified in @section:appendix:dataset-artifacts. Time zero denotes the start of the first arsenic injection shown in each panel. Solid lines show nodal concentrations on the left axes, with thicker lines at source nodes and thinner lines at downstream nodes. Gray dashed steps show the combined mass injection rate of all active source events on the right axes. Orange shading marks the injection intervals. Flow-dependent dilution, downstream mixing, and travel times shape the concentration histories even under constant mass injection. Concentration ranges differ between panels, while all injection-rate axes use the same range.]
