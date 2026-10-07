#import "@preview/acrostiche:0.7.0": acr
#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": multi-line-chart
#import "@preview/cetz:0.5.2"

#let data = json("../../dataset/simulation-run.json")
#let a = data.arrays
#let n = a.time_seconds.len()
#let days = data.duration_seconds / 86400
#let left-pad = 24pt
#let right-pad = 25pt
#let label = plot-label
#let number(value) = num(str(calc.round(value, digits: 2)))
#let injection-rate(value) = {
  let parts = str(calc.round(value, digits: 2)).split(".")
  let fraction = if parts.len() > 1 { parts.last() } else { "" }
  num(parts.first() + "." + fraction + "0" * calc.max(0, 2 - fraction.len()))
}
#let family-labels = (
  stairs: [ramp],
  "random-walk": [random walk],
  "piecewise-constant": [piecewise constant],
  periodic: [periodic],
  constant: [constant],
)

#let title-row(title, unit: none) = grid(
  columns: (left-pad, 1fr, right-pad),
  [],
  align(center, panel-title(title)),
  if unit == none { [] } else { align(left, move(dx: colorbar-gap, dy: 1pt, label(unit))) },
)

// CeTZ coordinates use points; the canvas is sized by its invisible outline.
#let panel-canvas(width, height, body, bottom: false, footer-height: tick-length) = {
  let total-height = height + if bottom { axis-footer-height } else { footer-height }
  let plot-width = width - left-pad - right-pad
  cetz.canvas(length: 1pt, padding: 0pt, {
    import cetz.draw: *
    let w = width / 1pt
    let h = height / 1pt
    let p = plot-width / 1pt
    let left = left-pad / 1pt
    let top = total-height / 1pt
    rect((0, 0), (w, top), fill: none, stroke: none)
    body(p, h, left, top)
    for day in range(int(days) + 1) {
      let x = left + day / days * p
      let tick = if calc.rem(day, 5) == 0 { tick-length } else { tick-length / 2 }
      line((x, top - h), (x, top - h - tick / 1pt), stroke: axis-rule)
      if bottom and calc.rem(day, 5) == 0 {
        content((x, top - h - x-tick-offset / 1pt), label(str(day)), anchor: "north")
      }
    }
    if bottom {
      content(
        (left + p / 2, top - h - (x-tick-offset + measure(label([0])).height + axis-title-gap) / 1pt),
        label([day within episode]),
        anchor: "north",
      )
    }
  })
}

#let guides(p, h, left, top) = {
  import cetz.draw: *
  for event in data.arsenic_events {
    let x1 = left + event.start_time_seconds / data.duration_seconds * p
    let x2 = left + event.end_time_seconds / data.duration_seconds * p
    rect((x1, top - h), (x2, top), fill: active-injection-fill, stroke: none)
  }
  for day in range(0, int(days) + 1, step: 5) {
    let x = left + day / days * p
    line((x, top - h), (x, top), stroke: grid-rule)
  }
}

#let curve(values, title, unit, height: 40pt) = context layout(size => {
  let minimum = calc.min(..values, 0)
  let maximum = calc.max(..values, 1)
  let magnitude = calc.pow(10, calc.floor(calc.log((maximum - minimum) / 4, base: 10)))
  let step = (1, 2, 5, 10).find(v => v * magnitude >= (maximum - minimum) / 4) * magnitude
  let lo = calc.floor(minimum / step) * step
  let hi = calc.ceil(maximum / step) * step
  let ticks = int(calc.round((hi - lo) / step)) + 1
  let segment-steps = int(calc.ceil(n / data.chlorine_families.len()))
  let annotations = ()
  for event in data.arsenic_events {
    annotations.push((
      type: "rect",
      x1: event.start_time_seconds / data.timestep_seconds,
      x2: event.end_time_seconds / data.timestep_seconds,
      y1: lo,
      y2: hi,
      fill: active-injection-fill,
    ))
  }
  for day in range(0, int(days) + 1, step: 5) {
    annotations.push((
      type: "content",
      x: day / days * (n - 1),
      y: hi,
      anchor: "top-left",
      body: line(end: (0pt, height), stroke: grid-rule),
    ))
  }
  for (i, family) in data.chlorine_families.enumerate() {
    let start = i * segment-steps
    let end = calc.min(n - 1, (i + 1) * segment-steps)
    if i > 0 {
      annotations.push((
        type: "content",
        x: start,
        y: hi,
        anchor: "top-left",
        body: line(end: (0pt, height + tick-length), stroke: boundary-rule),
      ))
    }
    annotations.push((
      type: "content",
      x: (start + end) / 2,
      y: lo,
      anchor: "top",
      dy: x-tick-offset,
      body: label(family-labels.at(family)),
    ))
  }
  let y-title = vertical-axis-title(unit)
  annotations.push((
    type: "content",
    x: 0,
    y: (lo + hi) / 2,
    anchor: "right",
    dx: measure(y-title).width - left-pad,
    body: y-title,
  ))
  for day in range(int(days) + 1) {
    let tick = if calc.rem(day, 5) == 0 { tick-length } else { tick-length / 2 }
    annotations.push((
      type: "content",
      x: day / days * (n - 1),
      y: lo,
      anchor: "top",
      body: line(end: (0pt, tick), stroke: axis-rule),
    ))
  }
  stack(
    dir: ttb,
    spacing: title-spacing,
    title-row(title),
    box(width: size.width, height: height + 16pt, multi-line-chart(
      (
        labels: values.map(_ => ""),
        series: (
          (name: "Injection", values: values),
          (name: "Minimum", values: values.map(_ => lo)),
          (name: "Maximum", values: values.map(_ => hi)),
        ),
      ),
      width: (size.width - left-pad - right-pad) * a.time_seconds.last() / data.duration_seconds + left-pad + right-pad,
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
        tick-count: ticks,
        tick-digits: 0,
      ),
    )),
  )
})

// Runs merge adjacent equal display colors only; all samples keep their color.
#let heatmap-runs(rows, hue, logarithmic) = {
  let maximum = calc.max(..rows.flatten())
  let transform(value) = if logarithmic { calc.ln(1 + value / 0.01) } else { value }
  let scale = transform(maximum)
  let colors = range(256).map(i => color.mix((white, (1 - i / 255) * 100%), (hue, i / 255 * 100%)))
  let runs = ()
  for (j, _) in data.node_ids.enumerate() {
    let previous = -1
    let start = 0
    for i in range(n + 1) {
      let k = if i == n { -1 } else {
        int(calc.round(calc.clamp(transform(calc.max(0, rows.at(i).at(j))) / scale, 0, 1) * 255))
      }
      if k != previous {
        if previous > 0 { runs.push((start, i, j, colors.at(previous))) }
        previous = k
        start = i
      }
    }
  }
  (runs: runs, maximum: maximum, transform: transform, scale: scale)
}

#let heatmap(rows, title, hue, logarithmic: false, bottom: false) = context layout(size => {
  let height = 67pt
  let prepared = heatmap-runs(rows, hue, logarithmic)
  let values = if logarithmic {
    (0, 0.01, 0.1, 1).filter(v => (prepared.transform)(v) < prepared.scale * 0.85) + (prepared.maximum,)
  } else {
    (0, prepared.maximum / 2, prepared.maximum)
  }
  stack(
    dir: ttb,
    spacing: title-spacing,
    title-row(title, unit: [#mg-l()]),
    panel-canvas(size.width, height, bottom: bottom, (p, h, left, top) => {
      import cetz.draw: *
      let row-height = h / data.node_ids.len()
      let cell-width = p / n
      for (first, end, j, fill) in prepared.runs {
        rect(
          (left + first * cell-width, top - (j + 1) * row-height),
          (left + end * cell-width + 0.08, top - j * row-height + 0.08),
          fill: fill,
          stroke: none,
        )
      }
      let y-title = vertical-axis-title([junction~ID])
      content((0, top - h / 2), y-title, anchor: "west")
      for id in ("2", "8", "14", "20", "26", "32") {
        let y = top - (data.node_ids.position(v => v == id) + 0.5) * row-height
        content((left - y-tick-gap / 1pt, y), label(id), anchor: "east")
      }
      let bar-x = left + p + colorbar-gap / 1pt
      rect(
        (bar-x, top - h),
        (bar-x + colorbar-width / 1pt, top),
        fill: gradient.linear(hue, white, angle: 90deg),
        stroke: none,
      )
      for value in values {
        let y = top - (1 - (prepared.transform)(value) / prepared.scale) * h
        content((bar-x + (colorbar-width + colorbar-label-gap) / 1pt, y), label(number(value)), anchor: "west")
      }
    }),
  )
})

#let sources = data.node_ids.filter(id => data.arsenic_events.any(e => e.node_id == id))
#let source-panel = context layout(size => {
  let height = calc.max(24pt, sources.len() * 11pt)
  stack(
    dir: ttb,
    spacing: title-spacing,
    title-row([(b) Arsenic injections (#g-min())]),
    panel-canvas(size.width, height, (p, h, left, top) => {
      import cetz.draw: *
      guides(p, h, left, top)
      let y-title = vertical-axis-title([junction~ID])
      content((0, top - h / 2), y-title, anchor: "west")
      for (j, id) in sources.enumerate() {
        let y = top - (j + 0.5) / sources.len() * h
        content((left - y-tick-gap / 1pt, y), label(id), anchor: "east")
      }
      for e in data.arsenic_events {
        let x = left + e.start_time_seconds / data.duration_seconds * p
        let w = (e.end_time_seconds - e.start_time_seconds) / data.duration_seconds * p
        let y = top - (sources.position(id => id == e.node_id) + 0.5) / sources.len() * h
        rect((x, y - 4), (x + w, y + 4), fill: orange, stroke: none)
        let rate = text(size: annotation-size, injection-rate(e.strength / 1000))
        let outside = measure(rate).width / 1pt + 3 > w
        content(
          (if outside { x + w + 2 } else { x + w / 2 }, y),
          text(fill: if outside { orange } else { white }, rate),
          anchor: if outside { "west" } else { "center" },
        )
      }
    }),
  )
})

#let simulation-figure = block(width: 100%)[
  #set par(justify: false, spacing: 0pt)
  #stack(
    dir: ttb,
    spacing: panel-spacing,
    curve(a.injection.map(v => v / 1000), [(a) Chlorine injection], [#g-min()]),
    source-panel,
    heatmap(a.chlorine, [(c) Chlorine concentration], accent),
    heatmap(a.arsenic, [(d) #acr("As(III)") concentration], orange, logarithmic: true, bottom: true),
    legend-spacing,
    legend-row(
      legend-item(legend-patch(active-injection-fill, stroke: guide-line-size + orange), [Active injection]),
      legend-item(legend-line(muted, stroke: boundary-rule), [Profile boundary]),
    ),
  )
]

#let simulation-caption = [Example trajectory from the offline #data.partition split for #raw(data.scenario_id), beginning at calendar day #data.calendar_start_day, with arsenic schedule #raw("arsenic-" + str(data.selection.arsenic_index)) (seed #identifier(data.selection.arsenic_seed)) and chlorine profile #raw("chlorine-" + str(data.selection.chlorine_index)) (seed #identifier(data.selection.chlorine_seed)). All panels share the episode time axis. Dotted lines and labels in (a) mark the chlorine-profile segments. Source bars identify injection nodes and rates. Heatmaps show individual junctions in ascending order and all 30-minute records. The #acr("As(III)") color scale uses $ln(1 + a / hat(a))$, where $a$ is the #acr("As(III)") concentration and $hat(a)$ = #mg-l[0.01].]
