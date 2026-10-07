#import "@preview/acrostiche:0.7.0": acr
#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": bar-chart

#let data = json("../../dataset/arsenic-statistics.json")
#assert(data.schema_version == 2)
#let catalog = data.catalog
#assert(catalog.dataset == "control_full" and catalog.split == "validation")
#let schedules = catalog.runs
#let events = schedules.map(s => s.events).flatten()
#let clusters = schedules.map(r => r.clusters).flatten()
#let label = plot-label
#let pct(n, total) = 100 * n / total
#let histogram(values, edges) = range(edges.len() - 1).map(i => pct(
  values.filter(v => v >= edges.at(i) and (v < edges.at(i + 1) or (i == edges.len() - 2 and v == edges.last()))).len(),
  values.len(),
))

#let durations = events.map(e => (e.end_time_seconds - e.start_time_seconds) / 3600)
// Fixed-schedule durations lie on the inclusive 2--72 h half-hour grid.
#let duration-values = range(4, 145).map(n => n / 2)
#let duration-edges = range(2, 73, step: 5)
#let duration-bin-sizes = range(duration-edges.len() - 1).map(i => duration-values
  .filter(v => (
    v >= duration-edges.at(i)
      and (v < duration-edges.at(i + 1) or (i == duration-edges.len() - 2 and v == duration-edges.last()))
  ))
  .len())
#let duration-density = (
  histogram(durations, duration-edges).enumerate().map(((i, share)) => share / duration-bin-sizes.at(i))
)
#let strengths = events.map(e => calc.log(e.strength, base: 10))
#let strength-low = calc.floor(calc.min(..strengths) * 2) / 2
#let strength-high = calc.ceil(calc.max(..strengths) * 2) / 2
#let strength-edges = range(int((strength-high - strength-low) * 2) + 1).map(i => strength-low + i / 2)
#let reach = clusters.map(c => c.node_peak_mg_l.filter(v => v > catalog.threshold_mg_l).len())
// Match the policy's wrapped calendar_time_seconds before taking weekly phase.
#let calendar-weekday(start-day, time) = calc.rem(
  int(calc.rem(start-day * 86400 + time, 365 * 86400) / 86400),
  7,
)
#let weekday-counts = {
  let observed = (0,) * 7
  let active = (0,) * 7
  for s in schedules {
    // A calendar day contains 48 observation intervals on the 30-minute grid.
    for time in range(0, catalog.duration_seconds, step: 86400) {
      let day = calendar-weekday(s.calendar_start_day, time)
      observed.at(day) += int(calc.min(86400, catalog.duration_seconds - time) / 1800)
    }
    // Count the union of source intervals, split at midnight. Overlaps count once.
    let covered-until = 0
    for e in s.events.sorted(key: e => e.start_time_seconds) {
      let start = calc.max(e.start_time_seconds, covered-until)
      let end = e.end_time_seconds
      if start < end {
        for episode-day in range(int(calc.floor(start / 86400)), int(calc.ceil(end / 86400))) {
          let time = episode-day * 86400
          let day = calendar-weekday(s.calendar_start_day, time)
          active.at(day) += int((calc.min(end, time + 86400) - calc.max(start, time)) / 1800)
        }
      }
      covered-until = calc.max(covered-until, end)
    }
  }
  (observed: observed, active: active)
}
#let coverage = range(7).map(day => pct(weekday-counts.active.at(day), weekday-counts.observed.at(day)))
#let locations = range(2, 33).map(node => pct(events.filter(e => e.node_id == str(node)).len(), events.len()))
#let multiplicity = range(1, 5).map(n => pct(schedules.filter(s => s.events.len() == n).len(), schedules.len()))
#let overlap = pct(
  schedules
    .filter(s => s
      .events
      .enumerate()
      .any(((i, e)) => s
        .events
        .slice(i + 1)
        .any(other => (
          calc.max(e.start_time_seconds, other.start_time_seconds)
            < calc.min(e.end_time_seconds, other.end_time_seconds)
        ))))
    .len(),
  schedules.len(),
)
#let phase-bin-hours = 3
#let phase-hours = range(0, 24, step: phase-bin-hours)
#let start-phase = range(7).map(day => phase-hours.map(hour => pct(
  events
    .filter(e => (
      calc.rem(int(e.start_time_seconds / 86400), 7) == day
        and calc.rem(e.start_time_seconds / 3600, 24) >= hour
        and calc.rem(e.start_time_seconds / 3600, 24) < hour + phase-bin-hours
    ))
    .len(),
  events.len(),
)))

// Schedule summaries share a full-width row below the six main panels.
#let schedule-bar(values, names, colors, title) = context layout(size => {
  stack(dir: ttb, spacing: 3pt, align(center, label(title)), stack(dir: ltr, spacing: 0pt, ..values
    .enumerate()
    .map(((i, value)) => box(
      width: size.width * value / 100,
      height: 14pt,
      fill: colors.at(i),
      stroke: 0.4pt + white,
      align(center + horizon, text(size: annotation-size, fill: white)[
        #names.at(i) / #percentage(str(calc.round(value, digits: 1)))
      ]),
    ))))
})

#let bars(values, title, axis-title, ticks, color: accent, y-label: [%], tick-digits: 0) = context layout(size => {
  let height = 52pt
  let left-pad = 24pt
  let right-pad = 5pt
  let plot-width = size.width - left-pad - right-pad
  let y-title = rotate(-90deg, reflow: true, label(y-label))
  let x-label-height = calc.max(..ticks.map(((_, body)) => measure(label(body)).height))
  stack(dir: ttb, spacing: title-spacing, align(center, panel-title(title)), box(
    width: size.width,
    height: height + axis-footer-height,
  )[
    #place(top + left, bar-chart(
      (labels: values.map(_ => ""), values: values),
      width: size.width,
      height: height,
      bar-width: 0.82,
      show-values: false,
      // Preflow the vertical title so Primaviz measures its rotated bounds.
      // Its own -90-degree rotation restores the intended orientation.
      y-label: rotate(90deg, y-title),
      theme: (
        ..chart-theme,
        palette: (color,),
        axis-padding-left: left-pad,
        axis-padding-right: right-pad,
        tick-count: 3,
        tick-digits: tick-digits,
      ),
    ))
    #for (position, body) in ticks {
      let x = left-pad + (position + 0.5) / values.len() * plot-width
      place(top + left, dx: x, dy: height, line(end: (0pt, tick-length), stroke: axis-rule))
      place(top + left, dx: x, dy: height + x-tick-offset, box(width: 0pt, align(center, label(body))))
    }
    #place(top + left, dx: left-pad, dy: height + x-tick-offset + x-label-height + axis-title-gap, box(
      width: plot-width,
      align(center, label(axis-title)),
    ))
  ])
})

#let phase-panel = context layout(size => {
  let left-pad = 24pt
  let right-pad = 5pt
  let height = 52pt
  let maximum = calc.max(..start-phase.flatten())
  let maximum-label = label(percentage(str(calc.round(maximum, digits: 1))))
  let scale-width = colorbar-gap + colorbar-width
  let width = size.width - left-pad - right-pad - scale-width
  let y-title = rotate(-90deg, reflow: true, label([day]))
  let y-label-width = calc.max(..range(7).map(day => measure(label(str(day))).width))
  let x-label-height = measure(label([24])).height
  stack(dir: ttb, spacing: title-spacing, align(center, panel-title([(a) Start phase])), box(
    width: size.width,
    height: height + axis-footer-height,
  )[
    #for day in range(7) {
      place(top + left, dx: left-pad - y-tick-gap, dy: (day + 0.5) * height / 7, box(width: 0pt, height: 0pt, align(
        right + horizon,
        label(
          str(day),
        ),
      )))
      for (bin, hour) in phase-hours.enumerate() {
        place(top + left, dx: left-pad + hour * width / 24, dy: day * height / 7, rect(
          width: width * phase-bin-hours / 24,
          height: height / 7,
          stroke: none,
          fill: color.mix(
            (white, (1 - start-phase.at(day).at(bin) / maximum) * 100%),
            (accent, start-phase.at(day).at(bin) / maximum * 100%),
          ),
        ))
      }
    }
    #place(
      top + left,
      dx: y-title-position(left-pad, y-label-width, measure(y-title).width),
      dy: (height - measure(y-title).height) / 2,
      y-title,
    )
    #for hour in (0, 6, 12, 18, 24) {
      place(top + left, dx: left-pad + hour / 24 * width, dy: height + x-tick-offset, box(width: 0pt, align(
        center,
        label(
          str(hour),
        ),
      )))
    }
    #place(top + left, dx: left-pad, dy: height + x-tick-offset + x-label-height + axis-title-gap, box(
      width: width,
      align(center, label([hour of day])),
    ))
    #place(top + left, dx: left-pad + width + colorbar-gap, rect(
      width: colorbar-width,
      height: height,
      stroke: none,
      fill: gradient.linear(
        white,
        accent,
        angle: 270deg,
      ),
    ))
    #place(top + left, dx: left-pad + width + scale-width + colorbar-label-gap, maximum-label)
    #place(
      top + left,
      dx: left-pad + width + scale-width + colorbar-label-gap,
      dy: height - measure(label([0])).height,
      label([0]),
    )
  ])
})

#let statistics-figure = block(width: 100%)[
  #set par(justify: false)
  #stack(
    dir: ttb,
    spacing: panel-spacing,
    grid(
      columns: (1fr, 1fr, 1fr),
      column-gutter: panel-gutter,
      row-gutter: panel-spacing,
      phase-panel,
      bars(coverage, [(b) Weekday injection coverage], [calendar weekday index], range(7).map(day => (day, str(day)))),
      bars(
        duration-density,
        [(c) Event duration],
        [duration (#hour())],
        (2, 12, 32, 52, 72).map(hour => ((hour - 2) / 5 - 0.5, str(hour))),
        y-label: [#percentage()~per~value],
        tick-digits: 1,
      ),

      bars(
        histogram(strengths, strength-edges),
        [(d) Injection strength],
        [mass rate (#mg-min()), log scale],
        range(int(calc.ceil(strength-low)), int(calc.floor(strength-high)) + 1).map(exponent => (
          (exponent - strength-low) * 2 - 0.5,
          num("1e" + str(exponent)),
        )),
      ),
      bars(locations, [(e) Source location], [junction ID], (2, 8, 14, 20, 26, 32).map(node => (node - 2, str(node)))),
      bars(
        range(32).map(n => pct(reach.filter(v => v == n).len(), reach.len())),
        [(f) Relevant spatial reach],
        [nodes per cluster],
        (0, 6, 12, 18, 24, 31).map(n => (n, str(n))),
      ),
    ),
    grid(
      columns: (1fr, 1fr),
      column-gutter: panel-gutter,
      schedule-bar(
        multiplicity,
        ([*1*], [*2*], [*3*], [*4*]),
        (accent, orange, purple, green),
        [Events per schedule],
      ),
      schedule-bar(
        (overlap, 100 - overlap),
        ([*Yes*], [*No*]),
        (purple, accent),
        [Schedules with overlapping injections],
      ),
    ),
  )
]

#let statistics-caption = [Arsenic-event statistics for the fixed control-validation cases, comprising #events.len() events in #schedules.len() schedules (#catalog.schedules_per_scenario per validation hydraulic scenario, seed #identifier(catalog.seed)). All panels use these same cases, simulated without chlorine. Transported #acr("As(III)") excludes the estimated immediate local contribution of newly injected arsenic at source nodes. (a) Colors show event-start shares by simulation day modulo seven and three-hour interval. (b) For each calendar weekday index (0--6), active-injection steps are divided by all observed 30-minute steps of that weekday across complete episodes. Weekday indices follow the recorded calendar times. (c) Five-hour classes span 2--#hour[72]. Each event percentage is divided by the number of admissible half-hour duration values in its class, including #hour[72] in the final class. (d, e) Bars show event percentages. Source junctions 13 and 22 are excluded, as is reservoir 1. (f) The #clusters.len() contamination clusters show the number of nodes exceeding #mg-l[0.01] transported #acr("As(III)") at any sampled time, including the clearance tail. Clusters group source events that are not separated by persistent clearance of the transported concentration sum across monitored nodes at #mg-l[0.01], retaining zero-reach clusters. The bottom bars show the percentages of schedules containing one to four events (left) and with or without overlapping injections (right). Local source correction follows @section:appendix:transported-arsenic.]
