#import "@preview/acrostiche:0.7.0": acr
#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": multi-line-chart

#let data = json("../../dataset/reference-window-example.json")
#assert(data.schema_version == 2 and data.dataset == "control_full")
#assert(data.split == "validation" and data.policy == "zero-injection")
#assert(data.scenario_id == "scenario-85" and data.schedule_index == 2)
#assert(data.events.len() == 1 and data.reference_windows.len() == 1)

#let event = data.events.first()
#let window = data.reference_windows.first()
#let n = data.time_seconds.len()
#let labels = data.time_seconds.map(_ => "")
#let index-at(time) = (time - data.plot_start_time_seconds) / data.timestep_seconds
#let end-hour = int((data.plot_end_time_seconds - event.start_time_seconds) / 3600)
#let hour-ticks = (-4,) + range(0, end-hour, step: 8) + (end-hour,)
#let arsenic-upper = calc.ceil(calc.max(..data.local_source_arsenic_sum_mg_l, ..data.transported_arsenic_sum_mg_l))
#let source-hours = (event.end_time_seconds - event.start_time_seconds) / 3600
#let source-end-to-clearance-hours = (window.end_time_seconds - event.end_time_seconds) / 3600
#let window-fill = reference-event-fill
// Values above the plotted y range place both duration spans in the top padding.
#let span-y = arsenic-upper * 1.08
#let span-label-y = arsenic-upper * 1.15

// Primaviz draws the trajectories and axes. Its annotations locate the
// source, reference window, and lag on the 30-minute report grid.
#let window-annotation(upper) = (
  type: "rect",
  x1: index-at(window.start_time_seconds),
  x2: index-at(window.end_time_seconds),
  y1: 0,
  y2: upper,
  fill: window-fill,
)
#let time-annotations = (
  hour-ticks
    .map(hour => (
      (
        type: "v-line",
        value: index-at(event.start_time_seconds + hour * 3600),
        color: grid-color,
        stroke: grid-line-size,
      ),
      (
        type: "content",
        x: index-at(event.start_time_seconds + hour * 3600),
        y: 0,
        anchor: "top",
        dy: x-tick-offset,
        body: plot-label(str(hour)),
      ),
    ))
    .flatten()
)
#let chart-style = (
  ..chart-theme,
  axis-padding-left: 22pt,
  axis-padding-top: 25pt,
  axis-padding-bottom: 20pt,
  tick-count: 4,
  rotated-threshold: 100000,
)

#let arsenic-series = (
  (name: "Local source", values: data.local_source_arsenic_sum_mg_l),
  (name: "Transported", values: data.transported_arsenic_sum_mg_l),
  (name: "Axis floor", values: (0,) * n),
  (name: "Axis ceiling", values: (arsenic-upper,) * n),
)
#let clear-color = white.transparentize(100%)
#let chart-height = 150pt

#let reference-window-figure = block(width: 100%, breakable: false)[
  #set par(justify: false)
  #context layout(size => stack(
    dir: ttb,
    spacing: legend-spacing,
    box(width: size.width, height: chart-height, clip: true, multi-line-chart(
      (labels: labels, series: arsenic-series),
      width: size.width,
      height: chart-height,
      show-points: false,
      show-legend: false,
      line-width: line-size,
      x-label: [hours since source onset],
      y-label: [#mg-l()],
      annotations: (
        window-annotation(arsenic-upper),
        (
          type: "v-line",
          value: index-at(event.end_time_seconds),
          color: muted,
          stroke: guide-line-size,
          dash: "dotted",
        ),
        (
          type: "v-line",
          value: index-at(window.end_time_seconds),
          color: orange,
          stroke: guide-line-size,
          dash: "dotted",
        ),
        (
          type: "errorbar",
          orientation: "h",
          y: span-y,
          low: index-at(event.start_time_seconds),
          high: index-at(event.end_time_seconds),
          color: muted,
          stroke: span-line-size,
          cap-width: 4pt,
        ),
        (
          type: "content",
          x: (index-at(event.start_time_seconds) + index-at(event.end_time_seconds)) / 2,
          y: span-label-y,
          body: plot-label([#hour(str(source-hours))], fill: muted),
        ),
        (
          type: "errorbar",
          orientation: "h",
          y: span-y,
          low: index-at(event.end_time_seconds),
          high: index-at(window.end_time_seconds),
          color: orange,
          stroke: span-line-size,
          cap-width: 4pt,
        ),
        (
          type: "content",
          x: (index-at(event.end_time_seconds) + index-at(window.end_time_seconds)) / 2,
          y: span-label-y,
          body: plot-label([#hour(str(source-end-to-clearance-hours))], fill: orange),
        ),
      )
        + time-annotations,
      theme: (
        ..chart-style,
        axis-padding-right: measure(plot-label(str(end-hour))).width / 2,
        palette: (muted, orange, clear-color, clear-color),
      ),
    )),
    legend-row(
      legend-item(legend-line(muted), [Local source contribution]),
      legend-item(legend-line(orange), [Transported #acr("As(III)")]),
      legend-item(legend-patch(window-fill, stroke: guide-line-size + orange), [Reference-event window]),
    ),
  ))
]

#reference-window-figure
