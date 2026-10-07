#import "../../numbers.typ": *
#import "../../figure-utils.typ": *
#import "@preview/primaviz:0.10.0": multi-line-chart
#import "data.typ": case-trajectory-data
#import "evaluation-plots.typ": chart-side-padding, results-chart-theme, ribbon-bounds, sd-ribbon, titled-panel

#let case-trajectory-figure(data: case-trajectory-data) = {
  assert(data.schema_version == 1)
  let policies = (data.controllers.ppo_best, data.controllers.finetuned_best)
  let n = data.time_days.len()
  let window-fill = reference-event-fill
  let transparent = white.transparentize(100%)
  let titles = (
    [(a) Chlorine injection],
    [(b) Nodal chlorine concentration],
    [(c) Transported As(III) concentration sum],
  )
  let units = ([#g-min()], [#mg-l()], [#mg-l()])
  let series = (
    policies.enumerate().map(((i, p)) => (name: str(i), values: p.injection_g_min)),
    policies.enumerate().map(((i, p)) => (name: str(i), values: p.chlorine_mean_mg_l)),
    policies.enumerate().map(((i, p)) => (name: str(i), values: p.transported_arsenic_sum_mg_l))
      + ((name: "Zero-chlorine reference", values: data.reference_transported_arsenic_sum_mg_l),),
  )
  let upper-values = series
    .enumerate()
    .map(((i, panel)) => {
      let values = (0,) + panel.map(s => s.values).flatten()
      if i == 1 {
        values += policies
          .map(p => range(n).map(j => p.chlorine_mean_mg_l.at(j) + p.chlorine_population_sd_mg_l.at(j)))
          .flatten()
      }
      calc.max(..values)
    })
  let bounds = upper-values.map(v => ribbon-bounds((0, v)))
  let rate-upper = calc.ceil(calc.max(..data.source_injection_g_min) / 10) * 10
  let rate-ticks = (0, rate-upper / 2, rate-upper)
  let rate-label(value) = plot-label(num(str(int(value))))
  let labels = data.time_days.map(t => if calc.rem(t, 5) == 0 { str(int(t)) } else { [] })
  set par(justify: false)
  block(width: 100%, context layout(size => {
    let paddings = range(3).map(i => chart-side-padding(units.at(i), bounds.at(i), last-x: [#num(
      data.time_days.last(),
    )]))
    let rate-title = vertical-axis-title([#g-min()])
    let rate-label-width = calc.max(..rate-ticks.map(value => measure(rate-label(value)).width))
    let theme = (
      results-chart-theme
        + (
          axis-padding-left: calc.max(..paddings.map(p => p.axis-padding-left)),
          axis-padding-right: y-tick-gap + rate-label-width + axis-title-gap + measure(rate-title).width,
        )
    )
    let height = 78pt
    let plot-width = size.width - theme.axis-padding-left - theme.axis-padding-right
    stack(
      dir: ttb,
      spacing: panel-spacing,
      ..range(3).map(i => titled-panel(titles.at(i), {
        let extent = (
          (name: "Axis floor", values: (0,) * n),
          // Keep the original maximum: rounding it twice changes native ticks.
          (name: "Axis ceiling", values: (upper-values.at(i),) * n),
        )
        let annotations = data.reference_windows.map(w => (
          type: "rect",
          x1: w.start_index,
          x2: w.end_index,
          y1: 0,
          y2: bounds.at(i).high,
          fill: window-fill,
        ))
        annotations += (2, 23).map(day => (
          type: "v-line",
          value: data.time_days.position(t => t == day),
          color: muted,
          stroke: guide-line-size,
          dash: "dashed",
        ))
        if i == 1 {
          annotations += (0.2, 0.4).map(v => (
            type: "h-line",
            value: v,
            color: muted,
            stroke: guide-line-size,
            dash: "dotted",
          ))
        }
        let palette = if i == 2 { (accent, orange, muted) } else { (accent, orange) }
        let footer = if i == 2 { axis-footer-height } else { x-tick-offset + label-size + tick-length }
        box(width: size.width, height: height + footer)[
          #let y-title = vertical-axis-title(units.at(i))
          #place(top + left, dy: (height - measure(y-title).height) / 2, y-title)
          #if i == 1 {
            for (j, p) in policies.enumerate() {
              let points = range(n).map(k => (
                x: data.time_days.at(k) / data.time_days.last(),
                mean: p.chlorine_mean_mg_l.at(k),
                sd: p.chlorine_population_sd_mg_l.at(k),
              ))
              place(top + left, dx: theme.axis-padding-left, sd-ribbon(
                points,
                plot-width,
                height,
                bounds.at(i),
                uncertainty-fill(palette.at(j)),
                nonnegative: true,
              ))
            }
          }
          #place(top + left, multi-line-chart(
            (labels: labels, series: series.at(i) + extent),
            width: size.width,
            height: height,
            show-points: false,
            show-legend: false,
            line-width: line-size,
            x-label: if i == 2 { [Simulation day] } else { [] },
            y-label: none,
            annotations: annotations,
            theme: theme + (palette: palette + (transparent, transparent)),
          ))
          #if i == 2 {
            // Preserve exact rate changes as rectangular steps on a separate
            // scale while retaining the transported-concentration axis.
            let x-at(j) = theme.axis-padding-left + data.time_days.at(j) / data.time_days.last() * plot-width
            let y-at(j) = height * (1 - data.source_injection_g_min.at(j) / rate-upper)
            let steps = (curve.move((x-at(0), y-at(0))),)
            for j in range(1, n) {
              if data.source_injection_g_min.at(j) != data.source_injection_g_min.at(j - 1) {
                steps += (curve.line((x-at(j), y-at(j - 1))), curve.line((x-at(j), y-at(j))))
              }
            }
            steps.push(curve.line((x-at(n - 1), y-at(n - 1))))
            place(top + left, curve(..steps, fill: none, stroke: injection-rule(muted)))
            place(top + left, dx: theme.axis-padding-left + plot-width, line(end: (0pt, height), stroke: axis-rule))
            for value in rate-ticks {
              place(
                top + left,
                dx: theme.axis-padding-left + plot-width + y-tick-gap,
                dy: height * (1 - value / rate-upper),
                box(height: 0pt, align(horizon, rate-label(value))),
              )
            }
            place(
              top + left,
              dx: theme.axis-padding-left + plot-width + y-tick-gap + rate-label-width + axis-title-gap,
              dy: (height - measure(rate-title).height) / 2,
              rate-title,
            )
          }
        ]
      })),
      legend-column(
        legend-row(
          legend-item(legend-patch(uncertainty-fill(accent), line-color: accent), [Selected baseline policy]),
          legend-item(legend-patch(uncertainty-fill(orange), line-color: orange), [Selected pretrained policy]),
          legend-item(legend-line(muted), [Zero-chlorine reference (c)]),
        ),
        legend-row(
          legend-item(legend-patch(window-fill), [Reference-event windows]),
          legend-item(legend-line(muted, stroke: boundary-rule), [Normal chlorine bounds (b)]),
        ),
        legend-row(
          legend-item(legend-line(muted, stroke: injection-rule(muted)), [Arsenic source injection (c, right axis)]),
        ),
      ),
    )
  }))
}
