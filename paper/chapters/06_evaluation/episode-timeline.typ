#import "../../figure-utils.typ": *

#let timeline-warmup-fill = surface
#let timeline-main-fill = blue-tint
#let timeline-tail-fill = orange-tint
#let timeline-main-rule = (paint: accent, thickness: 0.65pt)
#let timeline-tail-rule = (paint: orange, thickness: 0.65pt)

#set par(justify: false)

#let day-card(day) = {
  let warmup = day <= 2
  let fill = if warmup {
    timeline-warmup-fill
  } else if day <= 23 {
    timeline-main-fill
  } else {
    timeline-tail-fill
  }
  let outline = if warmup {
    axis-rule
  } else if day <= 23 {
    timeline-main-rule
  } else {
    timeline-tail-rule
  }

  box(
    width: 100%,
    height: 21pt,
    fill: fill,
    stroke: outline,
    radius: diagram-radius,
    align(center + horizon, text(
      size: annotation-size,
      weight: "semibold",
      fill: ink,
      str(day),
    )),
  )
}

#let step-brace(label, label-align: center) = layout(size => {
  let width = size.width
  let brace = rotate(
    90deg,
    reflow: true,
    text(fill: muted, math.lr(size: width, sym.brace.r)),
  )

  stack(
    dir: ttb,
    spacing: 3pt,
    box(width: width, height: 12pt, align(center + horizon, brace)),
    align(label-align, plot-label(label)),
  )
})

#let nowrap(body) = context {
  let w = measure(body).width
  box(width: w, body)
}

#let step-braces() = layout(size => {
  let gap = 2pt
  let day-width = (size.width - 24 * gap) / 25
  grid(
    columns: (
      2 * day-width + gap,
      21 * day-width + 20 * gap,
      2 * day-width + gap,
    ),
    column-gutter: gap,
    step-brace(
      [#box(width: 100%, align(center, [*2 days*])) \ #nowrap[return excluded] \ #nowrap[no physical metrics]],
      label-align: left,
    ),
    step-brace([*21 days* \ #nowrap[validation return] \ #nowrap[physical metrics fully evaluated]]),
    step-brace(
      [#box(width: 100%, align(
          center,
          [*2 days*],
        )) \ #nowrap[validation return] \ #nowrap[physical metrics until clearance]],
      label-align: right,
    ),
  )
})

#block(width: 100%, breakable: false, stack(
  dir: ttb,
  spacing: 0pt,
  grid(
    columns: (1fr,) * 25,
    column-gutter: 2pt,
    ..range(1, 26).map(day-card),
  ),
  diagram-label-gap,
  step-braces(),
  legend-spacing,
  legend-row(
    legend-item(legend-patch(timeline-warmup-fill, stroke: axis-rule), [Warm-up]),
    legend-item(legend-patch(timeline-main-fill, stroke: timeline-main-rule), [Main window]),
    legend-item(legend-patch(timeline-tail-fill, stroke: timeline-tail-rule), [Clearance tail]),
  ),
))
