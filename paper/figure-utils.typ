// Shared figure styles and drawing helpers for the entire thesis.
// Figure-specific data, renderers, routing, captions, and composition stay
// in the chapter files. This module provides only shared styles and primitives.
#import "@preview/primaviz:0.10.0": waffle-chart

#let accent = rgb("356A8A")
#let orange = rgb("B85F27")
#let purple = rgb("6C6B9F")
#let green = rgb("43846B")
#let rose = rgb("B65D69")
#let ink = rgb("26343C")
#let muted = rgb("5D686E")
#let axis-color = rgb("A5AFB5")
#let grid-color = rgb("E2E7EA")
#let surface = rgb("F4F7F9")
#let reference-bar-fill = rgb("D5DCE1")
#let blue-tint = color.mix((white, 92%), (accent, 8%))
#let orange-tint = color.mix((white, 92%), (orange, 8%))
#let uncertainty-fill(paint) = paint.transparentize(85%)
#let active-injection-fill = orange.transparentize(91%)
#let reference-event-fill = orange.transparentize(92%)
#let axis-rule = (
  thickness: 0.45pt,
  paint: axis-color,
)
#let grid-line-size = 0.35pt
#let grid-rule = grid-line-size + grid-color
#let line-size = 1pt
#let emphasized-line-size = 1.25 * line-size
#let seed-line-size = 0.5 * line-size
#let guide-line-size = 0.5pt
#let span-line-size = 0.6pt
#let label-size = 7pt
#let title-size = 8pt
#let title-spacing = 7pt
#let panel-spacing = 9pt
#let panel-gutter = 12pt
#let annotation-size = 6.5pt
#let axis-title-gap = 7pt
#let x-tick-offset = 6pt
#let y-tick-gap = 2pt
#let tick-length = 3pt
#let axis-footer-height = 33pt
#let colorbar-gap = 6pt
#let colorbar-width = 5pt
#let colorbar-label-gap = 3pt
#let boundary-rule = (paint: muted, thickness: guide-line-size, dash: "dotted")
#let injection-rule(paint) = (paint: paint, thickness: line-size, dash: "dashed")
#let source-line-size = emphasized-line-size
#let downstream-line-size = 0.75 * line-size
#let concentration-rule(paint, source: false) = if source {
  source-line-size + paint
} else {
  downstream-line-size + paint
}
#let chart-theme = (
  container-inset: 0pt,
  axis-padding-top: 0pt,
  axis-padding-bottom: 0pt,
  axis-label-size: label-size,
  axis-title-size: label-size,
  axis-label-gap: 2 * y-tick-gap,
  axis-title-gap: axis-title-gap,
  text-color: muted,
  axis-stroke: axis-rule,
  show-grid: false,
  rotated-threshold: 100000,
)
#let y-title-position(left-pad, tick-width, title-width) = {
  left-pad - y-tick-gap - tick-width - axis-title-gap - title-width
}

// Fixed font metrics keep every label on the same baseline. Center symbols
// against the cap height; descenders must not shift individual labels upward.
#let legend-symbol-width = 12pt
#let legend-symbol-height = 7pt
#let legend-text-gap = 3pt
#let legend-entry-gap = 16pt
#let legend-row-gap = 3pt
#let legend-spacing = panel-spacing
#let legend-item(symbol, body) = grid(
  columns: (legend-symbol-width, auto),
  column-gutter: legend-text-gap,
  align: left + horizon,
  box(width: legend-symbol-width, height: legend-symbol-height, place(center + horizon, symbol)),
  text(size: label-size, fill: muted, top-edge: "cap-height", bottom-edge: "baseline", body),
)
#let legend-line(paint, stroke: none) = line(
  length: legend-symbol-width,
  stroke: if stroke == none { line-size + paint } else { stroke },
)
#let legend-patch(fill, stroke: none, line-color: none, line-stroke: none) = box(
  width: legend-symbol-width,
  height: legend-symbol-height,
  fill: fill,
  stroke: stroke,
  if line-color != none { place(horizon, legend-line(line-color, stroke: line-stroke)) },
)
#let legend-row(..items) = align(center, grid(
  columns: (auto,) * items.pos().len(),
  column-gutter: legend-entry-gap,
  align: left + horizon,
  ..items.pos(),
))
#let legend-column(..items) = align(center, grid(
  columns: (auto,),
  row-gutter: legend-row-gap,
  align: left + horizon,
  ..items.pos(),
))
#let series-colors = (accent, orange, purple, green, rose)
#let plot-label(body, fill: muted, size: label-size) = text(size: size, fill: fill, body)
#let vertical-axis-title(body) = rotate(-90deg, reflow: true, text(
  top-edge: "bounds",
  bottom-edge: "bounds",
  plot-label(body),
))
#let panel-title(body) = text(size: title-size, weight: "semibold", fill: ink, body)


// Flow diagrams
#let diagram-radius = 2pt
#let diagram-connector-radius = 5pt
#let diagram-gutter = 8pt
#let diagram-side-gap = 14pt
#let diagram-branch-gap = 2em
#let diagram-cell-size = 1.8em
#let diagram-label-gap = 7pt
#let diagram-connector-width = 0.6pt
#let diagram-connector = diagram-connector-width + muted
#let diagram-head-width = 4pt
#let diagram-head-height = 3pt
#let diagram-box-inset = (x: 4pt, y: 10pt)
#let diagram-input-inset = (x: 4pt, y: 7pt)
#let diagram-actor-inset = (x: 4pt, y: 4pt)
#let diagram-summary-inset = (x: 12pt, y: 10pt)
#let diagram-segment-inset = (x: 8pt, y: 12pt)
#let diagram-group-fill = color.mix((white, 50%), (surface, 50%))
#let diagram-label(body, fill: muted) = block({
  set text(size: label-size, fill: fill, top-edge: "cap-height", bottom-edge: "bounds")
  body
})
#let diagram-title(body, fill: accent) = text(weight: "semibold", fill: fill, body)
#let diagram-stem(height, stroke: diagram-connector) = line(end: (0pt, height), stroke: stroke)
#let diagram-head(paint: muted, width: diagram-head-width, height: diagram-head-height, origin: (0pt, 0pt)) = polygon(
  origin,
  (origin.at(0) + width, origin.at(1)),
  (origin.at(0) + width / 2, origin.at(1) + height),
  fill: paint,
  stroke: none,
)
// A border-based bend stays in flow and works with solid or dashed strokes.
#let diagram-bend(
  corner,
  width: 100%,
  height: diagram-connector-radius,
  stroke: diagram-connector,
  radius: diagram-connector-radius,
) = {
  let is-top = corner in ("top-left", "top-right")
  let is-left = corner in ("top-left", "bottom-left")
  box(
    width: width,
    height: height,
    radius: ((corner): radius),
    stroke: (
      top: if is-top { stroke } else { none },
      bottom: if is-top { none } else { stroke },
      left: if is-left { stroke } else { none },
      right: if is-left { none } else { stroke },
    ),
  )
}
#let diagram-fork(
  width,
  source-x,
  left-x,
  right-x,
  height: 40pt,
  head-width: diagram-head-width,
  head-height: diagram-head-height,
) = {
  let shoulder-y = height - head-height - 13pt
  let bend = diagram-connector-radius
  let elbow-size = 2 * bend
  let stem = line(start: (source-x, 0pt), end: (source-x, shoulder-y - elbow-size), stroke: diagram-connector)
  // Use the same border radius at the split and at the downward turns.
  let left-branch = box(width: width, height: 0pt, {
    place(left + top, dx: source-x - elbow-size, dy: shoulder-y - elbow-size, diagram-bend(
      "bottom-right",
      width: elbow-size,
      height: elbow-size,
    ))
    place(left + top, dx: left-x, dy: shoulder-y, diagram-bend(
      "top-left",
      width: source-x - left-x - elbow-size,
      height: height - head-height - shoulder-y,
    ))
  })
  let right-branch = box(width: width, height: 0pt, {
    place(left + top, dx: source-x, dy: shoulder-y - elbow-size, diagram-bend(
      "bottom-left",
      width: elbow-size,
      height: elbow-size,
    ))
    place(left + top, dx: source-x + elbow-size, dy: shoulder-y, diagram-bend(
      "top-right",
      width: right-x - source-x - elbow-size,
      height: height - head-height - shoulder-y,
    ))
  })
  grid(
    columns: (1fr,),
    rows: (0pt, 0pt, 0pt, 0pt, 0pt, auto),
    align: left + top,
    box(width: width, height: 0pt, stem),
    box(width: width, height: 0pt, left-branch),
    box(width: width, height: 0pt, right-branch),
    box(width: width, height: 0pt, diagram-head(width: head-width, height: head-height, origin: (
      left-x - head-width / 2,
      height - head-height,
    ))),
    box(width: width, height: 0pt, diagram-head(width: head-width, height: head-height, origin: (
      right-x - head-width / 2,
      height - head-height,
    ))),
    box(width: width, height: height),
  )
}
#let diagram-arrow(
  height: 20pt,
  paint: muted,
  stroke: diagram-connector,
  head-width: diagram-head-width,
  head-height: diagram-head-height,
) = align(
  center,
  stack(
    dir: ttb,
    spacing: 0pt,
    diagram-stem(height - head-height, stroke: stroke),
    diagram-head(paint: paint, width: head-width, height: head-height),
  ),
)
#let diagram-box(
  body,
  fill: surface,
  inset: diagram-box-inset,
  width: 100%,
  stroke: axis-rule,
  radius: diagram-radius,
) = block(
  width: width,
  inset: inset,
  radius: radius,
  stroke: stroke,
  fill: fill,
  align(center, body),
)
// Matrix of equally sized cells, used for paired trajectory inputs.
#let diagram-matrix(rows: 5, columns: 5, cell-size: diagram-cell-size) = context box(
  width: columns * cell-size + 16pt,
  waffle-chart(
    (labels: ("Cells",), values: (rows * columns,)),
    rows: rows,
    cols: columns,
    size: (columns * cell-size).to-absolute(),
    gap: 0.12em.to-absolute(),
    rounded: false,
    show-legend: false,
    theme: (palette: (accent,), container-inset: 0pt),
  ),
)
#let diagram-loop(upper: true, width: 100%, height: 2em) = box(
  width: width,
  height: height,
  radius: (
    top: if upper { diagram-connector-radius } else { 0pt },
    bottom: if upper { 0pt } else { diagram-connector-radius },
  ),
  stroke: (
    top: if upper { diagram-connector } else { none },
    bottom: if upper { none } else { diagram-connector },
    left: diagram-connector,
    right: diagram-connector,
  ),
)




// Segmented diagram cards
#let branch-cell(body, first: false, last: false, inset: diagram-segment-inset) = grid.cell(
  align: center + horizon,
  stroke: if not first and not last {
    (left: axis-rule, right: axis-rule, top: none, bottom: none)
  } else { none },
  box(
    align(center + horizon, body),
    width: 100%,
    fill: if first { surface } else if last { surface } else { none },
    radius: (
      top: if first { diagram-radius } else { 0pt },
      bottom: if last { diagram-radius } else { 0pt },
    ),
    inset: inset,
    stroke: (
      left: if first or last { axis-rule } else { none },
      right: if first or last { axis-rule } else { none },
      top: if first { axis-rule } else { none },
      bottom: if last { axis-rule } else { none },
    ),
  ),
)

// Diagram cells use their content height directly, without an enclosing box.


// Diagram cell layout
#let diagram-cell(body, inset: (x: 16pt, bottom: 16pt)) = grid.cell(
  align: center + top,
  inset: inset,
  stroke: (left: axis-rule, right: axis-rule, top: none, bottom: none),
  body,
)


// Diagram cell layout
#let branch-label(body, fill: muted) = diagram-cell(
  inset: (x: 8pt, top: 8pt, bottom: diagram-label-gap),
  diagram-label(fill: fill, body),
)


// Alternative series palette already used by the results plots.
#let comparison-colors = (accent, rgb("C27620"), purple, rgb("4B8762"), rgb("AF5962"))

// A continuous unsmoothed path on a plot's local coordinate system.
#let plot-path(points, stroke) = place(top + left, curve(
  curve.move(points.first()),
  ..points.slice(1).map(p => curve.line(p)),
  stroke: stroke,
))
