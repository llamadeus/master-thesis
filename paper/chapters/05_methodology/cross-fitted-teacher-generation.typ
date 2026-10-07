#import "../../numbers.typ": *
#import "../../figure-utils.typ": *

#let left-width = 250pt

#set text(size: title-size, fill: ink)
#set par(justify: false, leading: 0.45em, spacing: 0pt)
#set block(above: 0pt, below: 0pt)
#set align(center + top)

#let inner-gap = 18pt
#let side-width = inner-gap
#let query-gap = 42pt
#let ensemble-gap = 58pt
#let actor-height = 42pt
#let selection-height = 30pt
#let connector-label-gap = 5pt
#let group-title-height = 24pt
#let group-outline-top = 12pt
#let lane-widths = (
  side-width,
  (left-width - 2 * side-width - query-gap) / 2,
  query-gap,
  (left-width - 2 * side-width - query-gap) / 2,
  side-width,
  ensemble-gap,
  1fr,
)
#let lane-row(..items) = grid(columns: lane-widths, align: center + horizon, ..items.pos())

#let input-split() = layout(size => {
  let actor-width = (left-width - 2 * side-width - query-gap) / 2
  let training-width = 2 * side-width + 2 * actor-width + query-gap
  diagram-fork(
    size.width,
    size.width / 2,
    training-width / 2,
    (training-width + ensemble-gap + size.width) / 2,
    height: 36pt,
  )
})

#let actor-box(body, fill: surface) = block(
  width: 100%,
  height: actor-height,
  inset: diagram-actor-inset,
  fill: fill,
  stroke: axis-rule,
  radius: diagram-radius,
  align(center + horizon, body),
)

// Connectors occupy the full gap and meet the centers of the adjacent boxes.
#let horizontal-connector(label, reverse: false) = layout(size => {
  let mid = actor-height / 2
  let head = diagram-head-height
  let half-head = diagram-head-width / 2
  let label-inset = if reverse { (left: side-width + 2pt) } else { 0pt }
  let shaft = if reverse {
    line(start: (head, mid), end: (size.width, mid), stroke: diagram-connector)
  } else {
    line(start: (0pt, mid), end: (size.width - head, mid), stroke: diagram-connector)
  }
  let tip = if reverse {
    polygon((0pt, mid), (head, mid - half-head), (head, mid + half-head), fill: muted, stroke: none)
  } else {
    polygon(
      (size.width, mid),
      (size.width - head, mid - half-head),
      (size.width - head, mid + half-head),
      fill: muted,
      stroke: none,
    )
  }
  grid(
    columns: (1fr,),
    rows: (0pt, 0pt, auto),
    align: left + top,
    box(width: size.width, height: 0pt, shaft),
    box(width: size.width, height: 0pt, tip),
    box(width: size.width, height: actor-height, box(
      width: size.width,
      height: mid - connector-label-gap,
      inset: label-inset,
      align(center + bottom, plot-label(label)),
    )),
  )
})

#let branch-header() = layout(size => {
  let branch-width = (size.width - 2 * side-width - query-gap) / 2
  let first-x = side-width + branch-width / 2
  let second-x = side-width + branch-width + query-gap + branch-width / 2
  let first = line(start: (first-x, 0pt), end: (first-x, group-title-height), stroke: diagram-connector)
  let second = line(start: (second-x, 0pt), end: (second-x, group-title-height), stroke: diagram-connector)
  grid(
    columns: (1fr,),
    rows: (0pt, 0pt, auto),
    align: left + top,
    box(width: size.width, height: 0pt, first),
    box(width: size.width, height: 0pt, second),
    box(width: size.width, height: group-title-height, inset: (top: 3pt), align(center + top, plot-label(
      [_Five-fold cross-fitting_],
      fill: ink,
    ))),
  )
})

#let selection-bridge() = layout(size => {
  let actor-width = (left-width - 2 * side-width - query-gap) / 2
  let forecaster-right = side-width + 2 * actor-width + query-gap
  let group-right = forecaster-right + side-width
  let validation-x = (group-right + ensemble-gap + size.width) / 2
  let target-x = forecaster-right - 18pt
  let branch-y = 16pt
  let selection-label = plot-label([Model selection])
  let selection-label-y = branch-y - connector-label-gap - measure(selection-label).height
  let selection-rule = (paint: muted, thickness: diagram-connector-width, dash: "dashed")
  let bend = diagram-connector-radius
  let elbow-size = 2 * bend
  let branch = box(width: size.width, height: 0pt, {
    place(left + top, dx: validation-x - elbow-size, dy: branch-y - elbow-size, diagram-bend(
      "bottom-right",
      width: elbow-size,
      height: elbow-size,
      stroke: selection-rule,
    ))
    place(left + top, dx: target-x, dy: branch-y, diagram-bend(
      "top-left",
      width: validation-x - target-x - elbow-size,
      height: selection-height - diagram-head-height - branch-y,
      stroke: selection-rule,
    ))
  })
  let head = diagram-head(origin: (target-x - diagram-head-width / 2, selection-height - diagram-head-height))
  grid(
    columns: (1fr,),
    rows: (0pt, 0pt, 0pt, auto),
    align: left + top,
    box(width: size.width, height: 0pt, branch),
    box(width: size.width, height: 0pt, head),
    box(width: size.width, height: 0pt, place(
      left + top,
      dx: group-right + 8pt,
      dy: selection-label-y,
      selection-label,
    )),
    grid(
      columns: lane-widths, align: center + horizon,
      [], diagram-arrow(height: selection-height), [],
      diagram-arrow(height: selection-height), [], [],
      diagram-arrow(height: selection-height),
    ),
  )
})

#let fold-process() = layout(size => {
  let actor-width = (left-width - 2 * side-width - query-gap) / 2
  let group-width = 2 * side-width + 2 * actor-width + query-gap
  let input-card(body) = diagram-box(body, fill: surface, inset: diagram-input-inset)
  let held-out-card = input-card([
    #diagram-title[Held-out fold] \
    #plot-label[16 scenarios / 400 trajectories]
  ])
  let other-folds-card = input-card([
    #diagram-title[Other four folds] \
    #plot-label[64 scenarios / 1600 trajectories]
  ])
  let input-height = calc.max(
    measure(held-out-card, width: actor-width).height,
    measure(other-folds-card, width: actor-width).height,
  )
  let body = grid(
    columns: lane-widths,
    align: center + horizon,

    grid.cell(colspan: 5, branch-header()), [],
    align(center, diagram-stem(group-title-height)),

    [], diagram-arrow(height: 4pt), [],
    diagram-arrow(height: 4pt), [], [], diagram-stem(4pt),

    [], held-out-card, [], other-folds-card, [], [],
    diagram-stem(input-height),

    grid.cell(colspan: 7, selection-bridge()),

    [],
    actor-box([
      #diagram-title[MPC teacher] \
      #plot-label[9600 labels per fold]
    ]),
    horizontal-connector([queries]),
    actor-box([
      #diagram-title[Forecaster] \
      #plot-label[one trained per fold]
    ]),
    grid.cell(colspan: 2, horizontal-connector(
      [
        queries to all \
        five forecasters
      ],
      reverse: true,
    )),
    actor-box([
      #diagram-title()[Ensemble MPC] \
      #plot-label[validation teacher]
    ]),

    [], diagram-stem(inner-gap), [],
    [], [], [], diagram-stem(inner-gap),
  )
  let outline = rect(
    width: group-width,
    height: measure(body, width: size.width).height - group-outline-top,
    fill: diagram-group-fill,
    radius: diagram-radius,
    stroke: axis-rule,
  )
  grid(
    columns: (1fr,),
    rows: (0pt, auto),
    align: left + top,
    box(width: size.width, height: 0pt, place(left + top, dy: group-outline-top, outline)),
    body,
  )
})

#block(width: 100%, breakable: false)[
  #align(center)[
    #diagram-box(
      [
        #diagram-title[Precomputed offline trajectories] \
        #plot-label[100 scenarios / 2500 trajectories]
      ],
      fill: surface,
      width: auto,
      inset: diagram-summary-inset,
    )
  ]

  #input-split()

  #lane-row(
    grid.cell(colspan: 5, diagram-box(
      [
        #diagram-title[Training split] \
        #plot-label[80 scenarios / 2000 trajectories]
      ],
      fill: surface,
    )),
    [],
    diagram-box([
      #diagram-title()[Validation split] \
      #plot-label[10 scenarios / 250 trajectories]
    ]),
  )

  #lane-row(
    [],
    align(center, diagram-stem(12pt)),
    [],
    align(center, diagram-stem(12pt)),
    [],
    [],
    align(center, diagram-stem(12pt)),
  )

  #fold-process()

  #lane-row(
    [],
    diagram-arrow(height: 12pt),
    [],
    [],
    [],
    [],
    diagram-arrow(height: 12pt),
  )

  #lane-row(
    grid.cell(colspan: 5, diagram-box(
      [
        #diagram-title[Combined out-of-fold demonstrations] \
        #plot-label[#num[48000] teacher labels from five folds]
      ],
      fill: surface,
    )),
    [],
    diagram-box([
      #diagram-title()[Behavior cloning validation targets] \
      #plot-label[6000 labels for validation]
    ]),
  )
]
