#import "../../figure-utils.typ": *

#set text(size: title-size, fill: ink)
#set par(justify: false, leading: 0.45em)

#let arsenic-label = rotate(-90deg, reflow: true, text(size: label-size, fill: ink)[5 arsenic schedules])

// The invisible matching label balances the side columns without fixed widths.
#let pairing = grid(
  columns: (auto, auto, auto),
  column-gutter: 0.4em,
  align: center + horizon,
  arsenic-label, diagram-matrix(), hide(arsenic-label),
)

// Each node spans two equal tracks. The loop ends at their shared centers.
#let interaction = grid(
  columns: (1fr,),
  row-gutter: diagram-label-gap + 2pt,
  align: center,
  grid(
    columns: (1fr, 1fr, 1.5fr, 1fr, 1fr),
    align: center + horizon,
    [], grid.cell(colspan: 3, diagram-loop()), [],

    grid.cell(colspan: 2, diagram-stem(diagram-head-height)), [], grid.cell(colspan: 2, diagram-head()),

    grid.cell(colspan: 2, diagram-box([Policy])),
    [],
    grid.cell(colspan: 2, diagram-box([Simulator])),

    grid.cell(colspan: 2, rotate(180deg, reflow: true, diagram-head())),
    [],
    grid.cell(colspan: 2, diagram-stem(diagram-head-height)),

    [], grid.cell(colspan: 3, diagram-loop(upper: false)), [],
  ),
  diagram-label[Chlorine concentrations],
)

// Both branches share automatic row heights, including the result row.
#block(width: 100%, breakable: false)[
  #set align(left + top)
  #set block(above: 0pt, below: 0pt)
  #align(center)[
    #block(inset: diagram-summary-inset, stroke: axis-rule, fill: surface, radius: diagram-radius)[
      #set par(spacing: 0pt)

      #text(weight: "semibold", fill: accent)[100 hydraulic base scenarios] \
      #text(size: label-size)[Hanoi network / demand patterns / uncertainties] \
      #v(1em)
      #plot-label[80 training / 10 validation / 10 test]
    ]
  ]
  #layout(size => {
    let left-x = (size.width - diagram-branch-gap) / 4
    diagram-fork(
      size.width,
      size.width / 2,
      left-x,
      size.width - left-x,
    )
  })
  #grid(
    columns: (1fr, 1fr),
    column-gutter: diagram-branch-gap,
    branch-cell(first: true, text(weight: "semibold", fill: accent)[Precomputed offline dataset]),
    branch-cell(first: true, text(weight: "semibold", fill: accent)[Online interaction data]),

    branch-cell(context {
      let labels = grid(
        columns: 1,
        row-gutter: 4pt,
        align: right,
        [Training:],
        [Validation:],
        [Test:],
      )

      grid(
        columns: (auto, auto, auto),
        column-gutter: 3pt,
        align: (right + horizon, center + horizon, left + horizon),
        labels,
        move(math.lr(size: measure(labels).height + 2pt, sym.brace.r), dy: 0.5pt),
        [5 fixed arsenic schedules per scenario],
      )
    }),
    branch-cell(grid(
      columns: 2,
      column-gutter: 2pt,
      row-gutter: 4pt,
      align: (right, left),
      [Training:], [new arsenic schedule per episode],
      [Validation:], [5 fixed schedules per scenario],
      [Test:], [10 fixed schedules per scenario],
    )),

    branch-label(fill: ink)[5 prescribed chlorine profiles], branch-label[Chlorine input],

    diagram-cell(pairing), diagram-cell(interaction),

    branch-cell(last: true)[Simulated in advance], branch-cell(last: true)[Simulated on demand],

    diagram-arrow(height: 1.85em), diagram-arrow(height: 1.85em),

    branch-cell(first: true, last: true)[
      #text(fill: accent, weight: "semibold")[2500 stored trajectories] \
      25 per base scenario
    ],
    branch-cell(first: true, last: true)[
      #text(fill: accent, weight: "semibold")[Interaction data generated on demand] \
      Chlorine inputs and resulting concentrations
    ],
  )
]
