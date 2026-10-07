#import "@preview/cetz:0.5.2"
#import "../../figure-utils.typ": *

// EPANET node coordinates for the Hanoi network.
#let nodes = (
  (1, 5251.17, 4485.98),
  (2, 5251.17, 5268.69),
  (3, 5227.80, 5969.63),
  (4, 6068.93, 5969.63),
  (5, 6676.40, 5969.63),
  (6, 7260.51, 5969.63),
  (7, 7260.51, 6600.47),
  (8, 7260.51, 7301.40),
  (9, 7260.51, 7990.65),
  (10, 6793.22, 7990.65),
  (11, 6793.22, 8551.40),
  (12, 6793.22, 8960.28),
  (13, 6150.70, 8960.28),
  (14, 6302.57, 7990.65),
  (15, 5683.41, 7990.65),
  (16, 5216.12, 7990.65),
  (17, 5216.12, 7535.05),
  (18, 5227.80, 7137.85),
  (19, 5227.80, 6542.06),
  (20, 4550.23, 5969.63),
  (21, 4550.23, 5303.74),
  (22, 4550.23, 4591.12),
  (23, 3977.80, 5969.63),
  (24, 3977.80, 7172.90),
  (25, 3954.44, 7990.65),
  (26, 4410.05, 7990.65),
  (27, 4818.93, 7990.65),
  (28, 3405.37, 5969.63),
  (29, 2716.12, 6004.67),
  (30, 2716.12, 6904.21),
  (31, 2716.12, 7990.65),
  (32, 3405.37, 7990.65),
)

// EPANET pipe IDs and their endpoint node IDs.
#let pipes = (
  (1, 1, 2),
  (2, 2, 3),
  (3, 3, 4),
  (4, 4, 5),
  (5, 5, 6),
  (6, 6, 7),
  (7, 7, 8),
  (8, 8, 9),
  (9, 9, 10),
  (10, 10, 11),
  (11, 11, 12),
  (12, 12, 13),
  (13, 10, 14),
  (14, 14, 15),
  (15, 15, 16),
  (16, 17, 16),
  (17, 17, 18),
  (18, 18, 19),
  (19, 19, 3),
  (20, 3, 20),
  (21, 20, 21),
  (22, 21, 22),
  (23, 20, 23),
  (24, 23, 24),
  (25, 24, 25),
  (26, 26, 25),
  (27, 27, 26),
  (28, 16, 27),
  (29, 23, 28),
  (30, 28, 29),
  (31, 29, 30),
  (32, 30, 31),
  (33, 32, 31),
  (34, 25, 32),
)

#let node-x-coordinates = nodes.map(node => node.at(1))
#let x-min = calc.min(..node-x-coordinates)
#let x-max = calc.max(..node-x-coordinates)
#let y-origin = 4400
#let y-scale = 470
#let node-position(id, x-span) = {
  let node = nodes.find(node => node.at(0) == id)
  (
    (node.at(1) - x-min) / (x-max - x-min) * x-span,
    (node.at(2) - y-origin) / y-scale,
  )
}

#let pipe-color = color.mix((white, 25%), (accent, 75%))

#let network = context layout(size => cetz.canvas(length: 12.5mm, padding: (x: 0.4pt, y: 2pt), {
  import cetz.draw: *
  let x-span = (size.width - 0.8pt) / 12.5mm - 2 * 0.22

  // Draw links first so that their endpoints disappear beneath the nodes.
  for pipe in pipes {
    line(
      node-position(pipe.at(1), x-span),
      node-position(pipe.at(2), x-span),
      stroke: (paint: pipe-color, thickness: line-size, cap: "round"),
    )
  }

  // Junctions 2--32 are numbered directly inside their node symbols.
  for node in nodes.filter(node => node.at(0) != 1) {
    let position = node-position(node.at(0), x-span)
    circle(
      position,
      radius: 0.22,
      fill: blue-tint,
      stroke: (paint: accent, thickness: 0.8pt),
    )
    content(
      position,
      text(size: label-size, weight: "semibold", fill: ink, str(node.at(0))),
    )
  }

  // Node 1 is the fixed-head reservoir and the network's chlorine source.
  let reservoir = node-position(1, x-span)
  let rx = reservoir.at(0)
  let ry = reservoir.at(1)
  line(
    (rx, ry + 0.28),
    (rx - 0.28, ry),
    (rx, ry - 0.28),
    (rx + 0.28, ry),
    close: true,
    fill: accent,
    stroke: (paint: accent, thickness: 0.8pt, join: "round"),
  )
  content(
    reservoir,
    text(size: label-size + 2pt, weight: "semibold", fill: white, [1]),
  )
}))

#block(width: 100%, stack(
  dir: ttb,
  spacing: panel-spacing,
  align(center, network),
  legend-spacing,
  legend-row(
    legend-item(rotate(45deg, square(size: 6pt, fill: accent, stroke: none)), [Reservoir and chlorine injection]),
    legend-item(circle(radius: 3pt, fill: blue-tint, stroke: 0.8pt + accent), [Junction]),
    legend-item(legend-line(pipe-color), [Pipe]),
  ),
))
