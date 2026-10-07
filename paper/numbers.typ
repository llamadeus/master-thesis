#import "@preview/zero:0.7.0" as zero

// Shared text-mode number and quantity functions. The document-wide Zero
// settings in main.typ also apply to these functions and table formatting.
#let num = zero.num
#let sci(value) = {
  let parts = value.split("e")
  assert(parts.len() == 2, message: "Expected scientific notation such as 1e-4")
  let exponent = int(parts.last())
  if exponent >= 0 {
    zero.num(value)
  } else {
    let power = zero.num[10] + super(typographic: false, [#sym.minus#zero.num(str(-exponent))])
    zero.num(parts.first()) + sym.space.thin + sym.wj + sym.times + sym.space.thin + sym.wj + power
  }
}
#let sci-unit(value, unit) = sci(value) + sym.space.thin + sym.wj + unit()
#let identifier(value) = zero.num(value, group: (threshold: calc.inf))
#let pct = zero.zi.percent
#let percentage = zero.zi.percent
#let meter = zero.zi.meter
#let milligram = zero.zi.declare("mg")
#let tonne = zero.zi.declare("t", alt: "tonnes")
#let minute = zero.zi.minute
#let hour = zero.zi.hour
#let unbreakable-unit(unit, alt: auto) = {
  let declared = zero.zi.declare(unit, alt: alt)
  (..args) => box(declared(..args))
}
#let mg-l = unbreakable-unit("mg/L")
#let mg-min = unbreakable-unit("mg/min")
#let g-min = unbreakable-unit("g/min")
#let l-min = unbreakable-unit("L/min")
#let mg-h-l = unbreakable-unit("mg h/L")
#let m3-h = unbreakable-unit([m#super[3]/h], alt: "cubic metres per hour")
#let per-day = unbreakable-unit("1/d")
#let l-mmol-day = unbreakable-unit("L/mmol/d")
#let mg-mmol = unbreakable-unit("mg/mmol")
