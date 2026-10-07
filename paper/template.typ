
#let navigation-header(chapter-title, section-title) = {
  set text(size: 10pt, fill: luma(80))
  set par(justify: false)
  grid(
    columns: (auto, 1fr),
    column-gutter: 1.5em,
    align: (left, right),
    chapter-title, section-title,
  )
  line(length: 100%, stroke: 0.4pt + luma(180))
}

#let running-header() = context {
  let current-page = here().page()
  let chapter-title = []
  let section-title = []
  let in-appendix = false

  // Carry headings onto continuation pages; use the latest headings on a
  // page with a chapter or section transition.
  for item in query(heading).filter(item => (
    item.level <= 2 and item.location().page() <= current-page
  )) {
    let title = if item.numbering == none {
      item.body
    } else {
      let prefix = "Chapter"
      if in-appendix {
        prefix = "Appendix"
      }
      if item.level > 1 {
        prefix = "Section"
      }

      [#prefix #numbering(item.numbering, ..counter(heading).at(item.location())) -- #item.body]
    }

    if item.has("label") and item.label == <chapter:appendix> {
      in-appendix = true
      chapter-title = title
      section-title = []
    }

    if item.level == 1 {
      chapter-title = title
      section-title = []
    } else {
      section-title = title
    }
  }

  navigation-header(chapter-title, section-title)
}
