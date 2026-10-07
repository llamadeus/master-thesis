#let cite-author(key) = {
  cite(key, form: "author", style: "chapters/10_references/short-authors.csl")
}

#let cite-prose(key) = {
  cite(key, form: "prose", style: "chapters/10_references/short-authors.csl")
}

#let chapter-ref(target) = context {
  let chapter = query(target).first()
  link(chapter.location(), chapter.body)
}


#let continued-header(
  columns,
  gap: 1.5em,
  top-stroke: 1.5pt,
) = (
  table.hline(stroke: none),

  table.cell(
    colspan: columns,
    inset: 0pt,
    align: left,
    context {
      let current-table = query(
        figure.where(kind: table).before(here()),
      ).last()

      let origin = current-table.location()

      if here().page() > origin.page() {
        let table-number = counter(figure.where(kind: table)).at(origin).first()

        pad(bottom: gap)[
          Table #table-number (continued)
        ]
      }
    },
  ),

  table.hline(stroke: top-stroke),
)

#let continued-tables(columns) = body => {
  set figure(placement: none)
  show figure.where(kind: table): set block(breakable: true)
  set table.cell(breakable: false)
  show figure.where(kind: table): fig => {
    show figure.caption: cap => block(
      above: 0pt,
      below: 0pt,
      sticky: true,
      breakable: false,
      cap,
    )
    fig
  }

  show table: it => {
    let fields = it.fields()
    let children = fields.remove("children")

    if children.len() == 0 {
      return it
    }

    let head = children.first()

    if head.children.len() == 0 or head.func() != table.header {
      return it
    }

    if head.children.first() == table.hline(stroke: none) {
      return it
    }

    let header-fields = head.fields()
    let header-children = header-fields.remove("children")

    table(
      ..fields,
      table.header(
        ..header-fields,
        ..continued-header(columns),
        ..header-children,
      ),
      ..children.slice(1),
    )
  }

  body
}
