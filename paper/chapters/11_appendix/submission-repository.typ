#import "@preview/zebra:0.1.0": qrcode

#let repository-url = "https://github.com/llamadeus/master-thesis"

= Submission Repository <section:appendix:submission-repository>

The submission repository contains the final thesis, its sources, and the code for dataset generation, training, and evaluation. It also provides the paper's numerical inputs, supporting tools and tests, and reproduction instructions.

#v(1em)
#align(center)[
  #stack(
    dir: ttb,
    spacing: 1.5em,
    link(repository-url)[
      #qrcode(repository-url, width: 3.5cm, background-fill: white)
    ],
    link(repository-url)[#raw(repository-url)],
  )
]
