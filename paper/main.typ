#import "@preview/acrostiche:0.7.0": *
#import "@preview/zero:0.7.0" as zero
#import "template.typ": *

#zero.set-num(math: false)
#zero.set-group(separator: sym.wj + sym.space.thin + sym.wj, threshold: (integer: 5, fractional: calc.inf))
#zero.set-unit(fraction: "inline")

#set page(numbering: none)
#set heading(numbering: "1.1")
#set par(justify: true)
#set text(lang: "en")
#set figure(placement: auto)
#set figure(boundary: "subsection") if sys.inputs.at("figure-boundary", default: "false") == "true"
#set math.equation(numbering: "(1)")

#let chapter(path) = {
  include path
}

#show heading: set block(above: 16pt, below: 11pt)
#show heading.where(level: 1): set heading(
  supplement: [Chapter],
)
#show heading.where(level: 1): it => {
  place.flush()
  it
}

#show figure: set block(above: 1.5em, below: 2em)
#show figure: set place(clearance: 2em)
#show figure.caption: set align(left)
#show figure.caption: set par(justify: false, linebreaks: "optimized")
#show figure.caption: set text(hyphenate: true)
#set figure(gap: 1.5em)

#init-acronyms((
  "As(III)": "arsenite",
  "BPTT": "backpropagation through time",
  "CEM": "cross-entropy method",
  "CI": "confidence interval",
  "DAgger": "dataset aggregation",
  "EPANET-MSX": "EPANET Multi-Species Extension",
  "GAE": "generalized advantage estimation",
  "KL": ([Kullback--Leibler divergence],),
  "LSTM": "long short-term memory",
  "MAE": "mean absolute error",
  "MDP": "Markov decision process",
  "MPC": "model predictive control",
  "MSE": "mean squared error",
  "PETS": "probabilistic ensembles with trajectory sampling",
  "POMDP": "partially observable Markov decision process",
  "PPO": "proximal policy optimization",
  "RL": "reinforcement learning",
  "RMSE": "root mean squared error",
  "RNN": "recurrent neural network",
  "SD": "standard deviation",
  "WHO": "World Health Organization",
))


#let paper-date = auto
#if sys.inputs.at("deterministic", default: "false") == "true" {
  paper-date = datetime(year: 2026, month: 10, day: 7)
}

#set document(
  author: ("Amadeus Beckmann",),
  title: [Pretraining for Efficient Reinforcement Learning-Based Chlorination Control in Drinking Water Distribution Systems],
  date: paper-date,
)

#text(size: 14pt)[
  #align(center)[
    #v(1cm)
    #text(size: 18pt, weight: "bold")[Universität Bielefeld]

    #text(size: 15pt)[Technische Fakultät]

    #v(3cm)
    #text(size: 16pt, weight: "bold")[Masterarbeit] \
    #text(size: 14pt)[
      Intelligent Interactive Systems / Master of Science
    ]

    #v(2cm)
    #text(size: 16pt, weight: "bold")[
      Pretraining for Efficient Reinforcement Learning-Based \
      Chlorination Control in Drinking Water Distribution Systems
    ]

    #v(0.5cm)
    vorgelegt von \
    #text(weight: "bold")[Amadeus Beckmann]

    #v(1fr)
    #grid(
      columns: (auto, auto),
      column-gutter: 0.35em,
      row-gutter: 0.65em,
      align: (right, left),
      "1. Gutachter:", "Dr. André Artelt",
      "2. Gutachter:", "Paul J. Stahlhofen, M. Sc.",
    )

    #v(1cm)
    #text(size: 14pt)[
      Bielefeld, 7. Oktober 2026
    ]
    #v(1.5cm)
  ]
]

#pagebreak(weak: true)
#set page(numbering: "i")
#include "chapters/00_abstract/chapter.typ"

#pagebreak(weak: true)
#show outline.entry.where(level: 1): it => {
  v(12pt, weak: true)
  text(it, weight: "bold")
}
#outline()

#pagebreak(weak: true)
#set page(header: navigation-header([Acronyms], []))
#print-index(title: "Acronyms", sorted: "up", numbering: none, outlined: false, row-gutter: 8pt)

#pagebreak(weak: true)
#set page(numbering: "1", header: running-header())
#counter(page).update(1)

#chapter("chapters/01_introduction/chapter.typ")
#chapter("chapters/02_related_work/chapter.typ")
#chapter("chapters/03_foundations/chapter.typ")
#chapter("chapters/04_dataset/chapter.typ")
#chapter("chapters/05_methodology/chapter.typ")
#chapter("chapters/06_evaluation/chapter.typ")
#chapter("chapters/07_results/chapter.typ")
#chapter("chapters/08_discussion/chapter.typ")
#chapter("chapters/09_conclusion/chapter.typ")
#chapter("chapters/10_references/chapter.typ")

#pagebreak(weak: true)
#chapter("chapters/11_appendix/chapter.typ")

#pagebreak(weak: true)
#set page(numbering: none, header: none)
#chapter("chapters/12_authorship/chapter.typ")
