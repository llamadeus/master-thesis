
#set heading(numbering: "A.1", supplement: [Appendix])
#show heading: set heading(
  supplement: [Appendix],
)
#counter(heading).update(0)
#heading(
  level: 1,
  numbering: none,
  outlined: false,
)[Appendix] <chapter:appendix>

#include "dataset-reproduction-artifacts.typ"
#include "simulation-configuration.typ"
#include "network-delay-diagnostic.typ"
#include "parameter-selection.typ"
#include "numerical-results.typ"
#include "submission-repository.typ"
