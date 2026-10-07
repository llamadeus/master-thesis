#import "data.typ": (
  learning-summaries, mean, sample-sd, selected-validation-checkpoints, t-critical-95-df4, training-seeds,
)

#let variant-learning-data = json("../../results/variant-learning-plot-data.json")
#assert(variant-learning-data.format_version == 1 and variant-learning-data.data_kind == "experimental")
#assert(variant-learning-data.rows.len() == 510)

#let variant-returns(approach, nominal) = training-seeds.map(seed => {
  let row = variant-learning-data.rows.find(row => (
    row.approach == approach and str(row.training_seed) == seed and row.nominal_interactions == nominal
  ))
  assert(row != none, message: "Variant checkpoint missing for a training seed")
  row.evaluation_return
})

#let variant-endpoint-summary(approach, nominal) = {
  let values = variant-returns(approach, nominal)
  let reference = learning-summaries.last().at("pretrained-values")
  let differences = values.zip(reference).map(((value, comparator)) => value - comparator)
  let difference-mean = mean(differences)
  let half-width = t-critical-95-df4 * sample-sd(differences) / calc.sqrt(training-seeds.len())
  (
    nominal: nominal,
    mean: mean(values),
    sd: sample-sd(values),
    differences: differences,
    difference-mean: difference-mean,
    ci-low: difference-mean - half-width,
    ci-high: difference-mean + half-width,
  )
}

#let extended-baseline-return = variant-endpoint-summary("baseline-4m", 4000000)
#let without-actor-return = variant-endpoint-summary("without-actor-weights", 1000000)
#let extended-baseline-first-crossing = range(0, 4000001, step: 50000).find(nominal => (
  mean(variant-returns("baseline-4m", nominal)) >= learning-summaries.last().at("pretrained-mean")
))
#let without-actor-first-crossing = range(0, 1000001, step: 50000).find(nominal => (
  mean(variant-returns("without-actor-weights", nominal)) >= learning-summaries.last().at("baseline-mean")
))

#let variant-runtime-summary(approach) = {
  let rows = training-seeds.map(seed => {
    let row = variant-learning-data.runtime_rows.find(row => (
      row.approach == approach and str(row.training_seed) == seed
    ))
    assert(row != none, message: "Variant runtime missing for a training seed")
    row
  })
  let values = rows.map(row => row.training_seconds)
  (
    mean: mean(values),
    sd: sample-sd(values),
    validation-mean: mean(rows.map(row => row.checkpoint_evaluation_seconds)),
  )
}

#let extended-baseline-training-cost = variant-runtime-summary("baseline-4m")
#let without-actor-training-cost = variant-runtime-summary("without-actor-weights")

#let pretrained-selected-returns = selected-validation-checkpoints.map(row => row.pretrained.evaluation_return)
#let pretrained-selected-return-mean = mean(pretrained-selected-returns)
#let pretrained-selected-return-sd = sample-sd(pretrained-selected-returns)

#let variant-selected-summary(approach) = {
  let checkpoints = training-seeds.map(seed => {
    let candidates = variant-learning-data
      .rows
      .filter(row => row.approach == approach and str(row.training_seed) == seed)
      .sorted(key: row => row.nominal_interactions)
    assert(candidates.len() > 0, message: "Variant checkpoints missing for a training seed")
    let highest-return = calc.max(..candidates.map(row => row.evaluation_return))
    candidates.find(row => row.evaluation_return == highest-return)
  })
  let values = checkpoints.map(row => row.evaluation_return)
  let differences = values.zip(pretrained-selected-returns).map(((value, comparator)) => value - comparator)
  let difference-mean = mean(differences)
  let half-width = t-critical-95-df4 * sample-sd(differences) / calc.sqrt(training-seeds.len())
  (
    mean: mean(values),
    nominal-checkpoints: checkpoints.map(row => row.nominal_interactions),
    sd: sample-sd(values),
    differences: differences,
    difference-mean: difference-mean,
    ci-low: difference-mean - half-width,
    ci-high: difference-mean + half-width,
  )
}

#let extended-baseline-selected-return = variant-selected-summary("baseline-4m")
#let without-actor-selected-return = variant-selected-summary("without-actor-weights")
