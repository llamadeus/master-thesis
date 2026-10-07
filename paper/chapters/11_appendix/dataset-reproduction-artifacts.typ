#import "@preview/acrostiche:0.7.0": acr
#import "../../numbers.typ": *
#import "../../utils.typ": chapter-ref


= Dataset Reproduction Artifacts <section:appendix:dataset-artifacts>

This appendix provides the additional information needed to reproduce the dataset described in the #chapter-ref(<chapter:dataset>) chapter.

For a base scenario with numeric index $i$, the demand-pattern seed is $1000 + i$. For network parameter uncertainties, a scenario-specific seed is derived deterministically from the master seed #identifier[42] and $i$ using the first 32-bit output of NumPy's `SeedSequence`.

For the offline dataset, fixed arsenic schedules use seeds #identifier[6103], #identifier[6104], and #identifier[6105] in the training, validation, and test partitions, respectively. Chlorine profiles use seeds #identifier[7103], #identifier[7104], and #identifier[7105] in the same order. Within each base scenario, an index determines the individual schedule or profile.

For online policy training, seed #identifier[8128] is combined with the environment index, the scenario identifier, and the number of earlier episodes generated for that scenario to generate arsenic schedules following @section:methodology:training-scenario-and-contamination-sampling. For the fixed control-validation and control-test catalogs, seeds #identifier[8129] and #identifier[8130], respectively, are combined with the base scenario and the schedule index within that scenario.


== Arsenic-Schedule Generation <section:appendix:arsenic-schedule-generation>

The scheduler generates the events in each schedule sequentially. For each event, it samples the duration uniformly from the half-hour grid between 2 and #hour[72], including both endpoints. It then samples a weekly start position uniformly from the 336 half-hour positions of a week.

For that start position, the scheduler selects uniformly among the eligible junction--week combinations described in @section:dataset:arsenic-events. If none is eligible, it samples another weekly position. This procedure is intended to give each weekday the same probability of an arsenic event starting. The probability of an event being active nevertheless differs across weekdays.

The feasibility of an injection rate is determined from the scenario's hydraulic outflow time series. For a candidate event, let $Q_"min"$ and $Q_"max"$ be the minimum and maximum positive outflow at its source junction, in #l-min(). Let $a_"min"$ = #mg-l[0.1] be the lower concentration limit, $a_"max"$ = #mg-l[10] the upper concentration limit, and $dot(m)_("As,max")$ = #mg-min[50000] the per-event mass-rate limit. The lowest and highest feasible targets for the maximum immediate local #acr("As(III)") contribution are

$
  a_("peak,min") & = a_"min" thin Q_"max" / Q_"min", \
  a_("peak,max") & = min(a_"max", dot(m)_("As,max") / Q_"min").
$

A candidate is rejected when $a_("peak,min") > a_("peak,max")$. Otherwise, the target $a^*_"peak"$ is sampled log-uniformly between $a_("peak,min")$ and $a_("peak,max")$. Its constant arsenic mass rate is

$
  dot(m)_"As" = a^*_"peak" thin Q_"min".
$

This choice accounts for the change in outflow during the event while satisfying the per-event limits given in @section:dataset:arsenic-events.


== Chlorine-Profile Generation <section:appendix:chlorine-profile-generation>

The order of the five profile segments is sampled uniformly from their possible permutations. A sampling band is chosen independently for the initial rate and each segment. The bands 0--#num[200000], #num[200000]--#num[350000], and #num[350000]--#mg-min[500000] have selection probabilities of #percentage[70], #percentage[20], and #percentage[10], respectively. Values and signal parameters are sampled uniformly within the selected band or the ranges given below.

The constant segment maintains a single injection rate. The periodic segment is a clipped sinusoid with a period of 2 to 240 reporting steps, a phase from 0 to $2 pi$, a center within its band, and an amplitude up to half the band's width. The piecewise-constant segment samples a new rate every 12 steps. The ramp interpolates from the preceding segment's final value to a sampled endpoint. The random walk also starts at that final value and changes by up to one tenth of its band's width per step, with values clipped to the band.

At each segment boundary, a linear transition from the preceding value replaces the first quarter of the next segment with #pct[25] probability. A final limit on rate changes also applies between the initial zero-injection state and the first profile value.
