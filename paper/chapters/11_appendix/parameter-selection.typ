#import "../../numbers.typ": *
#import "../../utils.typ": chapter-ref
#import "@preview/acrostiche:0.7.0": acr
#import "@preview/akatable:0.1.0": academic-table

= Policy and Reward Parameter Selection <section:appendix:parameter-selection>

This appendix documents the search summarized in @section:methodology:policy-hyperparameter-optimization and the subsequent reward-parameter comparison. All candidates used ordinarily initialized recurrent policies and training seed 42, with the architecture fixed throughout the search. The search first varied policy-training parameters while keeping the reward fixed. It then varied selected reward parameters with the chosen #acr("PPO") configuration held fixed.

Each candidate was evaluated on the same 50 validation cases, comprising five fixed arsenic schedules for each of ten validation scenarios. Validation took place at intervals of approximately #num[50000] training interactions. The budgets reported below are nominal totals across the four parallel environments. Because runs completed whole rollouts, their actual interaction counts could slightly exceed these budgets. Longer runs started from ordinary initialization rather than continuing the shorter screening checkpoints. The test scenarios were excluded from parameter selection.

== Policy-Training Parameters

The initial reference used a learning rate of #sci("1e-4"), 1024 rollout steps per environment, a minibatch size of 512, and an entropy coefficient of zero. The discount factor was $gamma = 0.99$ and the #acr("GAE") parameter was $lambda = 0.95$. The remaining settings were held at the values in @tab:methodology-recurrent-ppo-configuration. The first stage compared this reference with eight single-parameter variants, each trained for #num[500000] interactions. @tab:appendix:ppo-screening lists the alternatives, with all unlisted parameters unchanged.

The second stage compared selected settings and targeted combinations using the budgets in @tab:appendix:ppo-follow-up. It included a reference run at one million interactions so that the longer candidates could be assessed at the same training budget.

After selecting 2048 rollout steps per environment and a minibatch size of 512, the third stage compared all six combinations of $gamma in {0.98, 0.99, 0.995}$ and $lambda in {0.95, 0.98}$ at a budget of one million interactions. The five new combinations were compared with the existing run using $gamma = 0.99$ and $lambda = 0.95$. The final configuration kept these discount and #acr("GAE") parameters together with the learning rate of #sci("1e-4") and zero entropy coefficient.

#{
  set par(justify: false)
  set figure(placement: none)

  academic-table(
    [Single-parameter alternatives in the initial #acr("PPO") screening.],
    header: ([*Parameter*], [*Reference*], [*Alternatives*]),
    (
      [Learning rate],
      [#sci("1e-4")],
      [#sci("5e-5"), #sci("3e-4")],
      [Rollout steps per environment],
      [1024],
      [512, 2048],
      [Minibatch size],
      [512],
      [256, 1024],
      [Entropy coefficient],
      [0],
      [#sci("1e-3"), #sci("1e-2")],
    ),
    format: "booktabs",
    columns: (1.4fr, 1fr, 1fr),
    align: (left, center, center),
    label: <tab:appendix:ppo-screening>,
  )
}

#{
  set par(justify: false)
  set figure(placement: none)

  academic-table(
    [Follow-up #acr("PPO") runs. Interaction counts refer to training. Changes are relative to the initial reference in @tab:appendix:ppo-screening.],
    header: ([*Configuration change*], [*Nominal interactions*]),
    (
      [None],
      [#num[1000000]],
      [2048 rollout steps per environment],
      [#num[1000000]],
      [Minibatch size 256],
      [#num[1000000]],
      [2048 rollout steps and minibatch size 256],
      [#num[1000000]],
      [Learning rate #sci("3e-4")],
      [#num[1000000]],
      [2048 rollout steps and learning rate #sci("3e-4")],
      [#num[500000]],
    ),
    format: "booktabs",
    columns: (1fr, auto),
    align: (left, right),
    label: <tab:appendix:ppo-follow-up>,
  )
}

== Reward Parameters

The reward comparison started from the parameters in @tab:methodology-reward-parameters and used the selected #acr("PPO") configuration in @tab:methodology-recurrent-ppo-configuration. Four additional candidates each changed one reward parameter, as shown in @tab:appendix:reward-search. Each was trained for one million interactions and compared with the existing run at the same budget using the reference reward. The target, concentration thresholds, reference scales, movement and action-change weights, and startup timings remained fixed.

The comparison supported keeping the four-step hold, twelve-step release, and minimum upper-tracking factor of #sci("1e-4") as a compromise between contamination response and normal-operation control.

#{
  set par(justify: false)
  set figure(placement: none)

  academic-table(
    [Reward-parameter alternatives, each applied separately. Each control step corresponds to #minute[30].],
    header: ([*Parameter*], [*Reference*], [*Alternatives*]),
    (
      [Hold period],
      [4 steps / #hour[2]],
      [8 steps / #hour[4]],
      [Release duration],
      [12 steps / #hour[6]],
      [6 steps / #hour[3]],
      [Minimum upper-tracking factor],
      [#sci("1e-4")],
      [#sci("1e-3"), #sci("1e-5")],
    ),
    format: "booktabs",
    columns: (1.4fr, 1fr, 1fr),
    align: (left, center, center),
    label: <tab:appendix:reward-search>,
  )
}


== Selection Criteria and Supporting Diagnostic

Selection considered the validation learning curves, final performance, and averages over the last three evaluations. Validation return was assessed alongside physical control performance, including arsenic burden and chlorine excess during normal operation as defined in #chapter-ref(<chapter:evaluation>). The reward variants were compared using physical control performance because their returns correspond to different reward weightings.

An additional diagnostic measured chlorine shortfalls during reference-event operation. Using the notation and reference windows in @section:evaluation:reference-windows, it is defined for each validation case as

$
  f_("low")^"E" = frac(
    sum_(t in cal(T)_"M") sum_(v=1)^N I^"E" (t) thin bold(1)[c_(v,t) < L],
    N sum_(t in cal(T)_"M") I^"E" (t),
  ).
$

Here, $L$ = #mg-l[0.2] is the lower chlorine bound, and $cal(T)_"M"$ contains the timesteps of the main evaluation window. The diagnostic is the fraction of monitored node--time observations below this threshold during reference-event periods. Values equal to the threshold do not count as shortfalls. A case without eligible observations is recorded as not applicable. Applicable case values are averaged first over schedules within each scenario and then over scenarios. The clearance tail is excluded. This supporting diagnostic differs from the primary chlorine-shortfall metric in @eq:evaluation-normal-chlorine-shortfall, which measures shortfalls during reference-normal operation.

These comparisons guided the choice of practical settings for the principal experiment. The limited candidate sets and single training seed do not establish a global optimum or robustness across training seeds.
