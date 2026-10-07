#import "@preview/acrostiche:0.7.0": acrfull, acrfullcap, display-def, display-short
#import "../../numbers.typ": *
#import "../07_results/data.typ": (
  baseline-total-seconds, duration-text, fixed, fixed-pct, learning-summaries, metric-summary, pretrained-total-seconds,
)

#let endpoint-crossing = learning-summaries.find(row => (
  row.at("pretrained-mean") >= learning-summaries.last().at("baseline-mean")
))
#let burden = metric-summary("arsenic_burden")
#let clearance = metric-summary("arsenic_clearance_lead")
#let excess = metric-summary("normal_chlorine_excess_fraction")
#let chlorine-use = metric-summary("chlorine_use")

// Keep abstract expansions independent of first uses in the main text.
#align(center)[
  #block(width: 90%)[
    *Abstract*

    #align(left)[
      Adaptive chlorination in drinking-water distribution networks must maintain suitable residual concentrations while responding to contamination. Delayed transport and changing demand complicate control when measurements do not reveal the complete network state. #acrfullcap("RL") offers a way to learn feedback control, but training from scratch requires repeated environment interaction. This thesis examines whether #acrfull("MPC")-guided pretraining improves the online sample efficiency and final physical control performance of a recurrent chlorination policy.

      A multistep #acrfull("LSTM") forecaster supports an offline #display-short("MPC") teacher. Five-fold scenario-wise cross-fitting generates teacher actions with forecasters trained on other hydraulic scenarios. Behavior cloning uses these actions to initialize a recurrent actor, which is then fine-tuned with recurrent #display-def("PPO") (Recurrent#display-short("PPO")). Forecasting and #display-short("MPC") are confined to offline preparation. The deployed policy receives only chlorine measurements, its own dosing, and time records. The comparison uses the simulated Hanoi network with 31 monitored demand nodes, varying hydraulic conditions, and #acrfull("As(III)") contamination. Five matched pairs of training runs compare this initialization with training from scratch under a common nominal budget of one million online interactions.

      The pretrained approach achieved higher mean validation return at every evaluated checkpoint and first reached the baseline's one-million-interaction endpoint return at #num(endpoint-crossing.nominal) nominal interactions. After both approaches completed the common budget, policies selected by validation return were evaluated on 100 separate test cases. The selected pretrained policies had #fixed-pct(100 * (1 - burden.at("pretrained-mean") / burden.at("baseline-mean")), digits: 1) lower mean transported #display-short("As(III)") burden and reached persistent clearance #duration-text(clearance.at("difference-mean") * 3600) earlier on average in events where both policies exceeded the clearance threshold. Both differences favored pretraining in all five seed pairs. Chlorine use increased by #fixed-pct(100 * (chlorine-use.at("pretrained-mean") / chlorine-use.at("baseline-mean") - 1), digits: 1), and the normal-operation chlorine excess fraction increased by #fixed(100 * excess.at("difference-mean"), digits: 3) percentage points. An exploratory analysis located almost all additional excess within #hour[12] after clearance in the zero-chlorine reference runs. The shortfall difference remained uncertain, and neither approach exceeded the #mg-l[5.0] event-time chlorine ceiling in applicable observations.

      With fresh preparation, the summed stage runtimes across all five seeds, including shared trajectory generation and control evaluation, were about #fixed(pretrained-total-seconds / baseline-total-seconds, digits: 1) times the baseline total. Pretraining also required #num[2.4] million offline simulation steps for training trajectories. The results support improved online sample efficiency and contamination treatment under the modeled #display-short("As(III)") oxidation reaction, at the cost of additional preparation and chlorine use.
    ]
  ]
]
