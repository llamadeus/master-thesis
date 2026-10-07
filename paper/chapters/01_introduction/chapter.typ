#import "@preview/acrostiche:0.7.0": acr

= Introduction <chapter:introduction>

Safe drinking water is essential for human health and daily life @who2026guidelines. Drinking water safety depends on more than the quality of the water at its source. Risks must be managed throughout the entire supply chain, from the water source through the treatment plant to storage, distribution, and use by consumers @who2026guidelines. As water moves through the distribution network, it may be exposed to various microbial and chemical contaminants @who2026guidelines.

Protecting drinking-water quality involves preventing contamination, monitoring water quality, and responding to contamination events @who2026guidelines. Prevention seeks to stop contamination from entering the supply and to preserve water quality during distribution. Maintaining a disinfectant residual is one established measure for limiting microbial growth within a network @who2026guidelines. Monitoring then provides evidence of changes in water quality @who2026guidelines. It may combine conventional measurements such as laboratory tests @who2026guidelines with biological early-warning approaches. Freshwater mussels, for example, have been proposed as sentinels for drinking-water safety @ferreirarodriguez2023mussels and have been used to monitor raw water before treatment at facilities in Warsaw and Skawina, Poland @mpwik2019mussels @zwikskawina2023mussels. If contamination is suspected or confirmed, utilities must react to limit its spread and consequences @who2023watersafetyplan. Depending on the incident, available responses include increasing the disinfectant dose for microbial contamination, flushing the system, diverting flows, and isolating affected areas @who2026guidelines @usepa2018contaminationresponse.

Maintaining a chlorine residual is a well-established measure for preserving water quality after treatment, as it can provide partial protection against low-level microbial contamination while water travels through the distribution network @who2026guidelines. The residual that reaches different locations, however, varies with conditions in the network and changes over time @rossman2020epanet. A fixed dose at the point of injection therefore cannot be expected to maintain the same residual concentration throughout the system at all times. When residual concentrations can be monitored, chlorination can respond to variation through feedback @wang2021waterqualitympc. It thereby becomes a control problem in which dosing decisions must be adapted to the observed condition of the network.


== Motivation

Adaptive chlorination must balance several objectives. Too little residual chlorine can leave parts of a distribution network without the intended protection @who2026guidelines, while excessive chlorination can affect the taste of the water @who2026guidelines and increase chlorination costs @wang2021waterqualitympc. Finally, when contamination can be addressed through chlorination, a controller must respond quickly and effectively, and find a compromise between effective treatment and excessively chlorinating unaffected parts of the network @usepa2018contaminationresponse. A useful controller must therefore preserve residual protection while avoiding unnecessary dosing and still retain the ability to respond appropriately when contamination occurs.

This balance becomes a difficult task as the transport of water and dissolved substances depends on changing flow conditions in the network @rossman2020epanet. Changes in demand patterns affect the path that water takes, where it mixes, and how long it remains in the network @rossman2020epanet @wang2021waterqualitympc. During transport, chlorine reacts with substances in the water and biofilm on pipe walls, so longer travel times can reduce the residual that reaches downstream locations @rossman2020epanet. A dosing change at the source may therefore affect monitored locations only after a delay, and its effect can vary across the network. Control decisions must account for consequences that are both spatially distributed and delayed @wang2021waterqualitympc @vrachimis2024delay.

The combined effects of changing demand, flow conditions, local chlorine consumption, contamination, and preceding transport determine how chlorine concentrations develop across a distribution network @rossman2020epanet @shang2023epanetmsx @wang2021waterqualitympc. When the available measurements do not uniquely determine the underlying network state, chlorination control becomes a partially observable problem @kaelbling1998pomdp.

#acr("RL", cap: true) offers a way to develop a chlorination controller that learns from the consequences of its actions @sutton2018reinforcement. Direct training from scratch, however, requires repeated interaction with the environment before useful control behavior can emerge @chua2018pets @nagabandi2018dynamics @nair2018demonstrations. At the same time, predictive models and model-based control methods can provide prior information about the consequences of dosing decisions @chua2018pets @nagabandi2018dynamics @wang2021waterqualitympc.

Predictive control and #acr("RL") have both been investigated for drinking-water chlorination, including data-enabled predictive control and recurrent policy learning @perelman2025deepc @abzanov2025chlorine. Learned surrogate models have also been used to support the optimization of neural chlorination controllers @monsia2026neuroevolution. In other control domains, imitation of #acr("MPC") policies has provided an initialization for subsequent model-free #acr("RL") @nagabandi2018dynamics. These findings motivate investigating whether offline #acr("MPC") supervision can improve learning in partially observable chlorination control. This study therefore examines how #acr("MPC")-guided pretraining affects the online sample efficiency and final physical control performance of a recurrent policy under a fixed interaction budget. The additional computational cost of pretraining is assessed separately.


== Research Question

Based on the preceding motivation, this thesis addresses the following research question:

#block(width: 100%, above: 1.5em, below: 2em, block(
  stroke: (left: (paint: rgb("356A8A"), thickness: 2pt, cap: "round")),
  inset: (left: 12pt),
  outset: (left: -1pt, top: 4pt, bottom: 4pt),
)[
  *How does #acr("MPC")-guided pretraining affect the online sample efficiency and final physical control performance of a chlorination policy trained with RecurrentPPO compared with training from scratch?*
])

Online sample efficiency is assessed through learning curves over a common online interaction budget. This examines whether pretraining allows the policy to reach comparable performance with fewer online interactions. Final control performance is assessed on separate test cases using the policy selected through validation after completion of the common training budget. The comparison distinguishes an early learning advantage from better final control, as the baseline may catch up within the shared budget. Offline preparation and computational cost are accounted for separately, since fewer online interactions do not necessarily imply lower total training effort.


== Contributions and Study Scope

This thesis develops a pretraining pipeline that transfers predictive control decisions into a recurrent chlorination policy. A learned model of chlorine dynamics supports an offline #acr("MPC") teacher, whose demonstrations are used to initialize the policy through behavior cloning. RecurrentPPO then fine-tunes the policy through simulator interaction. The resulting controller uses the recurrent policy alone, with forecasting and #acr("MPC") used only during offline preparation. During training and deployment, the policy receives only measured chlorine concentrations and information derived from the controller's own dosing actions and time records.

The empirical contribution is a comparison with RecurrentPPO trained from scratch under the same online training conditions and interaction budget. It examines learning progress and final control quality alongside the additional preparation cost. The study is limited to adaptive chlorination in the simulated Hanoi drinking-water distribution network. The modeled reaction oxidizes #acr("As(III)") but does not establish removal of total arsenic or drinking-water safety. Its conclusions apply to the examined demand and contamination scenarios and training budgets.


== Thesis Structure

@chapter:related-work places the study within existing research on chlorination control and policy pretraining. @chapter:foundations introduces the simulation and learning concepts needed for the study, and @chapter:dataset describes the Hanoi network, hydraulic and contamination scenarios, and data partitions. Building on this setting, @chapter:methodology defines the control problem and develops the baseline and pretraining pipeline. @chapter:evaluation establishes how their learning progress, physical control performance, and costs are compared.

@chapter:results reports these measurements, and @chapter:discussion interprets them in relation to the research question and the study's limitations. @chapter:conclusion summarizes the answer and identifies directions for further work. The appendices provide dataset reproduction details, simulation settings, network-delay diagnostics, parameter-selection details, and supplementary numerical results.
