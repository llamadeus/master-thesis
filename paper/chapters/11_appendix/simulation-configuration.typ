#import "@preview/acrostiche:0.7.0": acr
#import "../../numbers.typ": *
= Simulation Configuration <section:appendix:simulation-configuration>

This appendix specifies the hydraulic settings and reaction equations used for the simulations described in @section:dataset:simulation-setup. It also defines the local source correction used to calculate transported #acr("As(III)") for the evaluation.

EPyT-Flow's LeakDB loader sets pressure-driven demand for the Hanoi scenarios, with minimum and required pressures of #meter[0] and #meter[0.1] and a pressure exponent of #num[0.5]. The loaded Hanoi network uses the Hazen--Williams head-loss relation @artelt2026epytflow017. These settings are used unchanged in the scenario files for this study. The required pressure is a synthetic study setting.

Let $c$ and $a$ denote the chlorine and #acr("As(III)") concentrations in #mg-l(). The pipe reaction rates are

$
  r_"Cl" & = -k_"B" thin c - k_2 thin (c thin a) / M_"As",
           wide r_"As" & = -k_2 thin (c thin a) / M_"Cl".
$ <eq:appendix-pipe-reaction-rates>

The first-order chlorine-decay coefficient is $k_"B"$ = #per-day("0.3", unit: (fraction: "power")), and the chlorine--#acr("As(III)") reaction coefficient is $k_2$ = #l-mmol-day[100000]. The factor $M_"Cl"$ = #mg-mmol[70.9] converts the chlorine concentration from its $"Cl"_2$-equivalent mass basis to a molar concentration. For #acr("As(III)"), reported as arsenic mass, the corresponding factor is $M_"As"$ = #mg-mmol[74.92]. These factors make the coupled reaction terms consume equal molar amounts of chlorine and #acr("As(III)"). The rate coefficients are synthetic assumptions rather than values calibrated to measured concentrations.


== Local Source Correction for Transported Arsenic <section:appendix:transported-arsenic>

The correction described in @section:evaluation:reference-windows starts with the reported #acr("As(III)") concentration $a_(v,t)^"raw"$ at monitored node $v$ and reporting step $t$. Let $dot(m)_("As",v) (t)$ be the local injection rate in #mg-min(). The nodal outflow is

$ Q_v (t) = max lr((0, d_v (t)), size: #125%) + sum_(e in cal(E)_v) max lr((0, q_(v,e) (t)), size: #125%). $

Here $d_v (t)$ is nodal demand, $cal(E)_v$ is the set of connected links, and $q_(v,e) (t)$ is positive for flow leaving the node. For the model's flow unit of cubic meters per hour, $kappa = 1000/60$ converts $Q_v$ to #l-min().

Because concentrations are reported at the end of each interval $Delta t$, the estimated local contribution at step $t$ uses the injection rate and outflow at the preceding step $t - 1$.

$
  hat(a)_(v,t)^"local" = cases(
    frac(dot(m)_("As",v) (t - 1), kappa thin Q_v (t - 1)) quad & "if" Q_v (t - 1) > delta_Q,
    0 & "otherwise",
  ).
$

Here $delta_Q = 10^(-9)$ in the hydraulic flow unit separates positive outflow from numerical zero. The initial sample at $t = 0$ has no preceding interval, so its estimated local contribution is zero.

An exception applies to a complete, contiguous source interval if no other monitored source is active and no other monitored node has detectable #acr("As(III)") at any time during that interval. In this case, the raw source-node concentration is treated as entirely local. Source activity and #acr("As(III)") detection use thresholds of #sci-unit("1e-9", mg-min) and #sci-unit("1e-9", mg-l), respectively. In all other cases, the hydraulic estimate applies. With $a_(v,t)^"local"$ denoting the selected local contribution, the transported concentration is

$ a_(v,t) = max(0, a_(v,t)^"raw" - a_(v,t)^"local"). $
