#import "../../numbers.typ": *
#import "@preview/acrostiche:0.7.0": acr
#import "@preview/akatable:0.1.0": academic-table
#import "@preview/zero:0.7.0": format-table, num
#import "../../utils.typ": chapter-ref, continued-tables


= Methodology <chapter:methodology>

This chapter describes the two training approaches compared in this study. Both policies use the same recurrent architecture and chlorination-control setting but differ in their initialization. The baseline approach trains a policy from an ordinary initialization. The pretrained approach initializes a policy from teacher demonstrations generated using a prediction model and then trains it online under the same conditions. We refer to this initialization procedure as #acr("MPC")-guided policy pretraining. The comparison assesses how this procedure affects online sample efficiency and final control performance under a common interaction budget.

Our pretraining pipeline first trains a multistep forecaster to predict the network's chlorine response to candidate injection sequences. #acr("MPC") uses these forecasts to evaluate the candidate sequences and select teacher actions. Behavior cloning trains the recurrent actor to reproduce these actions, and #acr("PPO") subsequently fine-tunes the initialized policy through interaction with the simulator. The forecaster and #acr("MPC") optimizer are used only during pretraining and are not part of #acr("PPO") fine-tuning, evaluation, or deployment.

In this chapter, we first define the chlorination-control problem, including its partial observability, operational objectives, and action constraints. We then describe the reward function and policy observations before presenting the shared recurrent policy-learning framework, the baseline approach, and the online training procedure. Next, we describe cross-fitted teacher generation, followed by behavior-cloned policy initialization and #acr("PPO") fine-tuning. Finally, we define the principal experimental comparison and document the implementation and reproducibility requirements. Both approaches use the Hanoi network, the generated hydraulic base scenarios, and arsenic schedules sampled using the procedure described in @section:dataset:arsenic-events. The prescribed chlorine injection profiles are used to generate the offline trajectories for pretraining.


== Chlorination-Control Problem

First, we establish the chlorination-control problem shared by both approaches. This includes defining the operational objectives and partial observability of the task and specifying the control action and actuator constraints.


=== Operational Objectives and Partial Observability <section:methodology:operational-objectives>

We begin by considering what the controller can influence within the testbed. Chlorine is injected at a single central source, whereas its concentration is monitored at multiple locations throughout the network. At every control step, the policy must therefore select one source action that maintains suitable chlorine concentrations across all monitored nodes while also responding to arsenic events that may originate at different junctions. This creates a network-wide control problem, since a decision taken at one location must account for responses that occur at different locations and at different times.

The controller does not, however, have direct access to the complete physical condition of the network. We consequently formulate the task as a #acr("POMDP"). Its latent state comprises the hydraulic and water-quality conditions, current demands, contamination events, uncertain physical properties, and the transport history that connects earlier conditions to the present network response. By contrast, the final deployed policy receives only the restricted observation defined later in this chapter.

This distinction is important because the controller's current observation does not uniquely identify the underlying state. Similar chlorine patterns can result from different demand conditions, arsenic events, pipe properties, or previous transport dynamics. A single measurement snapshot is therefore insufficient to determine which process produced the observed concentrations. A sequence of observations provides additional temporal evidence, which motivates the use of a recurrent policy that can combine current measurements with information retained from previous time steps.

With this information constraint in place, we can define the operational objectives for the chlorine controller. During normal operation, chlorine concentrations at all monitored nodes should remain within the range of #num[0.2]--#mg-l[0.4]. These bounds are adopted from the IJCAI #identifier[2025] AI for Drinking Water Chlorination Challenge @artelt2025chlorinationchallenge, which posed a similar control task on a larger network with wastewater contamination involving pathogens and organic matter. Its controller was required to respond to contamination events while maintaining node-level chlorine concentrations within this range. The same interval is used here to define the normal-operation objective. During arsenic events, however, the objective shifts to effective treatment through a rapid, but appropriate response. For this, the upper bound at monitored nodes is temporarily lifted from #num[0.4] to #mg-l[5.0] to allow for higher injections and effective treatment. This ceiling follows the health-based guideline value for free chlorine in drinking water published by the #acr("WHO") @who2026guidelines. Within this enlarged operating range, the controller should maintain or restore sufficient chlorine residuals at affected nodes while limiting the controllable #acr("As(III)") burden. The policy-independent periods over which normal operation and contamination response are evaluated are defined in @section:evaluation:reference-windows.

Maintaining a positive chlorine residual also supports both arsenic treatment and indirect event detection. It provides free chlorine that can react with #acr("As(III)") after contaminated water enters the pipe network. A sufficient chlorine residual can also make disturbances visible through changes in chlorine concentration to which the controller can respond. Such variations may indicate ongoing arsenic contamination, but they can also result from changing demand patterns and the corresponding change of flow through the network. The observed chlorine response therefore does not uniquely identify its underlying cause, reinforcing the partial observability of the control problem.


=== Control Action and Actuator Constraints <section:methodology:control-action-and-actuator-constraints>

As introduced in @section:dataset:hanoi-network, node 1 is the only chlorine-injection location, and the corresponding actuator is both bounded and rate-limited. We now formalize how the policy interacts with this actuator.

At each 30-minute control step $t$, the policy produces a scalar normalized action $a_t in [-1, 1]$ from the observation after step $t - 1$. The resulting observation is recorded after step $t$. This action does not specify an absolute injection rate. Instead, it determines the direction and magnitude of the change relative to the previous step's injection. An action of $1$ requests the maximum increase, an action of $-1$ requests the maximum decrease, and an action of $0$ leaves the injection rate unchanged.

Let $u_t$ denote the physical chlorine injection rate in #mg-min(). Let $Delta u_"max"$ = #mg-min[50000] denote the maximum permitted change per control step and $u_"max"$ = #mg-min[500000] the upper rate limit. The normalized action is scaled by $Delta u_"max"$ and added cumulatively to the previous rate. The resulting value is then clipped to the admissible interval:

$
  u_t = op("clip")(u_(t - 1) + Delta u_"max" thin a_t, 0, u_"max"), quad a_t in [-1, 1].
$ <eq:methodology-actuator-update>

Clipping the injection rate ensures that the actuator cannot request a negative injection rate or exceed the configured upper bound. When the previous rate is close to either bound, the realized change can therefore be smaller than the change represented by the normalized action. Each control episode starts with an injection rate $u_0$ of #mg-min[0], so the policy establishes dosing from the first control step. Both policy approaches use this same action representation, initial value, and actuator constraints, so their comparison is not affected by differences in how control actions are applied to the network.

Both limits are practical choices rather than empirically optimized actuator parameters. The upper bound keeps offline profiles, #acr("MPC") plans, and online policy training within the injection range used to generate the forecaster's training data. This alone does not ensure coverage of all resulting sequences or network states. The rate-change limit balances dosing precision against response speed: smaller values could delay contamination response, while larger values would amplify small variations in normalized policy output into larger physical dosing changes, potentially complicating fine adjustments and promoting oscillations.


== Reward Function <section:methodology:reward-function>

Both training approaches use the same reward design to maintain adequate chlorine levels throughout the network. The reward also discourages unnecessary concentration fluctuations and abrupt changes in the requested dosing increments. When chlorine levels fall, as can occur during contamination, the weighting of these objectives shifts to give greater priority to restoring sufficient residuals.

To express these objectives, the reward combines three penalty components: chlorine tracking, wasted chlorine movement, and changes in the normalized action. The reward is the negative of their sum,

$ r_t = -lr((P_("track",t) + P_("move",t) + P_("action",t)), size: #150%). $

Its maximum value is zero, and larger penalties produce more negative rewards. A continuous recovery factor $beta_t in [0, 1]$ adjusts the tracking and movement costs in response to low chlorine residuals. At $beta_t = 0$, these costs use their ordinary weighting, while $beta_t = 1$ favors recovery from low chlorine residuals.

Let $N$ be the number of monitored nodes, indexed by $v = 1, dots, N$. Let $c_(v,t)$ denote the chlorine concentration at monitored node $v$ after control step $t$. Let $c^*$ = #mg-l[0.30] denote the target concentration and $c_s$ = #mg-l[0.01] the reference concentration scale.


=== Chlorine Tracking

The tracking penalty $P_("track",t)$ averages the squared, scaled deviations of chlorine concentrations from the target $c^*$ across all monitored nodes. It is defined as

$
  P_("track",t) = L_t + f_t thin U_t.
$

Here, $L_t$ measures deviations below the target and $U_t$ measures deviations above it, while $f_t$ weights the upper contribution. Evaluating these contributions separately allows the pressure to restore a low residual to be maintained while the penalty for concentrations above the target is temporarily reduced during recovery. The two contributions are defined as

$
  L_t & = 1/N sum_(v=1)^N (max(c^* - c_(v,t), 0) / c_s)^2, \
  U_t & = 1/N sum_(v=1)^N (max(c_(v,t) - c^*, 0) / c_s)^2.
$

The reference scale $c_s$ sets the magnitude of the tracking contributions relative to the other reward components. Deviations smaller than this scale make relatively small contributions, while larger deviations receive increasingly strong penalties through the quadratic scaling. This weighting emphasizes larger concentration errors while remaining sensitive to smaller deviations. Averaging across all monitored nodes gives each location equal weight.

The recovery factor $beta_t$ determines the upper weighting through

$
  f_t = (1 - beta_t) + 0.0001 thin beta_t.
$

At $beta_t = 0$, both sides of the target receive equal weight. As $beta_t$ increases, $f_t$ decreases linearly. At $beta_t = 1$, the upper contribution is reduced to one ten-thousandth of its ordinary value, while the lower contribution remains unchanged. The upper weighting therefore remains positive but is substantially reduced during recovery. This allows increased chlorine levels at some nodes to correct a low residual elsewhere in the network.


=== Wasted Chlorine Movement

The wasted-movement penalty $P_("move",t)$ evaluates changes in chlorine concentrations between two consecutive steps. It penalizes the part of a concentration change that does not reduce the absolute error relative to the target. The penalty is defined as

$
  P_("move",t) = 0.1 thin lr([(1 - beta_t) thin M_t + beta_t thin tilde(M)_t], size: #125%).
$

Here, $M_t$ averages the wasted-movement costs equally across all monitored nodes, while $tilde(M)_t$ weights them according to each node's previous chlorine concentration. The recovery factor $beta_t$ controls the transition between these averages. The coefficient #num[0.1] sets the contribution of this penalty relative to the other reward components.

To distinguish corrective movement from wasted movement, let $m_(v,t)$ denote the total concentration change at node $v$ and $d_(v,t)$ the reduction in its absolute deviation from the target,

$
  m_(v,t) & = abs(c_(v,t) - c_(v,t-1)), \
  d_(v,t) & = max lr((abs(c_(v,t-1) - c^*) - abs(c_(v,t) - c^*), 0), size: #125%).
$

The remaining movement is $w_(v,t) = max(m_(v,t) - d_(v,t), 0)$. This wasted movement is zero for a correction that approaches the target without crossing it. It penalizes movement away from the target, as well as changes that pass through the target and continue beyond it. The corresponding node-level cost is

$
  W_(v,t) = (w_(v,t) / c_s)^2.
$

The reference scale $c_s$ follows the scaling used for chlorine tracking. Small amounts of wasted movement therefore make relatively small contributions, while larger amounts receive increasingly strong penalties through squaring.

During recovery, the weighting shifts toward nodes whose previous concentration was not already high. This allows concentrations at other nodes to rise while low residuals are being restored. To obtain a gradual transition between full and zero node weight, we use the clipped smoothstep function

$
  S(x) = z^2 thin (3 - 2z), quad z = op("clip")(x, 0, 1).
$

Let $c_g$ = #mg-l[0.35] be the full-weight threshold and $Delta c$ = #mg-l[0.05] the transition width. The node weight is then

$
  g_(v,t) = 1 - S((c_(v,t-1) - c_g) / (Delta c)).
$

It is one at or below #mg-l[0.35], decreases smoothly between #num[0.35] and #mg-l[0.40], and is zero at or above #mg-l[0.40]. During recovery, this reduces the penalty for further increases at nodes with already high concentrations, giving priority to restoring low residuals elsewhere. Using the previous concentration prevents an increase from lowering its own weight during the step in which that increase is evaluated. The ordinary and weighted averages are

$
  M_t = 1/N sum_(v=1)^N W_(v,t),
  quad
  tilde(M)_t = (sum_(v=1)^N g_(v,t) thin W_(v,t)) / (sum_(v=1)^N g_(v,t)).
$

When every node weight is zero, $tilde(M)_t$ is defined as zero. Otherwise, normalization by the sum of the weights prevents its scale from decreasing merely because fewer nodes contribute. At $beta_t = 0$, the penalty uses the ordinary average $M_t$. At $beta_t = 1$, it uses the weighted average $tilde(M)_t$, while intermediate values blend the two.


=== Action Change

The action-change penalty $P_("action",t)$ penalizes differences between consecutive normalized actions. It discourages abrupt changes in the requested dosing increments and is defined as

$
  P_("action",t) = 0.1 thin ((a_t - a_(t-1)) / 0.05)^2.
$

As defined in @section:methodology:control-action-and-actuator-constraints, $a_t$ requests an injection increment of $Delta u_"max" thin a_t$. The difference $a_t - a_(t-1)$ therefore measures how this requested increment changes between consecutive steps. Repeating the same action gives zero action-change penalty, even when that action continues to increase or decrease the injection rate.

The reference scale of #num[0.05] corresponds to a change of #mg-min[2500] in the requested increment, or #pct[5] of the maximum permitted increment of #mg-min[50000]. Smaller differences make relatively small contributions, while larger differences receive increasingly strong penalties through quadratic scaling. A weight of #num[0.1] balances the action-change cost against the tracking and movement penalties.

The penalty uses the requested actions rather than the realized injection changes. Actuator clipping therefore does not reduce the penalty when the injection rate approaches its lower or upper bound.


=== Recovery Mechanism

The recovery factor $beta_t$ adjusts the tracking and movement penalties when low chlorine residuals require corrective action. Its value depends on the severity of the observed depletion, the time since the latest recovery trigger, and the time elapsed since the start of the episode. These contributions are combined as

$
  beta_t = v_t thin eta_t thin S(ell_t).
$

Here, $eta_t$ stores the recovery demand at the latest trigger, $ell_t$ controls how long its effect persists, and $v_t$ gradually activates recovery at the beginning of an episode. The function $S$ is the clipped smoothstep function defined above.

Recovery demand is determined by the lowest measured chlorine concentration, so depletion at an individual node can affect the reward weighting. Let $c_r$ = #mg-l[0.20] be the recovery trigger concentration. Using the transition width $Delta c$ defined above, the instantaneous demand is

$
  rho_t = S((c_r - min_(1<=v<=N) c_(v,t)) / (Delta c)).
$

It is one when the minimum is at or below #mg-l[0.15] and zero when the minimum is at or above #mg-l[0.20]. Between these thresholds, the demand increases smoothly as the minimum concentration falls. The interval of #mg-l[0.05] determines the concentration range over which this transition occurs.

To maintain the recovery weighting beyond an individual low measurement, each trigger starts a hold period followed by a gradual release. Let $k_t$ denote the number of steps since the latest trigger, capped at 16. The release remainder is

$
  ell_t = 1 - op("clip")((k_t - 4) / 12, 0, 1).
$

It remains one for four steps after a trigger and then decreases linearly to zero over twelve further steps. Applying $S$ to this remainder gives a smooth reduction in the recovery weighting. Without another trigger, the product $eta_t thin S(ell_t)$ therefore remains constant during a two-hour hold and then decays to zero over a six-hour release.

A new trigger occurs when $rho_t$ is positive and at least as large as $eta_(t-1) thin S(ell_(t-1))$, the previous stored demand after applying the release function. The trigger sets $eta_t = rho_t$ and restarts the hold period at $k_t = 0$. Otherwise, the stored value is held and $k_t$ increases by one until it reaches 16, when $eta_t$ is set to zero. Persistent or renewed depletion can therefore restart the recovery period before the previous response has decayed completely.

At the beginning of an episode, recovery remains inactive for the first twelve executed control steps. During this period, $eta_t = 0$ and $k_t = 16$, giving the policy an initial period to establish dosing before low residuals activate recovery. From step 13 onward, the state follows the update rule above. The startup activation $v_t$ introduces its effect linearly over twelve steps,

$
  v_t = op("clip")((t - 12) / 12, 0, 1).
$

Startup activation reaches its full value at step 24, after which $beta_t = eta_t thin S(ell_t)$. The initial delay and gradual activation are intended to prevent chlorine concentrations, which are initialized to zero at episode start, from immediately encouraging excessive injection through the recovery mechanism. The initial simulation sample does not advance this schedule.


=== Reward Components and Parameters

@tab:methodology-reward-components summarizes the three penalty components and their roles. The movement and action-change penalties compare consecutive executed control steps and are therefore zero at the first executed control step of an episode.

@tab:methodology-reward-parameters lists the selected reference scales, penalty weights, and concentration thresholds, together with the timing parameters of the recovery mechanism. The reference scales and weights set the relative contributions of the penalties, while the thresholds define transitions in node weighting and recovery demand. The timing parameters specify the startup delay and gradual activation, together with the hold and release periods following a recovery trigger.

#{
  show: continued-tables(3)
  set par(justify: false)

  academic-table(
    [Components of the reward function.],
    header: ([*Component*], [*Penalty*], [*Role*]),
    (
      [Chlorine tracking],
      [$P_("track",t) = L_t + f_t thin U_t$],
      [Penalizes deviations from #mg-l[0.30] across all monitored nodes. Recovery reduces the weighting above the target while preserving the full penalty below it.],

      [Wasted chlorine \ movement],
      [$P_("move",t) = 0.1 thin [(1 - beta_t) thin M_t + beta_t thin tilde(M)_t]$],
      [Penalizes concentration changes that do not reduce the absolute error relative to the target. Recovery shifts the weighting toward nodes with lower previous concentrations.],

      [Action change],
      [$P_("action",t) = 0.1 thin ((a_t - a_(t-1)) / 0.05)^2$],
      [Penalizes changes in the requested injection increment between consecutive steps.],
    ),
    format: "booktabs",
    columns: (auto, 1fr, 1fr),
    align: (left, center, left),
    label: <tab:methodology-reward-components>,
  )
}

#{
  show: continued-tables(2)
  set par(justify: false)

  academic-table(
    [Parameters of the reward function. Each control step corresponds to #minute[30].],
    header: ([*Parameter*], [*Value*]),
    (
      table.cell(colspan: 2, fill: luma(235), align: left)[*Chlorine tracking*],

      [Target concentration $c^*$],
      [#mg-l[0.30]],

      [Tracking reference scale],
      [#mg-l[0.01]],

      [Minimum upper-tracking factor],
      [#num[0.0001]],

      table.cell(colspan: 2, fill: luma(235), align: left)[*Wasted chlorine movement*],

      [Movement reference scale],
      [#mg-l[0.01]],

      [Movement penalty weight],
      [#num[0.1]],

      [Full-weight threshold],
      [#mg-l[0.35]],

      [Zero-weight threshold],
      [#mg-l[0.40]],

      table.cell(colspan: 2, fill: luma(235), align: left)[*Action change*],

      [Action-difference reference scale],
      [#num[0.05]],

      [Action-change penalty weight],
      [#num[0.1]],

      table.cell(colspan: 2, fill: luma(235), align: left)[*Recovery mechanism*],

      [Threshold for $rho_t = 1$],
      [#mg-l[0.15]],

      [Threshold for $rho_t = 0$],
      [#mg-l[0.20]],

      [Hold period after a trigger],
      [4 steps / #hour[2]],

      [Release duration after the hold],
      [12 steps / #hour[6]],

      [Startup delay],
      [12 steps / #hour[6]],

      [Startup activation period],
      [12 steps / #hour[6]],
    ),
    format: "booktabs",
    columns: (2fr, 1fr),
    align: (left, center),
    label: <tab:methodology-reward-parameters>,
  )
}


=== Parameter Selection and Optimization

The reward parameters combine heuristic design choices with a focused empirical comparison. The target of #mg-l[0.30] is the midpoint of the normal-operation range of #num[0.2]--#mg-l[0.4]. The movement-weight thresholds of #mg-l[0.35] and #mg-l[0.40] were chosen through inspection of chlorine trajectories. The recovery thresholds follow the assumption that concentrations below the lower operating bound may indicate contamination-related depletion. Recovery demand therefore begins below #mg-l[0.20] and reaches its maximum at #mg-l[0.15]. These thresholds express a heuristic response to low residuals, whose possible causes were discussed above.

The reference scales and penalty weights were chosen during exploratory development of several reward formulations. We kept values that produced useful control behavior in these preliminary runs, without a systematic comparison of the scales or weights. The startup delay and activation duration were chosen to bring recovery into full effect within the first day of an episode. These choices were held fixed during the subsequent parameter search.

We subsequently compared alternative hold periods, release durations, and minimum upper-tracking factors while keeping the selected policy-training hyperparameters fixed. Selection considered contam-?ination response and chlorine control during normal operation on separate validation cases. Physical control performance was used because changing the reward parameters also changes the scale and meaning of the return. The tested configurations and selection procedure are documented in @section:appendix:parameter-selection. The comparison led to the reward configuration in @tab:methodology-reward-parameters.


== Policy Observations <section:methodology:policy-observations>

Both policies receive the same 43-dimensional observation, comprising chlorine measurements, dosing information, calendar features, and the state variables of the recovery mechanism. Node-level chlorine measurements describe the spatial distribution of residuals, while the current injection and its realized change describe the applied dosing. The calendar features indicate the daily, weekly, and annual phases of the periodic demand patterns.

During stochastic training, the action is sampled after the actor's recurrent state has been updated, so that state does not directly record the sampled value. The observation returned after step $t$ therefore includes the selected action $a_t$, clipped to $[-1, 1]$, together with the applied injection $u_t$, and is used to select $a_(t + 1)$. Including both the action and the realized injection change distinguishes the requested increment from the change achieved under the actuator constraints in @section:methodology:control-action-and-actuator-constraints.

The most recent action also serves as the reference for the action-change penalty at the next control step. Because actuator clipping can map different normalized actions to the same realized injection change, that change alone does not determine the penalty for a given subsequent action.

The recovery factor $beta_t$, release remainder $ell_t$, and startup activation $v_t$ provide the temporal context of the reward weighting. These values depend on earlier chlorine depletion and elapsed control steps, so current chlorine measurements alone are insufficient to determine them. Including them in the observation allows the policy to account for the current weighting of tracking and movement costs and the progress of the recovery mechanism.

All observation components can be obtained from chlorine measurements and the controller's own action and time records. Arsenic, demand, pressure, pipe flow, and forecasts are excluded from the policy inputs. The deployed controller therefore uses only a subset of the recorded simulation variables and operates independently of the forecasting model used during pretraining. @tab:methodology-policy-observation lists the raw observation components. Their normalization is described in the following section on recurrent policy learning.

#{
  show: continued-tables(4)
  set par(justify: false)

  academic-table(
    [Policy observation for the 31 monitored junctions of the Hanoi network. Time indices refer to the completed control step. Vector indices are zero-based and follow the order of the observation components.],
    header: ([*Observation component*], [*Entries*], [*Indices*], [*Definition and purpose*]),
    (
      [Node chlorine],
      [31],
      [0--30],
      [Chlorine concentration in #mg-l() at each monitored junction, ordered according to the node list. These entries preserve the spatial distribution available to the controller.],

      [Current injection],
      [1],
      [31],
      [Physical injection rate $u_t$ in #mg-min() applied during the completed control step. It is zero in the initial observation.],

      [Realized injection change],
      [1],
      [32],
      [Difference $u_t - u_(t-1)$ in #mg-min(), including the effect of actuator clipping. It is zero in the initial observation.],

      [Time of day],
      [2],
      [33--34],
      [Sine and cosine encoding of the daily phase.],

      [Time of week],
      [2],
      [35--36],
      [Sine and cosine encoding of the weekly phase.],

      [Time of year],
      [2],
      [37--38],
      [Sine and cosine encoding of the annual phase.],

      [Most recent action],
      [1],
      [39],
      [Normalized action $a_t in [-1, 1]$ applied in the completed control step to produce $u_t$. It is zero in the initial observation.],

      [Recovery factor],
      [1],
      [40],
      [Applied recovery factor $beta_t$, including startup activation. It controls the weighting of the tracking and movement costs.],

      [Release remainder],
      [1],
      [41],
      [Fraction $ell_t$, equal to one at a trigger and throughout the hold period, decreasing linearly during release, and zero when the release is complete.],

      [Startup activation],
      [1],
      [42],
      [Factor $v_t$, equal to zero during the first twelve executed control steps, increasing linearly to one over the next twelve steps, and remaining one thereafter.],
    ),
    format: "booktabs",
    columns: (auto, auto, auto, 1fr),
    align: (left, right, center, left),
    label: <tab:methodology-policy-observation>,
  )
}


== Recurrent Policy Learning and Baseline

We next introduce the recurrent policy-learning framework shared by both approaches, which builds on the control problem, reward design, and policy information defined above.


=== RecurrentPPO Architecture and Baseline

We use RecurrentPPO from SB3-Contrib @sb3contrib2026release, with its `MlpLstmPolicy` implementation, to process the 43-dimensional observation defined above. It extends #acr("PPO") @schulman2017ppo within the Stable-Baselines3 ecosystem @raffin2021stablebaselines3. A feedforward multilayer perceptron supplied with a fixed window of past observations could also use this temporal context, although the window length would determine which past observations are directly available. Here, we use the #acr("LSTM") mechanism to learn a compact representation of the observation history that is updated at each control step, without specifying a fixed input window for the policy. This choice provides a common recurrent architecture for evaluating #acr("MPC")-guided pretraining, while a comparison with history-stacked multilayer perceptron policies lies outside the scope of the study.

At each time step, the observation vector is processed by two separate single-layer LSTMs with 128 hidden units, one for the actor and one for the critic. The actor and critic therefore learn their own recurrent representations rather than sharing a single memory state. Each #acr("LSTM") output is subsequently processed by two fully connected layers of 64 units with hyperbolic-tangent activation functions. The actor uses the resulting representation to parameterize a probability distribution over the normalized control action, whereas the critic estimates the value of the current observation history.

Training proceeds in four parallel control environments. Each environment contributes 2048 consecutive transitions to a rollout, giving 8192 transitions per #acr("PPO") update cycle. These transitions are divided into minibatches of 512 samples and reused for ten optimization epochs. @tab:methodology-recurrent-ppo-configuration summarizes the remaining optimization settings. For a nominal training budget of #num[1000000] environment interactions and 8192 transitions per rollout, the implementation reaches #num[1007616] interactions before stopping.

Both observations and rewards are normalized online using statistics accumulated from the training environments. Each observation component is standardized by its running mean and variance and then clipped to the interval $[-10, 10]$. Rewards are scaled using the running variance of discounted returns, with the same discount factor of $gamma = 0.99$, and are likewise clipped to $[-10, 10]$. Later, during validation and testing, the learned observation statistics are held fixed and reward normalization is disabled. This preserves the input scale learned during training while allowing policy performance to be assessed using the original reward values.

The architecture and training procedure described above define the RecurrentPPO setup shared by both policy approaches. The baseline approach trains the recurrent policy from an ordinary random initialization. The behavior-cloned policy introduced later uses the same architecture and #acr("PPO") configuration. Consequently, the two approaches differ in how the actor and its observation normalization are initialized, not in the representational capacity available during #acr("PPO") training. The following subsection defines the hydraulic scenarios and arsenic schedules used to train both approaches under the common interaction budget.

#{
  show: continued-tables(3)
  set par(justify: false)

  academic-table(
    [RecurrentPPO configuration used by both policy approaches.],
    header: ([*Configuration item*], [*Value*], [*Role*]),
    (
      [Learning rate],
      [#sci("1e-4")],
      [Sets the step size of the Adam optimizer.],

      [Discount factor $gamma$],
      [#num[0.99]],
      [Weights future rewards when estimating discounted returns.],

      [#acr("GAE") parameter $lambda$],
      [#num[0.95]],
      [Controls the bias--variance trade-off of #acr("GAE").],

      [#acr("PPO") clipping range],
      [#num[0.2]],
      [Clips the surrogate objective to discourage excessively large policy updates.],

      [Target #acr("KL") divergence],
      [#num[0.02]],
      [Provides an early-stopping safeguard against excessively large policy updates.],

      [Value-loss coefficient],
      [#num[0.5]],
      [Weights the critic loss within the joint #acr("PPO") objective.],

      [Entropy coefficient],
      [0],
      [Adds no separate entropy bonus beyond the stochasticity of the learned action distribution.],

      [Maximum gradient norm],
      [#num[0.5]],
      [Clips the global gradient norm to stabilize optimization.],

      [Rollout size],
      [8192],
      [Number of transitions collected per rollout.],

      [Minibatch size],
      [512],
      [Number of transitions per optimization minibatch.],

      [Optimization epochs],
      [10],
      [Number of passes over each rollout during a #acr("PPO") update.],
    ),
    format: "booktabs",
    columns: (auto, auto, 1fr),
    align: (left, right, left),
    label: <tab:methodology-recurrent-ppo-configuration>,
  )
}

=== Training Scenario and Contamination Sampling <section:methodology:training-scenario-and-contamination-sampling>

Both #acr("MPC")-guided pretraining and policy learning use the same 80 hydraulic scenarios in the training partition, corresponding to scenario identifiers `scenario-0` to `scenario-79` in @tab:dataset-split, defined in @section:dataset:splits. This keeps the hydraulic scenario collection consistent across the two stages, although their action and state distributions may still differ. Diversity between the individual learning stages is instead maintained through the scenario-wise cross-fitting for #acr("MPC")-guided pretraining described in @sec:methodology-scenario-wise-cross-fitting and through newly generated contamination schedules for policy training.

Each of the four parallel #acr("PPO") environments receives its own deterministic permutation of the 80 training scenarios. This permutation is derived from the policy-training seed together with the environment index, so the environments generally encounter the same scenario collection in different orders. At every environment reset, the next scenario in the corresponding permutation is selected. After all 80 scenarios have been used once, the environment returns to the beginning and repeats the same order without reshuffling between cycles. This procedure gives every environment regular coverage of the complete training split while reducing the synchronization of identical hydraulic conditions across the parallel rollouts.

Whenever an environment selects a scenario, it also generates a new arsenic schedule. Unlike forecasting, validation, and testing, policy training does not draw from a finite set of schedules. Each training environment instead draws from an unbounded deterministic stream. Every generated schedule contains between one and four events and follows the common procedure for sampling event times, locations, durations, and strengths defined in @section:dataset:arsenic-events. The policy therefore encounters new contamination conditions after every reset, even when it returns to a hydraulic scenario seen in an earlier cycle.

The stream must provide varied schedules while remaining reproducible. For this purpose, each parallel environment extends the dedicated control-training schedule seed of 8128 with its environment index. Within an environment, a separate episode counter is maintained for every hydraulic scenario. A schedule is therefore determined by the schedule seed, the environment-specific seed path, the scenario identifier, and the number of earlier episodes generated for that scenario. As a result, repeated visits to a scenario produce different schedules, while a repeated run with the same configuration reconstructs the same sequence.

The schedule seed is intentionally separate from the policy-training seed. Changing the latter can therefore alter the neural-network initialization and the environment-specific scenario permutations without redefining the contamination stream associated with a given environment, scenario, and episode counter. This separation is important for controlled comparisons between policy initializations because it prevents changes to the learning seed from implicitly changing how individual arsenic schedules are generated.


=== Hyperparameter Optimization <section:methodology:policy-hyperparameter-optimization>

The #acr("PPO") configuration was selected through a staged search using ordinarily initialized policies, with the recurrent architecture and reward parameters held fixed. An initial screening varied the learning rate, rollout length, minibatch size, and entropy coefficient one at a time, using #num[500000] environment interactions per candidate. Selected settings and targeted combinations were then compared with budgets of up to one million interactions, followed by a comparison of discount factors and #acr("GAE") parameters at the same budget.

Selection considered learning progress, contamination response, and chlorine control during normal operation on separate validation cases. The test scenarios were excluded from parameter selection. The resulting settings in @tab:methodology-recurrent-ppo-configuration are shared by the baseline approach and the pretrained approach. The search used one training seed and a limited set of candidates, providing a practical configuration for this comparison. The tested configurations, budgets, and selection procedure are documented in @section:appendix:parameter-selection.


== Cross-Fitted Teacher Generation <sec:methodology-main-study-predictive-pretraining>

Having defined the online training conditions shared by both policy approaches, we now turn to the offline teacher-generation stage used to initialize the second policy.

Generating the teacher dataset requires estimating how candidate action sequences affect future chlorine concentrations, since #acr("MPC") compares these sequences through their predicted cumulative rewards. Evaluating every candidate directly with the hydraulic and water-quality simulator would require repeated, computationally expensive simulations. We therefore train multistep forecasters to approximate the network's chlorine response and use them within #acr("MPC") to generate teacher actions. Scenario-wise cross-fitting keeps these actions out of sample with respect to forecaster training. The resulting teacher actions form the demonstration dataset used for behavior cloning.

@fig:methodology-cross-fitted-teacher-generation summarizes the five-fold data flow from offline trajectories to teacher demonstrations.

#figure(
  include "cross-fitted-teacher-generation.typ",
  caption: [Scenario-wise cross-fitted teacher generation. In each of five folds per training seed, #acr("MPC") draws query states from one held-out fold and queries a forecaster trained on the other four folds. The resulting teacher labels are combined for behavior cloning training. The same five trained forecasters are used together by ensemble #acr("MPC") to label query states from the separate validation trajectories for behavior cloning validation. The test split of 10 scenarios or 250 trajectories is omitted because it is not used for teacher generation.],
) <fig:methodology-cross-fitted-teacher-generation>


=== Data Allocation and Scenario-Wise Cross-Fitting <sec:methodology-scenario-wise-cross-fitting>

As #acr("MPC") teacher actions are generated using a pretrained multistep forecaster rather than online simulator evaluations, the data used at each stage must be selected carefully. #acr("MPC") uses the forecaster to search for action sequences with high predicted returns. If this search were performed on scenarios that the forecaster had already encountered during training, the selected actions could benefit from scenario-specific model fit or exploit prediction errors present in those trajectories. The resulting action sequences could consequently provide an overly optimistic representation of the teacher's performance on unseen hydraulic conditions.

One way to avoid this overlap would be to construct two independently simulated datasets. The first would be used to train the forecaster, while the second would provide previously unseen query states for the #acr("MPC") teacher. However, constructing such datasets with comparable coverage would require a substantial number of simulator interactions. The offline training dataset already contains 2000 trajectories, corresponding to #num[2.4] million simulator time steps. Generating an equivalent collection of trajectories for the #acr("MPC") teacher would require another #num[2.4] million simulation steps and therefore double the simulation effort before behavior cloning begins.

To avoid this additional simulation effort, the pipeline instead uses five-fold cross-fitting across the 80 scenarios assigned to the forecasting-training split. These scenarios are partitioned in stable round-robin order into five non-overlapping folds of 16 scenarios. For each fold, one forecaster is trained on the 64 scenarios contained in the other four folds. All trajectories associated with the 16 held-out scenarios, including their different chlorine profiles and arsenic schedules, remain excluded from that forecaster's training data. The precomputed offline trajectories of the held-out fold then provide query states for the corresponding #acr("MPC") teacher without requiring further simulator runs.

The held-out role rotates across the five folds. Each scenario is therefore used to train four forecasters and supplies teacher query states for the remaining forecaster. Cross-fitting requires five predictive models instead of one, but allows all 80 scenarios to contribute both to model fitting and to out-of-fold teacher generation.

The ten global forecasting-validation scenarios are used to select the forecasting configuration and checkpoints, and later to construct ensemble teacher labels and select the behavior-cloning checkpoint. This reuse makes the intermediate validation losses model-selection criteria, not independent estimates of teacher-action quality. Final policy quality is assessed on the separate test cases.

Once a fold-specific forecaster has been trained, #acr("MPC") uses it to evaluate candidate injection sequences for query states from its held-out fold. For each query state, the first action of the sequence with the highest predicted return is retained as the teacher target. Repeating this process for all five folds produces five teacher datasets, which are subsequently combined into a single out-of-fold dataset for behavior cloning. Every forecasting-training scenario contributes to this training dataset exactly once. Separate teacher labels are generated for the existing forecasting-validation trajectories using all five forecasters as an ensemble. These labels provide the validation targets for early stopping and checkpoint selection during behavior cloning.

Cross-fitting reduces the risk of optimistic teacher supervision, but it does not make the selected actions equivalent to simulator-derived ground truth. #acr("MPC") continues to evaluate candidate actions using predicted chlorine trajectories. The quality of its decisions therefore depends on how accurately each fold-specific forecaster represents the dynamics of its held-out scenarios. The procedure ensures that this accuracy is assessed outside the scenarios used for fitting, but it cannot remove errors caused by imperfect generalization.


=== History-Window and Forecast-Horizon Selection <section:methodology:temporal-window-selection>

The forecasting model uses a 48-step history and a 16-step prediction horizon, corresponding to #hour[24] of past context and eight hours of planned future injections at the 30-minute sampling interval. The prediction horizon also defines the action sequence evaluated by #acr("MPC").

We checked the timing of the selected windows by comparing the network response to a brief increase in chlorine injection with a constant-injection reference. Extending the history from 20 to #hour[24] covers most remaining detectable chlorine effects, while longer histories add little coverage. The 24-hour window also spans a complete daily cycle. By the eight-hour horizon, almost all tested pulse effects have begun and peaked, and most have settled. Longer horizons cover only small additional shares of cases while requiring longer forecasts and evaluation of more predicted steps for each #acr("MPC") plan. Both lengths are practical choices for this study. The diagnostic supports this choice but does not establish an empirically optimal horizon, forecasting accuracy, or controller performance. Its protocol, results, and limitations are reported in @section:appendix:network-delay.


=== Forecasting Data and Causal Inputs <section:methodology:forecasting-data>

Each cross-fitted forecasting model is trained on the precomputed offline trajectories. Its 64 training scenarios each combine five prescribed chlorine-injection profiles with five fixed arsenic schedules, yielding 25 trajectories per scenario and 1600 training trajectories per fold-specific model.

The complete pairing varies the chlorine input and contamination schedule while leaving the demand pattern and physical network parameters fixed. The prescribed profiles include both near-steady and changing segments. Starting from zero injection, the profile generator limits every change between consecutive 30-minute injection rates to #mg-min[50000], matching the actuator constraint in @section:methodology:control-action-and-actuator-constraints. This exposes the forecaster to sustained operating conditions and delayed responses after changes in chlorine injection, even though the action sequences do not necessarily correspond to realistic behavior. Multiple arsenic schedules also prevent a hydraulic scenario from being associated with only one contamination pattern. The choice of five profiles and five schedules is a computational-budget decision rather than an empirically optimized configuration.

Every trajectory covers 25 days at the 30-minute reporting interval defined in @section:dataset:simulation-setup, resulting in 1201 ordered samples from the initial state to the final state. Supervised examples are extracted by moving a forecast origin through this sequence in increments of one sample. For each origin, the preceding 48 samples provide a 24-hour causal history, while the following 16 samples form an eight-hour prediction horizon. A complete trajectory yields 1138 overlapping windows because examples are retained only where both the full history and the full horizon are available.

The information within one window is divided according to when it is available. The historical input contains the measured chlorine concentration at all 31 monitored nodes, the chlorine injection that was applied at the source, and sine--cosine encodings of the daily, weekly, and annual phases. Together, these quantities form 38 features at each of the 48 historical steps. The future input contains only information that is known when a forecast is requested, namely the planned source-injection sequence and the same six calendar features. This makes seven features for each of the 16 future steps. The prediction target is the chlorine concentration at all 31 monitored nodes over these 16 steps. @tab:methodology-forecast-inputs summarizes this causal arrangement.

Before forecaster training, the forecasting inputs and targets are standardized to account for their different numerical scales. Historical and future-known inputs are standardized feature by feature, while separate statistics are used for each node-level chlorine target. All means and standard deviations are estimated exclusively from training data and subsequently applied unchanged to held-out data. The normalization statistics are computed across the extracted forecasting windows. Each sample contributes once for every window in which it appears, so the resulting scaling reflects the frequency with which it is encountered during forecaster training.

Advancing the forecast origin by one sample creates overlapping and therefore strongly correlated windows. An example-level split could place nearly identical histories and targets in both training and validation data, causing evaluation performance to reflect this overlap rather than generalization to unseen hydraulic conditions. We therefore preserve the scenario-level partitions defined in @section:dataset:splits and keep all 25 trajectories and their derived windows together. This prevents information leakage and supports the cross-fitting design defined in @sec:methodology-scenario-wise-cross-fitting.

#{
  set figure(placement: none)
  set par(justify: false)
  show table: format-table(none, auto, auto, none)

  academic-table(
    [Causal input and target structure of one supervised forecasting window.],
    header: ([*Window component*], [*Steps*], [*Features*], [*Contents and role*]),
    (
      [History],
      [48],
      [38],
      [Chlorine at 31 monitored nodes, the applied source injection, and six cyclic calendar features. It describes the observed #hour[24] preceding the forecast horizon.],

      [Planned injection],
      [16],
      [7],
      [The planned source injection and six cyclic calendar features for the next eight hours. During #acr("MPC"), the injection entries are replaced by a candidate control sequence.],

      [Prediction target],
      [16],
      [31],
      [Future chlorine levels at all monitored nodes. These values supervise the forecaster but are not available as inputs.],
    ),
    format: "booktabs",
    columns: (auto, auto, auto, 1fr),
    align: (left, right, right, left),
    label: <tab:methodology-forecast-inputs>,
  )
}


=== Explicit Multistep Chlorine Forecaster

The forecasting task defined above requires a model that can represent the action-dependent network response over the selected horizon. For this purpose, we use the sequence-to-sequence encoder--decoder #acr("LSTM") to approximate how the simulator would respond to a planned chlorine-injection sequence. The encoder learns a compact representation of the observed history, while the decoder combines this context with the planned injections and calendar features at each future step. This structure allows #acr("MPC") to compare the predicted chlorine responses to different injection sequences under the same historical conditions. We adopt this architecture as a practical forecasting model for evaluating #acr("MPC")-guided policy pretraining, while a systematic comparison with alternative forecasting architectures lies outside the scope of the study.

The encoder processes the normalized $48 times 38$ historical input and summarizes the preceding #hour[24] in the hidden and cell states of a single-layer #acr("LSTM") with 192 hidden units. These states initialize a single-layer #acr("LSTM") decoder with the same hidden size. The decoder then processes the $16 times 7$ future-known input, which combines a candidate injection sequence with the corresponding calendar features. A shared linear output head maps the decoder state at each future step to the chlorine concentrations of all 31 monitored nodes. The resulting prediction therefore has shape $16 times 31$ and represents absolute chlorine levels at each junction over the next eight hours.

Training minimizes the #acr("MSE") between predicted and simulated chlorine levels after independently standardizing the target of each monitored node with statistics fitted on the training split. The loss gives equal weight to all 16 horizon steps and 31 node targets. After every epoch, predictions are transformed back to #mg-l() and the overall validation #acr("RMSE") is computed across nodes and horizon steps. The checkpoint with the lowest validation #acr("RMSE") is retained, and training stops early after ten consecutive epochs without an improvement. The model is optimized with AdamW for at most 100 epochs, using a batch size of 256, a learning rate of #sci("7.5e-4"), and a weight decay of #sci("1e-5"). These hyperparameters were fixed without systematic optimization, as a broader hyperparameter search was considered outside the scope of this study.

The architecture and optimization procedure define the forecasting configuration used throughout the teacher-generation stage. To generate teacher labels, a forecaster is trained separately within each scenario-wise cross-fitting fold, as described in @sec:methodology-scenario-wise-cross-fitting. Consequently, the #acr("MPC") teacher does not evaluate a query state with a forecaster trained on the corresponding hydraulic scenario.


=== MPC Teacher and CEM Optimization

Next, we use #acr("MPC") together with the fold-specific forecasters to generate teacher actions. #acr("MPC") searches over the same normalized rate-change actions available to the control policy, while the forecasters predict the resulting chlorine trajectories. These trajectories are scored using the policy-training reward, and the first action of the highest-scoring plan is retained as the teacher action.

Each teacher action is generated for a query state representing a selected time point in one of the precomputed trajectories. The query state provides the recent forecasting history and the current control and reward context from which #acr("MPC") begins its search.

For each query state, #acr("MPC") evaluates candidate action sequences over the forecaster's complete 16-step horizon, equivalent to eight hours. Directly optimizing one independent action at every step results in a 16-dimensional search space. We therefore represent each sequence using eight control points at zero-based horizon offsets 0, 1, 2, 3, 5, 8, 11, and 15 and obtain the remaining actions through linear interpolation. The control points are concentrated near the beginning of the horizon and spaced more coarsely at later steps. This gives the beginning of the plan, including the action retained as the teacher label, greater temporal resolution while reducing the dimension of the search. All interpolated actions remain within $[-1, 1]$.

Each interpolated action sequence is first converted into a feasible physical injection sequence using the actuator update in @eq:methodology-actuator-update. For a given query state, all candidates share the same normalized forecasting history and future calendar features and differ only in their planned injection sequences. The fold-specific forecaster evaluates these candidates as one batch and predicts a $16 times 31$ chlorine trajectory for each plan, expressed in #mg-l() after reversing the target normalization. Before scoring, the predictions are checked for finite values and negative concentrations are set to zero. A nonfinite prediction causes the complete #acr("MPC") query to fail.

These trajectories are evaluated sequentially using the same stateful reward mechanism as policy training. Every candidate begins from the same reward state at the query time, ensuring that all plans share identical recovery and temporal context. For candidate $m$, the #acr("MPC") objective is

$ J^((m))_t = sum_(h=1)^16 gamma^(h - 1) thin r^((m))_(t + h), quad gamma = 0.99. $

Thus, earlier predicted outcomes receive slightly greater weight, while delayed network responses within the eight-hour horizon still contribute to the comparison. No separate terminal value is added after the final forecast step.

For behavior-cloning validation, every candidate plan is predicted by all five fold-specific forecasters. Each predicted chlorine trajectory is scored separately from the same initial reward state, yielding one planned return $J^((m,k))_t$ per model $k$. The ensemble objective is their arithmetic mean,

$ overline(J)^((m))_t = 1/5 sum_(k=1)^5 J^((m,k))_t. $

#acr("CEM") optimizes this mean return, and the first action of the best plan becomes the validation teacher label. Chlorine predictions are not averaged before reward evaluation, and independently selected actions are not averaged.

The continuous control-point vector is optimized with the #acr("CEM"). #acr("CEM") maintains an independent Gaussian sampling distribution for each of the eight control points. The initial mean is zero and the initial standard deviation is one, after which sampled values are clipped to the admissible action interval. In every iteration, 63 plans are drawn from this distribution and combined with one fixed all-zero plan. The latter represents holding the current injection constant throughout the horizon and provides the optimizer with the same reference candidate in every iteration. All 64 plans are expanded, forecast, and scored as described above.

With 64 candidates, the #pct[10] elite fraction is rounded up to seven, retaining those with the highest predicted returns. Their component-wise mean and standard deviation define the next sampling distribution. Each standard deviation is bounded below by $1/60$ to prevent complete collapse of the search distribution. After four iterations, the optimizer returns the highest-scoring candidate across all populations. @tab:methodology-cem-configuration summarizes the planning configuration.

At each recorded query state, #acr("MPC") optimizes a complete 16-step plan and retains only its first normalized action as the teacher label. The selected action is stored as a teacher label but is not applied to the simulator to generate subsequent trajectory states. The remaining actions serve only to evaluate the longer-term implications of that decision and are discarded when the behavior-cloning example is formed. The predicted chlorine trajectories and the forecaster's internal representations likewise remain confined to #acr("MPC") scoring.

#{
  show: continued-tables(3)
  set par(justify: false)

  academic-table(
    [#acr("CEM") configuration used to optimize #acr("MPC") teacher actions.],
    header: ([*Configuration item*], [*Value*], [*Role*]),
    (
      [Planning horizon],
      [16 steps / #hour[8]],
      [Matches the explicit multistep forecast horizon.],
      [Action parameterization],
      [8 control points],
      [Uses offsets 0, 1, 2, 3, 5, 8, 11, and 15 and linearly interpolates the remaining actions.],

      [Candidate population],
      [64],
      [Evaluates 63 sampled plans and one fixed hold plan per iteration.],
      [#acr("CEM") iterations],
      [4],
      [Repeatedly refines the control-point sampling distribution.],
      [Elite set],
      [7 candidates],
      [Uses the highest-scoring #pct[10] of the population to update the distribution.],
      [Initial distribution],
      [$mu = 0$, $sigma = 1$],
      [Starts from zero-mean independent Gaussian control points before clipping to $[-1, 1]$.],

      [Minimum standard deviation],
      [1/60],
      [Maintains non-zero sampling variation in every control-point dimension.],
      [Reward discount],
      [$gamma = 0.99$],
      [Weights the predicted rewards over the planning horizon.],
    ),
    format: "booktabs",
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    label: <tab:methodology-cem-configuration>,
  )
}


=== Query-State Sampling and Teacher-Dataset Construction

Within each held-out fold, query states are sampled from the corresponding precomputed trajectories. A query time must leave sufficient data for both the preceding history and the subsequent planning horizon. The 48-step causal history includes the query itself as its final sample, while the 16-step planning horizon starts at the following sample.

Applying #acr("MPC") at all valid positions would produce more labels than required for pretraining, many of which would be closely related. The sampling procedure therefore selects 24 query times from each trajectory. It divides the valid range into 24 consecutive strata of approximately equal length and samples one query from each stratum. The resulting labels span the complete usable portion of the 25-day episode instead of being concentrated in a small number of periods.

The temporal strata distribute queries across each trajectory, but do not necessarily balance their positions within the daily demand cycle. Each candidate query corresponds to one of 48 half-hour slots within a day. When selecting from a stratum, the sampler identifies the slots currently least represented across the fold and samples from their candidate positions. The teacher dataset consequently covers both episode progression and the complete daily cycle without imposing fixed query intervals within individual trajectories. Trajectory and stratum order are randomized deterministically so that this balancing does not introduce systematic associations with trajectory characteristics or position within the episode.

For each selected query time, the recorded trajectory is replayed from the beginning of the episode up to and including the query. This reconstructs the reward state under the temporal context in which it originally evolved and provides the initial state for evaluating candidate action sequences. The 48 causal samples ending at the query form the forecasting history defined in @tab:methodology-forecast-inputs, which the fold-specific forecaster uses to predict the chlorine response to each candidate sequence.

After the query state has been reconstructed, the fold-specific #acr("MPC") teacher applies the optimization procedure described above. The first normalized action of the highest-scoring sequence becomes the teacher target at that query time. Repeating this step for all sampled queries produces one set of query-level targets for each held-out fold. The five sets are then combined into a single scenario-wise out-of-fold teacher dataset. The same query-sampling and reward-state reconstruction procedure is applied to the 250 existing forecasting-validation trajectories, producing 24 ensemble-teacher labels per trajectory. If all #acr("MPC") queries succeed, the 24 query times per trajectory yield #num[48000] training and 6000 validation teacher labels per seed.

The resulting dataset associates each complete episode with the teacher actions generated at its sampled query times. During behavior cloning, the recurrent actor processes the complete sequence of policy observations and is supervised only at these labeled positions. The forecaster and #acr("MPC") optimizer therefore supply the action targets but do not become part of the actor's inputs.


== Behavior Cloning and PPO Fine-Tuning

The teacher dataset constructed above provides the action targets used to initialize the policy of the second training approach. This section explains how we first train the recurrent actor to reproduce these actions and then transfer the resulting actor and observation-normalization state to the policy before online fine-tuning begins.


=== Policy Initialization

Behavior cloning uses the actions from the #acr("MPC") demonstration dataset to initialize the recurrent actor for the second policy approach. The cloned actor has the same architecture as the actor defined for RecurrentPPO above, comprising a single-layer #acr("LSTM") with 128 hidden units, two fully connected layers with 64 units each, and a scalar action output. At each labeled query state, the actor #acr("LSTM") is trained to reproduce the first normalized action selected by the #acr("MPC") teacher.

Each teacher action was generated from a 48-step forecasting history, but presenting only this window to the actor would reset its recurrent state before every query. Behavior cloning therefore processes each complete 1201-sample episode as a chronological sequence of normalized 43-value policy observations. The actor's recurrent state is initialized to zero only at the beginning of an episode. To limit the computational cost, training uses truncated #acr("BPTT") with 48-step chunks. After each chunk, the recurrent state is detached from the gradient graph and passed to the next chunk. The chunk length was selected as a practical configuration rather than through systematic optimization and is therefore not assumed to be optimal. All observations, including those in chunks without query labels, contribute to the evolving recurrent state, while the cloning loss is evaluated only at the 24 labeled query positions in each episode.

Behavior cloning uses the combined demonstrations from all 80 training scenarios, comprising 2000 trajectories. Early stopping and checkpoint selection use the separate validation demonstrations described in @sec:methodology-scenario-wise-cross-fitting. Each of the ten validation scenarios retains all 25 trajectories and all 24 labels per trajectory. Complete validation episodes reconstruct the actor's recurrent state in the same way as training episodes, while validation #acr("MSE") is computed only at labeled query positions. Validation data contribute neither gradients nor observation-normalization statistics.

Observation normalization is fitted before optimization using every time step from the 2000 complete trajectories covering all 80 training scenarios. The ten validation scenarios remain excluded from these statistics. Each observation component is standardized and clipped to $[-10, 10]$, matching the preprocessing used by RecurrentPPO. The fitted normalization state is held fixed during behavior cloning and saved together with the policy for subsequent fine-tuning.

At each labeled query, the #acr("MSE") measures the difference between the actor mean and the teacher action. Adam optimizes the actor #acr("LSTM"), its two fully connected layers, and the action-output layer with a learning rate of #sci("3e-4"). Each minibatch contains 16 complete trajectories, and actor gradients are cleared before its first chunk. For each chunk with labels, squared errors at the labeled queries are summed and divided by the total number of labels in the minibatch before backpropagation. The resulting gradients accumulate across chunks, are clipped to norm $0.5$, and are applied in one Adam step after the final chunk. The deterministic teacher actions provide no target for the action-distribution standard deviation. We set this standard deviation to the median, across training queries, of the final #acr("CEM") elite standard deviations for the first action. This is a practical initialization rather than a value optimized for #acr("PPO").

Training runs for at most 100 epochs. After each epoch, the #acr("MSE") is evaluated at the labeled queries in the 250 complete trajectories from the ten separate validation scenarios. The actor with the lowest validation loss is retained, and optimization stops after ten consecutive epochs without improvement. The selected behavior-cloning result supplies the actor parameters, initial action standard deviation, and observation-normalization state used in the following initialization stage. The critic remains ordinarily initialized.


=== Transfer of Actor and Normalization State

Before #acr("PPO") fine-tuning, the second RecurrentPPO policy receives the actor trained through behavior cloning. Because both stages use the same actor architecture, the parameters of the actor #acr("LSTM"), its two fully connected layers, and the action-output layer can be transferred directly. The action-distribution standard deviation specified during behavior cloning is transferred as well. The actor therefore begins online training with parameters already fitted to reproduce the #acr("MPC") teacher actions.

The critic is not transferred because the teacher dataset provides action labels but no return or value targets. Its #acr("LSTM") and value network retain their ordinary random initialization and learn from returns collected during #acr("PPO") training. Behavior cloning thus initializes only the policy component that is directly supervised by the #acr("MPC") teacher.

The observation-normalization state fitted on the behavior-cloning training trajectories, including its running count, is transferred with the actor. During #acr("PPO") training, these statistics continue to incorporate observations from the control-training environments and can adapt to states visited by the fine-tuned policy. Behavior cloning provides no reward-normalization state, so these statistics are accumulated from online rewards in the same way as in the baseline approach.


=== RecurrentPPO Fine-Tuning

After the transfer, #acr("PPO") fine-tunes the recurrent policy through interaction with the EPANET-based hydraulic and water-quality simulator, using a newly initialized optimizer. It updates the transferred actor and its action-distribution parameters while the ordinarily initialized critic is trained from the collected returns. The actor can therefore adapt the behavior learned from the #acr("MPC") teacher using rewards observed during closed-loop control.

Fine-tuning uses the same nominal budget of one million environment interactions and the RecurrentPPO configuration in @tab:methodology-recurrent-ppo-configuration. It also follows the control-training scenario and contamination-sampling procedure defined in @section:methodology:training-scenario-and-contamination-sampling. The baseline approach and the pretrained approach consequently use the same #acr("PPO") update budget, reward signal, and online training conditions. Their intended methodological difference is the actor parameters, initial action standard deviation, and observation-normalization state available when #acr("PPO") begins.

Neither the chlorine forecaster nor the #acr("MPC") optimizer participates in fine-tuning, evaluation, or deployment. These stages use only the recurrent policy and the deployable 43-value observation defined in @tab:methodology-policy-observation. After fine-tuning, both approaches therefore produce policies with the same architecture and inference inputs. The following comparison can thus assess the effect of policy initialization without changing the controller inputs or architecture.


== Principal Experimental Comparison

The principal experiment compares ordinarily initialized RecurrentPPO with #acr("MPC")-guided policy pretraining under a fixed online interaction budget. Pretraining supplies the actor parameters, initial action standard deviation, and observation-normalization state, while the critic remains ordinarily initialized. Both approaches share the architecture, reward, training scenarios, contamination sampling, and online budget, so the comparison assesses the effect of initialization. The additional variants below examine the contributions of a larger online budget and the cloned actor weights.


=== Additional Comparison Variants <section:methodology:additional-variants>

Two additional variants supplement the principal comparison. The extended-budget baseline trains ordinarily initialized RecurrentPPO for four million nominal online interactions. This budget corresponds to the three million simulator steps used to generate the precomputed offline trajectories across the training, validation, and test partitions, plus the pretrained approach's one million nominal online interactions. It examines whether additional online training allows the baseline approach to reach the pretrained approach's validation return. Separate simulator checks and control evaluations are excluded from this budget comparison.

The variant without actor weights receives the observation-normalization state and initial action standard deviation from the same behavior-cloning result as its paired pretrained run, while its actor remains ordinarily initialized. It trains for one million nominal online interactions to assess the contribution of the cloned actor weights conditional on these two transferred components. Both variants retain the ordinarily initialized critic and the common #acr("PPO") settings, training scenarios, and contamination sampling. Observation-normalization statistics continue to update during online training. Their evaluation is defined in @section:evaluation:additional-variants.


== Implementation and Reproducibility <section:methodology:implementation-and-reproducibility>

For each training seed listed in #chapter-ref(<chapter:evaluation>), the five cross-fitted forecasters are trained separately, training and ensemble-validation #acr("MPC") teacher labels are regenerated using that seed's five forecasters, and behavior cloning and both #acr("PPO") runs are repeated. The precomputed offline trajectories and fixed evaluation cases are shared across seeds.

The experiments use Python 3.13.3 @pythonsoftwarefoundation2025python3133 with NumPy 2.4.4 @numpy2026release, PyTorch 2.11.0 @pytorch2026release, Gymnasium 1.2.3 @gymnasium2025release, Stable-Baselines3 2.8.0 @sb32026release, SB3-Contrib 2.8.0 @sb3contrib2026release, and EPyT-Flow 0.17.0 @artelt2026epytflow017. EPyT-Flow calls EPANET-PLUS 0.3.1, which provides the interface to OWA-EPANET 2.3.6 and #acr("EPANET-MSX") 2.0.0 @artelt2026epanetplus.

The training computer used for the reported runtime measurements runs on Windows 11 and has an AMD Ryzen 7 9800X3D processor with eight physical cores and eight logical processors available to the operating system, 32 GiB of DDR5 memory configured at 6000 MT/s, and an NVIDIA GeForce RTX 3080 Ti with 12 GiB of dedicated GPU memory.
