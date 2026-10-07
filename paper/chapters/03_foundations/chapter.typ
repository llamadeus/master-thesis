#import "@preview/acrostiche:0.7.0": acr

= Foundations <chapter:foundations>

Water and dissolved substances take time to move through a distribution network @rossman2020epanet. Their paths and travel times depend on the hydraulic conditions, so a change in solute injection may affect downstream concentrations only after a transport delay. A controller must therefore account for consequences that unfold over time. When, on top of that, the complete system state is not directly observable, it must additionally base its decisions on partial observations @kaelbling1998pomdp.

This problem draws on dynamical systems, water-quality simulation, machine learning, and control theory, and the underlying foundations are introduced in this chapter. It first introduces trajectories and observations, followed by the models used for sequential decisions and forecasting. It then presents predictive control and imitation learning, while specific choices concerning the network, data allocation, model configuration, and control objective are deferred to later chapters.


== Mathematical Foundations

Prediction and control can be described in terms of the evolution of a physical system. The state describes the system at one instant, while a trajectory orders successive states over time. Probability represents variation among possible trajectories, and time-series notation organizes the measurements obtained from them. These concepts provide a common notation for the evolving system state, the observations derived from it, and the uncertainty affecting its development.


=== Discrete-Time Dynamical Systems and Trajectories

A dynamical system can be represented by a state that evolves over time @rawlings2020mpc. At discrete instants $t = 0, 1, dots, T$, a state transition with a control input and a disturbance can be written as follows.

$
  bold(x)_(t+1) = f(bold(x)_t, u_t, bold(w)_t),
$

where $bold(x)_t$ denotes the system state, $u_t$ a control input, and $bold(w)_t$ an external disturbance or unresolved source of variation. The transition function $f$ maps these quantities to the next state. Random disturbances make this a stochastic model @rawlings2020mpc. A Markov state contains the past information needed to determine the next-state distribution given the control input @sutton2018reinforcement. Its components need not be directly observable.

A simulation run or system interaction generates an ordered sequence of states and controls @sutton2018reinforcement. We write this trajectory as

$
  tau = (bold(x)_0, u_0, bold(x)_1, u_1, dots, u_(T-1), bold(x)_T).
$

The temporal order is part of the trajectory. Rearranging otherwise identical values generally describes a different evolution because controls, transport delays, and accumulated disturbances affect later states.

An observation $bold(o)_t = h(bold(x)_t, bold(v)_t)$ represents a measurement of the state, where $bold(v)_t$ accounts for measurement effects. This notation includes the additive-noise model $bold(o)_t = h_0 (bold(x)_t) + bold(v)_t$ as a special case @rawlings2020mpc. If the measurement uniquely determines the state, it provides full state information. Otherwise, several observations and the intervening actions may help infer the current condition @kaelbling1998pomdp.


=== Time-Series Representations

A univariate time series is an ordered sequence $y_1, y_2, dots, y_T$ indexed by time @bergmeir2018crossvalidation. In a multivariate series, each entry is a vector $bold(y)_t in RR^m$, where $m$ is the number of variables. With a fixed sampling interval $Delta t$, an index difference of $k$ corresponds to a physical duration of $k thin Delta t$. Successive entries can be statistically dependent @bergmeir2018crossvalidation.

At a forecast origin $t$, a history window of length $H$ contains the observations available up to that instant, while a prediction horizon of length $K$ contains the subsequent targets,

$
  bold(X)_t & = (bold(y)_(t-H+1), dots, bold(y)_t), \
  bold(Y)_t & = (bold(y)_(t+1), dots, bold(y)_(t+K)).
$

Moving the forecast origin along a trajectory yields multiple forecasting windows @bergmeir2018crossvalidation. Adjacent windows generally overlap and share historical or future values. They therefore do not constitute independent repetitions of the underlying trajectory.

Periodic calendar variables can be represented with sine and cosine terms @hyndman2021forecasting. For phase $q$ with period $P$, the encoding

$
  phi(q) = lr((sin(2 pi q/P), cos(2 pi q/P)), size: #125%)
$

maps the phase to the unit circle. The pair identifies the phase modulo $P$ and keeps the end and beginning of the cycle adjacent. Separate pairs can represent daily, weekly, and annual cycles without the jump in a scalar index at each cycle boundary.


== Sequential Decision-Making

A forecast describes how a system may evolve under given inputs, whereas a controller selects inputs that change this evolution @rawlings2020mpc. Actions can affect both future outcomes and the conditions under which subsequent decisions are made. Sequential decision-making captures this feedback. Markov decision processes provide the basic model, and partial observability accounts for measurements that reveal only part of the physical state @kaelbling1998pomdp.


=== Markov Decision Processes

A #acr("MDP") describes sequential interaction through states, actions, transition probabilities, and rewards @sutton2018reinforcement. Including the discount factor, we write

$
  cal(M) = (cal(S), cal(A), P, R, gamma).
$

Here, $cal(S)$ is the state space and $cal(A)$ the action space. In a finite-action problem, $cal(A)$ contains a discrete set of choices, while in continuous control, it is instead a subset of $RR^d$ @sutton2018reinforcement. The transition kernel $P(s'|s,a)$ describes how the current state and action determine the next-state distribution, and $R(s,a,s')$ specifies the expected immediate reward. The discount factor $gamma in [0,1]$ determines the relative weight of future rewards, with $gamma=1$ permitted for finite episodic returns. The Markov property requires the current state to contain all past information relevant to the next-state and reward distribution @sutton2018reinforcement. A policy $pi(a|s)$ maps states to action probabilities or densities. Together with an initial-state distribution, the policy and transition dynamics induce a distribution over trajectories.

This formulation assumes that the controller has access to a state representation that satisfies the Markov property @kaelbling1998pomdp. When the available observations reveal only part of that state, the decision process must instead account for partial observability.


=== Partial Observability

A #acr("POMDP") extends an #acr("MDP") with an observation space and an observation model @kaelbling1998pomdp. We denote the observation space by $cal(O)$. For the special case in which observations depend only on the resulting state, we use the model $Omega(o|s)$. The environment evolves through a latent state $S_t$, but the agent receives only an observation $O_t$. Even if the latent state is Markovian, the current observation need not be. The same measurement can arise from several hidden physical conditions with different future consequences @kaelbling1998pomdp.

The observation-action history available at time $t$ can be written as $cal(H)_t = (o_0, a_0, dots, a_(t-1), o_t)$. Given an initial belief, the history determines a belief state over the latent states @kaelbling1998pomdp. For discrete states,

$
  b_t (s) = Pr(S_t=s | cal(H)_t).
$


This belief state is a probability distribution over latent states @kaelbling1998pomdp. Exact Bayesian belief updates require the transition and observation models. Decisions may instead be conditioned on the observation-action history or on a compact representation learned from it. Recurrent models, introduced later in this chapter, provide one way to learn such a representation @ni2022recurrent.

Such representations can retain evidence that is absent from the current observation alone @ni2022recurrent. A learned history representation need not equal the Bayesian belief state. Moreover, identical observation-action histories yield the same representation when the initial recurrent state and deterministic update rule are fixed, so they cannot resolve the underlying state ambiguity.


== Hydraulic and Water-Quality Simulation

A change in solute injection can produce a delayed concentration response at downstream locations @rossman2020epanet. Water follows paths determined by the current flows, transports dissolved substances downstream, and mixes at junctions while chemical reactions continue @nrc2006distribution. A node's concentration therefore depends on the chemical inputs and on the hydraulic history that transported the water there. Hydraulic and water-quality simulation represents these processes as coupled parts of the same evolving system.


=== Water Distribution Networks and Hydraulic States

A water distribution network can be represented as a graph $cal(G)=(cal(V),cal(E))$ @rossman2020epanet. Reservoirs, tanks, and demand junctions form the nodes $cal(V)$, while pipes, pumps, and valves form the links $cal(E)$. Elevation, storage, pipe geometry, roughness, and component settings supplement this topology with the properties required for hydraulic calculation.

At each junction, conservation of water requires the incoming flow to equal the outgoing flow plus external demand @rossman2020epanet.

$
  sum_(e in cal(E)_v^"in") q_e (t)
  =
  sum_(e in cal(E)_v^"out") q_e (t) + d_v (t),
$

where $cal(E)_v^"in"$ and $cal(E)_v^"out"$ contain the links whose reference orientations point into and out of junction $v$, respectively. The signed volumetric flow through link $e$ is $q_e (t)$, and $d_v (t)$ denotes the external demand flow at the junction. Hydraulic head combines elevation and pressure head, while link equations relate head differences to flow through pipes, pumps, and valves. Solving these coupled continuity and energy relations yields nodal heads, pressures, and link flows for the specified demand model @rossman2020epanet.

A fixed reference orientation defines the sign of each computed link flow without requiring water to move permanently in that direction @rossman2020epanet. A negative value indicates flow opposite to the reference orientation. As demands, storage levels, and active components change, flow magnitude and direction may change as well @rossman2020epanet. The time-varying hydraulic conditions therefore determine the effective downstream paths and travel times of transported substances even though the physical topology remains fixed.


=== Solute Transport, Mixing, and Mass-Rate Sources <section:foundations:transport-mixing-mass-rate>

Advection transports a dissolved species with the bulk water flow @shang2023epanetmsx. In a pipe with cross-sectional area $A_e$ and volumetric flow $q_e (t)$, the mean velocity is $v_e (t)=frac(q_e (t), A_e)$. When longitudinal dispersion is neglected, the concentration field $c_e (x,t)$ follows an advection--reaction equation of the form @rossman2020epanet

$
  frac(partial c_e, partial t)
  + v_e (t) frac(partial c_e, partial x)
  = r(c_e).
$

Here, $x$ denotes position along the pipe and $t$ denotes continuous physical time. The function $r$ describes reactions that consume or produce the species. Through $v_e (t)$, the time-varying hydraulic conditions determine the transport direction and residence time @rossman2020epanet.

At an ideal junction without storage, incoming streams are assumed to mix instantaneously @rossman2020epanet. Conservation of solute mass yields a single junction concentration from the mass carried by the incoming flows, any externally added mass, and the total flow leaving the junction. This concentration applies to all water leaving the junction, including local demand and flow entering downstream links. Mixing thus combines concentrations from different transport paths and carries their preceding reaction histories into the downstream network.

A mass-rate source specifies injected solute mass per unit time, whereas concentration denotes mass per unit water volume @rossman2020epanet. If a source adds mass at rate $dot(m)$ to water flowing at a positive rate $q$, its concentration contribution under ideal mixing is $dot(m)/q$. A fixed mass rate therefore does not imply a fixed concentration. Higher flows dilute the input more strongly, and concentrations at downstream locations also depend on mixing, reaction, and transport delay @rossman2020epanet.


=== Free-Chlorine Decay and Chlorine--Arsenite Reactions

Free-chlorine loss is commonly represented by a first-order decay model, whose rate is proportional to the local chlorine concentration @rossman2020epanet @shang2023epanetmsx. In this model, the concentration in a given volume of water decreases with increasing residence time. Transport, dilution, and mixing determine its concentration profile along each flow path @rossman2020epanet.

Free chlorine also oxidizes #acr("As(III)") to arsenic(V) where both species coexist @dodd2006arsenite. The reaction consumes both reactants in a 1:1 molar ratio, and its rate depends on their concentrations. The local reaction extent also depends on contact time. Reactant concentrations and contact times depend on upstream inputs, flow-dependent dilution, mixing, and travel time @rossman2020epanet. The chemical response therefore remains coupled to the hydraulic trajectory.


=== EPANET and EPANET-MSX

EPANET is a simulator for extended-period hydraulic and water-quality analysis of pressurized pipe networks @rossman2020epanet. It solves the network equations over a sequence of hydraulic conditions and tracks head, pressure, flow, demand, tank level, and dissolved-species concentration. Reservoirs provide hydraulic boundary conditions, while time-varying junction demands influence the resulting flow distribution. Its water-quality model uses the hydraulic state to represent pipe transport and mixing at junctions, while configured reactions determine changes in concentration @rossman2020epanet. Hydraulic and water-quality calculations may be advanced at different time resolutions.

#acr("EPANET-MSX") extends this formulation to multiple interacting species @shang2023epanetmsx. For kinetic reactions, a concentration vector $bold(c)$ and a vector of kinetic parameters $bold(k)$ define a vector-valued reaction function $r(bold(c), bold(k))$, where $r$ contains the species-specific reaction rates, whose components may depend on several species. Transport and reaction can thus be simulated jointly under the same time-varying hydraulic state @shang2023epanetmsx.


== Neural Models for Sequential Data

Neural networks approximate mappings from inputs to outputs by learning from examples @goodfellow2016deep. A feedforward model receiving only the current measurement has no internal memory. Historical information can instead be supplied as a fixed input window or retained in a recurrent state. Recurrent networks use the latter approach by updating a recurrent state throughout the sequence.


=== Parameterized Models and Gradient-Based Learning

A neural network is a parameterized function $F_bold(theta)$ composed of affine transformations and nonlinear operations @goodfellow2016deep. Its parameters $bold(theta)$ are learned weights and biases, while its inputs and outputs are the values it receives and computes. In supervised learning, a loss function $cal(L)$ assigns a cost to each model prediction relative to its target. Averaging this loss over the training examples gives an empirical training objective. When this objective is differentiable with respect to $bold(theta)$, its gradient guides the parameter updates. For $N$ supervised examples $(bold(X)^(i),bold(Y)^(i))$, empirical parameter estimation can be written as @goodfellow2016deep

$
  hat(bold(theta)) = arg min_bold(theta) 1/N sum_(i=1)^N
  cal(L) lr((F_bold(theta) (bold(X)^(i)), bold(Y)^(i)), size: #125%).
$

Backpropagation applies the chain rule to differentiate this objective with respect to every parameter @goodfellow2016deep. Minibatch optimization estimates the gradient from a subset of examples instead of evaluating the complete dataset for each update. This stochastic estimate reduces the computational cost per update and permits repeated parameter updates during training.

Large gradients can destabilize optimization @pascanu2013rnn. Gradient-norm clipping rescales a gradient $bold(g)$ when $norm(bold(g))$ exceeds a chosen threshold. This limits its magnitude while preserving its direction. Clipping mitigates exploding gradients.

Model parameters, auxiliary optimizer state, and fitted preprocessing statistics are distinct from individual model inputs and outputs and from the objective function.


=== Recurrent Neural Networks and Backpropagation Through Time

A simple #acr("RNN") processes a sequence by updating a hidden state @goodfellow2016deep,

$
       bold(h)_t & = phi(bold(W)_x thin bold(x)_t + bold(W)_h thin bold(h)_(t-1) + bold(b)_h), \
  hat(bold(y))_t & = g(bold(W)_y thin bold(h)_t + bold(b)_y).
$

Here, $bold(x)_t$, $bold(h)_t$, and $hat(bold(y))_t$ are the input, hidden-state, and output vectors. The quantities $bold(W)_x$, $bold(W)_h$, and $bold(W)_y$ are weight matrices, $bold(b)_h$ and $bold(b)_y$ are bias vectors, and $phi$ and $g$ are activation functions. The hidden state $bold(h)_t$ summarizes the inputs processed through time $t$ @goodfellow2016deep. Reusing the same matrices and biases at every step imposes a common transition rule and allows the model to process sequences of different lengths.

#acr("BPTT", cap: true) unfolds the recurrence into a computation graph containing one instance of the recurrent cell per time step, where the instances share parameters @pascanu2013rnn. Gradients from later losses pass through the intervening state transitions, and their contributions to each parameter are accumulated before the optimizer update.

This backward propagation multiplies chains of transition Jacobians @pascanu2013rnn. Contracting products attenuate the contribution of early states to gradients of later losses, while amplifying products can make these gradients grow rapidly. These effects are known as vanishing and exploding gradients. Although gradient clipping mitigates the exploding case, it cannot restore long-range learning signals that have vanished @goodfellow2016deep. Gated architectures provide more direct paths for information and gradient flow.

Full #acr("BPTT") differentiates through the complete sequence @goodfellow2016deep. Truncated #acr("BPTT") instead operates on shorter consecutive fragments @pleines2022recurrent. The final recurrent state of one fragment can initialize the next and thereby preserve information in the forward pass. Treating this boundary state as constant during differentiation prevents gradients from crossing the fragment boundary. For independent episodes or trajectories, the recurrent state is reinitialized at their boundaries. Otherwise, the recurrence could carry information from one trajectory into another.


=== Long Short-Term Memory Networks

#acr("LSTM") is a gated recurrent architecture designed to preserve relevant information over long sequences @hochreiter1997lstm. Its memory cells provide paths for learning signals to persist across many time steps, aiming to help the network learn dependencies over long time intervals. A forget gate adds a learned multiplier to the previous cell state @gers2000forget. The LSTM update can be written in vector form as follows, with $tanh$ used to compute the candidate state and the hidden-state output @goodfellow2016deep:

$
  bold(i)_t & = sigma(bold(W)_i thin bold(x)_t + bold(U)_i thin bold(h)_(t-1) + bold(b)_i), \
  bold(f)_t & = sigma(bold(W)_f thin bold(x)_t + bold(U)_f thin bold(h)_(t-1) + bold(b)_f), \
  bold(o)_t & = sigma(bold(W)_o thin bold(x)_t + bold(U)_o thin bold(h)_(t-1) + bold(b)_o), \
  bold(g)_t & = tanh(bold(W)_g thin bold(x)_t + bold(U)_g thin bold(h)_(t-1) + bold(b)_g), \
  bold(c)_t & = bold(f)_t dot bold(c)_(t-1) + bold(i)_t dot bold(g)_t, \
  bold(h)_t & = bold(o)_t dot tanh(bold(c)_t).
$

Here, $bold(i)_t$, $bold(f)_t$, and $bold(o)_t$ are the input, forget, and output gate vectors, $bold(g)_t$ is the candidate cell-state vector, $bold(c)_t$ is the cell-state vector, and $bold(h)_t$ is the exposed hidden-state vector. The $bold(W)$ and $bold(U)$ quantities are weight matrices, and the $bold(b)$ quantities are bias vectors. The logistic sigmoid $sigma$ and $tanh$ act component-wise, and $dot$ denotes element-wise multiplication. The input gate controls what is written to the cell state, the forget gate what is retained, and the output gate what is exposed.

The additive cell-state update provides a direct path for retaining information @gers2000forget. Forget-gate values near one preserve the corresponding contributions from the previous cell state, whereas values near zero suppress them. The gates can therefore learn which information to retain. The hidden and cell states can provide a compact history representation for sequence forecasting and for decisions that cannot rely on the current observation alone @goodfellow2016deep @hochreiter1997lstm.


== Reinforcement Learning

In many interactive control problems, representative examples of the correct action are difficult to obtain @sutton2018reinforcement. The outcome of an interaction can nevertheless be evaluated through a reward. #acr("RL") uses this feedback to improve a policy. Because actions influence both later rewards and the states encountered next, learning must account for delayed outcomes when optimizing expected long-term return.


=== Policies, Rewards, and Returns <section:foundations:policies-rewards-returns>

A stochastic policy $pi_bold(theta) (a|s)$ defines an action distribution conditional on the available state representation @sutton2018reinforcement. During interaction, action $A_t$ affects the next-state distribution, and the environment emits an immediate reward $R_(t+1)$. For an episode ending at $T$, the discounted return from time $t$ is

$
  G_t = sum_(k=0)^(T-t-1) gamma^k thin R_(t+k+1),
$

where $gamma in [0,1]$ determines the relative weight of delayed rewards. For this episodic formulation, the objective is the expected return over trajectories induced by the policy, $J(bold(theta))=E_(tau tilde pi_bold(theta)) [G_0]$.

For continuous control, a common policy parameterization uses a Gaussian distribution with a learned mean and standard deviation @sutton2018reinforcement @schulman2017ppo. Sampling from this distribution provides exploration, while deterministic evaluation can use its mean. The mean determines the central action and the standard deviation controls the spread of sampled actions.

An action may influence rewards many steps later. Temporal credit assignment determines how these outcomes affect the earlier decisions that contributed to them @sutton2018reinforcement. For $0 < gamma < 1$, discounting reduces the influence of distant rewards without removing this dependency. The reward should be designed so that maximizing expected return promotes the intended behavior @sutton2018reinforcement.

A finite penalty discourages a constraint violation by reducing its reward, but does not exclude the violating action or outcome @rawlings2020mpc. Hard constraints instead exclude inadmissible choices from the feasible set.

Expected return is the optimization objective used to train the policy. External evaluation criteria instead assess whether the resulting behavior satisfies the scientific or operational goals. A high training return indicates success under the encoded objective, which need not capture every intended outcome @sutton2018reinforcement. Separate outcome measures are therefore required for behavior that this scalar signal does not fully represent.

Sample efficiency concerns the amount of environment interaction needed to achieve a given level of performance @sutton2018reinforcement. Learning curves compare performance as this interaction budget increases. Sample efficiency measures data requirements, whereas computational efficiency concerns processing time and resource use @ni2022recurrent.


=== Actor--Critic Learning

Actor--critic methods combine an action-selecting actor with a critic that estimates expected return @sutton2018reinforcement. For a policy $pi$, the state-value and action-value functions are

$
     V^pi (s) & = E_pi [G_t | S_t=s], \
  Q^pi (s, a) & = E_pi [G_t | S_t=s,A_t=a].
$

Their difference $A^pi (s, a)=Q^pi (s, a)-V^pi (s)$ is the advantage @schulman2016gae. It indicates whether an action is expected to perform better or worse than the policy's typical behavior in the same state. The actor can increase the probability of actions with positive estimated advantage and decrease it for actions with negative estimated advantage.

The critic approximates the value function from sampled trajectories @schulman2016gae. A one-step temporal-difference residual is

$
  delta_t = R_(t+1) + gamma thin V_bold(phi) (S_(t+1)) - V_bold(phi) (S_t).
$

The bootstrap value is zero at a true terminal state @sutton2018reinforcement. The residual compares the critic's current estimate with the observed reward plus its estimate at the successor state. Residuals from several time steps can be combined to estimate advantages @schulman2016gae.

Actor and critic may use separate feature extractors or share an intermediate representation with distinct output heads @ni2022recurrent. With otherwise matched feature extractors, sharing avoids duplicate parameters and feature computation but also lets both objectives update the same representation. Separate representations avoid this interaction between their feature-learning gradients @ni2022recurrent.


=== Proximal Policy Optimization

#acr("PPO") is an on-policy actor--critic method that alternates between data collection and optimization @schulman2017ppo. The current policy first collects a rollout, which is divided into minibatches and reused for several optimization epochs before being discarded. This reuse improves sample utilization within an update. However, large parameter updates can cause the new policy to diverge substantially from the policy that generated the rollout @schulman2017ppo.

#acr("PPO") measures this change for each sampled action using the probability ratio between the current policy and the policy that generated the rollout

$
  r_t (bold(theta)) = frac(pi_bold(theta) (A_t|S_t), pi_(bold(theta)_"old") (A_t|S_t)).
$

Given an advantage estimate $hat(A)_t$, the clipped surrogate objective is @schulman2017ppo

$
  L^"CLIP" (bold(theta)) = E_t lr(
    [min lr(
        (
          r_t (bold(theta)) thin hat(A)_t,
          op("clip") lr((r_t (bold(theta)), 1-epsilon, 1+epsilon), size: #125%) thin hat(A)_t
        ), size: #125%
      )], size: #125%
  ).
$

Here, $E_t$ denotes the empirical average over the sampled rollout steps. Once the ratio crosses the clipping interval in a direction that would improve the unclipped objective, clipping removes the incentive to move the sampled action probability farther @schulman2017ppo. It discourages abrupt policy changes but does not impose a hard constraint on every parameter or action probability.

#acr("GAE", cap: true) combines temporal-difference residuals over several future steps @schulman2016gae. For a trajectory fragment ending at $T$, its truncated form is @schulman2017ppo

$
  hat(A)_t^"GAE" (gamma,lambda)
  = sum_(l=0)^(T-t-1) (gamma lambda)^l thin delta_(t+l).
$

Small values of $lambda$ emphasize short bootstrapped estimates, which depend more strongly on the critic and often have lower variance @schulman2016gae. Larger values incorporate longer reward sequences and generally trade lower bootstrap bias for higher variance.

At the end of a rollout fragment, the environment may still be continuing @schulman2017ppo. The final temporal-difference residual then retains the critic's estimate of the successor state, whereas a true terminal state has no future return to bootstrap @sutton2018reinforcement. Treating a continuing fragment boundary as termination would omit this remaining value and can bias the advantage estimates.

The policy objective is optimized together with a regression objective for the critic @schulman2017ppo. A common combined loss is

$
  L(bold(theta), bold(phi)) & = -L^"CLIP" (bold(theta)) + c_V thin L_V (bold(phi)), \
            L_V (bold(phi)) & = E_t [(V_bold(phi) (S_t) - hat(V)_t)^2],
$

where $hat(V)_t$ is a return target held fixed during the update and $c_V$ weights the value loss. With #acr("GAE"), this target can be formed by adding the estimated advantage to the value estimate used during rollout collection @schulman2016gae. The combined loss provides optimization terms for both policy improvement and value estimation. An optional entropy bonus encourages a broader action distribution, where setting its coefficient to zero gives the objective above @schulman2017ppo.

The #acr("KL") measures the difference between two probability distributions @kullback1951information. For continuous densities $p$ and $q$ with common support,

$
  D_"KL" (p parallel q) = integral p(a) ln(p(a) / q(a)) dif a.
$

The divergence is non-negative and vanishes exactly when the distributions agree almost everywhere @kullback1951information. It is generally asymmetric, so the argument order matters. For #acr("PPO"), taking $p$ as the previous policy and $q$ as the updated policy at a given state measures change relative to the distribution that generated the actions @schulman2017ppo.

An estimated #acr("KL") divergence averaged over sampled rollout states can serve as a diagnostic and stop further policy updates when a chosen threshold is exceeded @dossa2021early. Unlike the clipped sample-wise ratio, this statistic aggregates distributional change across sampled states. Neither clipping nor this stopping rule guarantees an exact bound outside the observed rollout.


=== Recurrent Proximal Policy Optimization

Recurrent #acr("PPO") introduces a recurrent representation into the actor--critic architecture @pleines2022recurrent. At time $t$, a recurrent policy updates its memory as $bold(h)_t=f_bold(theta) (bold(h)_(t-1), bold(o)_t)$ and conditions the action distribution on $bold(h)_t$. In a partially observable environment, this memory summarizes preceding observations that may distinguish latent situations with similar current measurements @ni2022recurrent. Past actions can also inform the memory when they are included in the recurrent inputs @pleines2022recurrent.

Training must preserve the sequential dependencies on which this memory depends @pleines2022recurrent. Rollout fragments are processed as ordered sequences rather than independently shuffled transitions. Each fragment receives the recurrent state associated with its first time step. Backpropagation can stop at fragment boundaries while the forward state continues into the next fragment.

For independent episodes, the recurrent state is reset at each episode boundary @huang2022ppoimplementation. Otherwise, information from a completed trajectory could influence decisions in the next one. Applying the same reset convention during data collection, optimization, and evaluation gives the recurrent state a consistent meaning.


== Neural Time-Series Forecasting

A forecasting model approximates future values from information available at the forecast origin, which may include an observation history and known future inputs @hyndman2021forecasting. Constructing such a forecast therefore requires a strict distinction between information available at that origin and values observed only afterward. The model architecture determines how available information is processed to generate future predictions.


=== Supervised Multistep Forecasting

Supervised multistep forecasting pairs the history available at a forecast origin with the subsequent target sequence. Examples derived from simulated trajectories must obey the same information restriction as deployment, so inputs may only contain information available when the prediction is made @hyndman2021forecasting. Moving the origin along a trajectory produces multiple history--future pairs, including the overlapping examples defined above.

With $M$ target variables and $K$ future steps, the task is multivariate, multi-horizon regression. A parameterized forecaster can be written as

$
  hat(bold(Y))_t = F_bold(theta) (bold(X)_t, bold(Z)_t),
$

where $bold(X)_t$ is the observed history and $bold(Z)_t$ contains future covariates known or assumed at time $t$ @hyndman2021forecasting. If future actions affect the target, including a planned action sequence in $bold(Z)_t$ makes the forecast action-conditioned. Its output then applies to that sequence rather than to arbitrary future actions.

Observed histories, known future covariates, and prediction targets have distinct roles. Historical target values may be valid inputs because they are available at the forecast origin @hyndman2021forecasting. At prediction time, future target values are unavailable. They serve as labels for training or evaluation. Future covariates are valid inputs only if they are known in advance or explicitly assumed as part of the conditional forecast @hyndman2021forecasting.


=== Encoder--Decoder LSTMs

An encoder--decoder architecture maps an input sequence to an output sequence of possibly different length @cho2014encoderdecoder. An #acr("LSTM") encoder can compress the input history into a fixed-dimensional representation that initializes an #acr("LSTM") decoder @sutskever2014sequence. This representation provides the context for generating the future sequence.

At each forecast step, the decoder updates its recurrent state using this context and any step-specific input @goodfellow2016deep. Such inputs can include known future covariates or a planned control action. Conditioning each step on the corresponding future information allows predictions to depend on those inputs.

The decoder may predict absolute target levels or deltas between successive target values. For delta prediction, absolute predictions are reconstructed from the final observation $bold(y)_t$ by accumulating the predicted deltas,

$
  Delta bold(y)_(t+k) & = bold(y)_(t+k) - bold(y)_(t+k-1), \
   hat(bold(y))_(t+k) & = bold(y)_t + sum_(j=1)^k hat(Delta bold(y))_(t+j).
$

Delta prediction anchors the sequence at the final observed level, but an error in an early increment propagates to every later reconstructed value.

A decoder can produce the full prediction horizon without feeding its predictions back as later inputs @goodfellow2016deep. An autoregressive decoder instead conditions each step on its preceding output @bengio2015scheduled. This makes dependencies between successive forecasts explicit but allows prediction errors to accumulate through feedback. If training uses preceding ground-truth targets, inference introduces an additional mismatch because only model-generated outputs are then available.


=== Forecasting Objectives and Error Measures

#acr("MSE", cap: true) averages squared prediction errors @hyndman2006accuracy. For $N$ examples, $K$ forecast steps, and $M$ target variables, let $y_(i,k,m)$ and $hat(y)_(i,k,m)$ denote the target and prediction for example $i$, forecast step $k$, and variable $m$. Averaging over all three indices gives

$
  "MSE" = 1/(N thin K thin M)
  sum_(i=1)^N sum_(k=1)^K sum_(m=1)^M
  (hat(y)_(i,k,m)-y_(i,k,m))^2.
$

#acr("MSE") is differentiable and penalizes large residuals quadratically, making it a common regression objective @goodfellow2016deep. Weights can change the relative importance of variables or horizons. In that case, the optimized quantity is the specified weighted objective rather than an unweighted average error.

When the target variables share a common unit, #acr("MAE") and #acr("RMSE") summarize prediction errors in that unit @hyndman2006accuracy. They are defined as

#{
  import "@preview/equate:0.3.3": equate
  show: equate.with(
    sub-numbering: false,
    number-mode: "line",
  )

  set math.equation(numbering: "(1)")

  $
     "MAE" & = 1/(N thin K thin M)
             sum_(i=1)^N sum_(k=1)^K sum_(m=1)^M
             abs(hat(y)_(i,k,m)-y_(i,k,m)), \
    "RMSE" & = sqrt(
               1/(N thin K thin M)
               sum_(i=1)^N sum_(k=1)^K sum_(m=1)^M
               (hat(y)_(i,k,m)-y_(i,k,m))^2
             ).
  $
}

#acr("MAE") weights absolute residuals linearly, whereas #acr("RMSE") gives large errors more influence through squaring @hyndman2006accuracy. Targets may be normalized during training so that variables with different scales contribute comparably to optimization. To report errors in physical units, predictions and targets are first transformed back to those units. A normalized training loss and a physical-unit evaluation metric therefore have different numerical meanings.


== Predictive Control and Imitation Learning

In predictive control, a predictive model makes the expected consequences of alternative actions comparable @rawlings2020mpc. #acr("MPC") performs this comparison repeatedly over a finite horizon. If gradients through the predictive model are unavailable, sampling-based search can select the action sequence @chua2018pets. The resulting actions can also provide demonstrations for behavior cloning @ahn2023mpcimitation. A cloned policy can subsequently be fine-tuned through #acr("RL") @nair2018demonstrations.


=== Model Predictive Control

#acr("MPC") chooses a finite sequence of future actions by optimizing their predicted outcomes over a planning horizon @rawlings2020mpc. Given the current information $I_t$, a candidate control sequence $bold(U)_t = (u_t, dots, u_(t+K-1))$ produces a forecast $hat(bold(Y))_t (bold(U)_t) = (hat(bold(y))_(t+1), dots, hat(bold(y))_(t+K))$. The argument $bold(U)_t$ makes the dependence on the candidate controls explicit. We write the resulting optimization as

$
  bold(U)_t^* = arg min_(bold(U) in cal(U)^K)
  C lr((hat(bold(Y))_t (bold(U)), bold(U)), size: #125%),
$

where $C$ encodes the control cost and $cal(U)$ is the feasible action set. State or output constraints can additionally restrict the candidate sequences when the predictive model represents the required quantities @rawlings2020mpc.

In its standard discrete-time formulation, #acr("MPC") follows the receding-horizon principle, where only the first action of $bold(U)_t^*$ is applied @rawlings2020mpc. When the next observation arrives, the optimization is solved again using the updated information. Replanning reduces reliance on an open-loop sequence and lets new measurements inform subsequent predictions @rawlings2020mpc.

Control quality depends on the predictive model, planning horizon, and optimizer @rawlings2020mpc. Model error makes future trajectories uncertain. The horizon determines which delayed effects appear explicitly in the predicted trajectory. Online optimization must fit within the available computation time @ahn2023mpcimitation. Receding-horizon feedback alone does not guarantee robustness to model error @rawlings2020mpc. It also continues to optimize the specified objective, even if that objective omits a relevant outcome.


=== Cross-Entropy Method

The #acr("CEM") searches for high-performing candidate vectors by sampling and updating a proposal distribution @deboer2005cem. Applied to predictive control, it searches action sequences without differentiating through the objective or predictive model @chua2018pets. For a scalar control problem with horizon $K$, one choice is a multivariate Gaussian with mean $bold(mu) in RR^K$ and diagonal covariance matrix $bold(Sigma) in RR^(K times K)$ @botev2013cem.

Each iteration samples a finite population of candidate sequences, evaluates their objective values, and retains a fixed top fraction as the elite set @deboer2005cem. For the diagonal Gaussian, the empirical coordinate-wise means and variances of these elites determine the proposal parameters @botev2013cem. The means and standard deviations may then be smoothed. Repeated updates direct sampling toward regions favored by the evaluated elites. The algorithm returns either the final mean or the best evaluated candidate as an approximate solution @botev2013cem.

For bounded controls, candidates can be drawn from a truncated Gaussian distribution @botev2013cem. A finite population and iteration budget do not guarantee that #acr("CEM") finds the global optimum. Premature shrinking of the proposal distribution can stop exploration, motivating smoothing or the injection of additional variance @botev2013cem.


=== Behavior Cloning and Transfer Learning

Behavior cloning treats observation--action demonstrations as supervised examples @ross2011dagger. For $N$ demonstrations, a teacher produces action $a_i^"T"$ for observation or history $bold(o)_i$, where $"T"$ denotes the teacher. A student action predictor $m_bold(theta)$ learns to reproduce these actions, for example by minimizing

$
  hat(bold(theta)) = arg min_bold(theta) 1/N sum_(i=1)^N
  cal(L)_"BC" lr((m_bold(theta) (bold(o)_i), a_i^"T"), size: #125%).
$

For continuous actions, $cal(L)_"BC"$ may be a regression loss, while for discrete actions, it may be a classification loss @ross2011dagger. This objective fits the student on the demonstration distribution. Exact reproduction additionally requires that the student inputs contain the information needed to choose the demonstrated actions.

For a Gaussian policy, $m_bold(theta)$ can represent the action mean introduced in @section:foundations:policies-rewards-returns. When the standard deviation is parameterized separately, a squared-error loss on the mean does not train it.

Closed-loop use can create distribution shift @ross2011dagger. A student error may change the next state and lead to observations that were rare or absent in the teacher data. Further errors can then accumulate as the student acts outside its supervised training distribution. Low one-step imitation error is therefore insufficient evidence of closed-loop policy quality.

Behavior cloning can serve as pretraining rather than as the final policy-learning stage @nair2018demonstrations. Its parameters initialize a policy for a downstream objective, providing a starting point shaped by the demonstrations. Fine-tuning then adapts them using the downstream signal. The benefit of transferring learned features depends on their suitability for the target task @yosinski2014transfer. Task-specific features can limit that benefit. Transferring a policy also requires compatible input and action meanings, or a corresponding adaptation of its interface.
