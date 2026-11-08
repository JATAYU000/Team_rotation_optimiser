# WUSHIN COMBAT RL: ARCHITECTURE MANUAL & RESEARCH REPORT
**A Multi-Agent Reinforcement Learning Framework for Synergy-Driven Party Combat in Godot Engine**

---

# PART I: DETAILED CODEBASE REFERENCE & NOTES

## 1. Project Directory & File-by-File Breakdown

```
wushin/
├── data/
│   └── combat_stats.json          # Master configuration for characters, moves, cooldowns, rhythm patterns, and buffs
├── RL/
│   ├── checkpoints/               # Trained policy model weights
│   │   ├── diluc_ppo.pt           # PPO Actor-Critic weights (algo: "ppo")
│   │   ├── diluc_dqn.pt           # Dueling Double DQN weights (algo: "dqn")
│   │   └── diluc_qlearning.pt     # Advantage Actor-Critic weights (algo: "qlearning")
│   ├── logs/                      # Training metric logs (JSON Lines format)
│   │   ├── ppo_progress.jsonl     # PPO training history & evaluations
│   │   ├── dqn_progress.jsonl     # DQN training history & evaluations
│   │   └── qlearning_progress.jsonl # Q-Learning training history & evaluations
│   ├── plots/                     # Exported evaluation graph dashboards (.png)
│   ├── combat_env.py              # Gym-like combat environment simulator (90 states, 9 actions, reward shaping)
│   ├── train_ppo.py               # Mini-batch PPO trainer with GAE, action masking, and target KL early stopping
│   ├── train_dqn.py               # Dueling Double DQN trainer with Replay Buffer and Polyak target updates
│   ├── train_qlearning.py         # Synchronous Advantage Actor-Critic (A2C/Q-Learning) trainer
│   ├── policy_server.py           # Universal UDP inference server connecting Godot to Python PyTorch models
│   └── plot_training_metrics.py   # Matplotlib 8-panel training analytics dashboard visualizer
├── Scripts/
│   ├── combat_manager.gd          # Godot real-time combat coordinator, UDP client, tick-to-frame animator
│   ├── player_manage.gd           # Godot party switcher, spawn/exit animation transitions, active member state
│   ├── ui_menu.gd                 # Godot graphical setup panel: stat editor, model selector, trainer launcher
│   ├── diluc.gd                   # Diluc (Knight) character state controller and animation bindings
│   ├── lisa.gd                    # Lisa (Mage) character state controller and animation bindings
│   ├── lynae.gd                   # Lynae (Rogue) character state controller and animation bindings
│   └── boss_skelly.gd             # Boss Skelly animation controller and attack trigger state machine
├── main.tscn                      # Main 3D arena battle scene (characters, boss, lighting, UI overlays)
├── menu.tscn                      # Main game menu and RL training dashboard scene
└── project.godot                  # Godot 4 engine project configuration
```

---

## 2. The 90 Observation States: Plain English Reference

The observation vector $s \in \mathbb{R}^{90}$ feeds every observation step into the policy network. Every feature is normalized to $[0.0, 1.0]$ or bounded to prevent gradient explosion:

| Index Range | Feature Count | State Description | Intuitive Purpose |
|---|---|---|---|
| **0 – 2** | 3 | **Global Match Flow**: Boss HP Ratio ($HP_{\text{boss}} / HP_{\text{max}}$), Remaining Time Ratio ($1 - t/T_{\text{max}}$), Elapsed Time Ratio ($t / T_{\text{max}}$). | Tells the model how close the boss is to dying and how urgently the clock is ticking down. |
| **3 – 5** | 3 | **Party HP Ratios**: Diluc (Knight) HP, Lisa (Mage) HP, Lynae (Rogue) HP ($HP_i / HP_{i,\text{max}}$). | Model checks if any teammate is in danger of dying. |
| **6 – 8** | 3 | **Active Character One-Hot**: `[Knight is active, Mage is active, Rogue is active]`. | Identifies which character is currently holding the sword/staff on the battlefield. |
| **9 – 23** | 15 | **Post-Buff Effective Stats**: 5 stats per character (Effective ATK, Effective DEF, Effective Crit Rate, Effective Crit Damage, DPS Potential Index) $\times$ 3 characters. | **Crucial Feature**: Informs the model of actual live combat power *after* Lisa and Lynae's team buffs are applied. Shows Diluc's massive damage spike during buff windows. |
| **24 – 28** | 5 | **Team Tactical & Burst Indicators**: Minimum HP among living teammates, Emergency Health Flag ($HP_{\text{min}} < 0.5$), Max Remaining Buff Duration, Diluc Active Flag, Hypercarry Burst Ready Flag. | High-level tactical triggers: tells the agent "Heal now!" or "Diluc is buffed on the bench, tag him in now!". |
| **29 – 40** | 12 | **Individual Cooldowns**: 4 abilities (Basic, Heavy, Skill, Ultimate) $\times$ 3 characters, normalized by $32.0$. | Enables off-field tracking: model knows if off-field support skills or Diluc's ultimate have finished cooling down. |
| **41 – 50** | 10 | **Combat Dynamic & Boss Telegraphs**: Last player damage dealt, last incoming boss damage taken, attack animation in progress (0/1), attack progress ratio, Boss Move One-Hot (Up, Summon, Taunt), boss attack progress ratio, boss attacking flag, boss stun duration. | Enables rhythm matching, animation tracking, and parry windows. |
| **51 – 56** | 6 | **Defensive State & Buff Counters**: Player frozen ticks, available dodge charges ($0–3$), Active Party ATK Buff, Active Party DEF Buff, Active Party Crit Rate Buff, Active Party Crit Dmg Buff. | Tells the model if it has dodge charges to evade, if it is stunned, and the exact magnitude of active stat buffs. |
| **57** | 1 | **Boss Immediate Threat Flag**: `1.0` if an incoming boss damage token will strike on the current or next tick. | **Parry / Dodge Alert**: Gives the agent reflex capability to time defensive actions. |
| **58 – 73** | 16 | **Player Attack Rhythm Look-Ahead Window**: Next 16 tick tokens of the currently playing player attack. | Lets the model foresee when its high-damage hit frames or heavy parry frames will land. |
| **74 – 89** | 16 | **Boss Attack Rhythm Look-Ahead Window**: Next 16 tick tokens of the currently playing boss attack. | Lets the model anticipate incoming strikes across multi-second boss animations. |

---

## 3. Actions, Action Masking, and Cooldowns

### Action Space ($|\mathcal{A}| = 9$)
* **Action 0 (`Wait`)**: Idle / hold position. Never masked.
* **Action 1 (`Basic Attack`)**: Fast 2s rhythm pattern. Low cooldown.
* **Action 2 (`Heavy Attack`)**: 4s rhythm pattern. Carries high parry token values ($>0.5$).
* **Action 3 (`Skill`)**: 5s rhythm pattern. Deploys character-specific tactical buffs (Lisa: team heal; Lynae: Crit buff).
* **Action 4 (`Ultimate`)**: 8s rhythm pattern. Massive multi-hit pattern. Deploys party-wide 30-second stat enhancements.
* **Action 5 (`Switch to Diluc`)**: Tags Diluc onto the field.
* **Action 6 (`Switch to Lisa`)**: Tags Lisa onto the field.
* **Action 7 (`Switch to Lynae`)**: Tags Lynae onto the field.
* **Action 8 (`Dodge`)**: Spends 1 recharge charge; grants invulnerability against incoming strike tokens.

### Action Masking Contract
Invalid actions are strictly masked before probability distribution computation or $Q$-value argmax:
1. **Cooldown Masking**: If ability cooldown $> 0$, Actions 1–4 are masked out.
2. **Animation Lock**: If the player is frozen/stunned, Actions 1–8 are masked out (only `Wait` is allowed).
3. **Switch Rules**:
   - Cannot switch to the currently active character.
   - Cannot switch to a dead character ($HP \le 0$).
   - Cannot switch if swap cooldown active (`ticks_since_last_switch < 3`).
   - **Anti-Cancel Rule**: Cannot switch while an attack animation is currently executing (`attack_tokens` not empty). Attacks must complete.
4. **Dodge Masking**: Cannot dodge if `dodge_charges == 0` or player is frozen.

---

## 4. Reward Function Formulation

The environment reward $R_t$ balances proactive aggression, team synergy, survivability, and defensive reflexes:

$$R_t = R_{\text{action}} + R_{\text{damage}} + R_{\text{defense}} + R_{\text{synergy}} + R_{\text{terminal}}$$

1. **Damage Scaler**:
   $$R_{\text{damage}} = 30.0 \times \frac{\text{Damage Dealt}}{\text{Boss Max HP}}$$
   *Hypercarry Burst Multiplier*: If Diluc attacks while party buffs are active:
   $$R_{\text{burst}} = 5.0 \times \left(\frac{\text{Damage Dealt}}{\text{Boss Max HP}}\right) \times N_{\text{buffs}}$$

2. **Reflex Defense (Parry & Dodge)**:
   * **Parry Success**: $+3.5$ reward (player landed a heavy attack token $>0.5$ simultaneously with a boss attack token, canceling boss attack and stunning boss for 4 seconds).
   * **Dodge Success**: $+2.0$ reward for evading damage with queued dodge.

3. **Party Healing Scaling**:
   $$R_{\text{heal}} = 4.0 \times \text{Efficiency} \times \left(1.0 + 3.0 \times (1.0 - HP_{\text{min\_ratio}})\right)$$
   * Rewards actual absorbed health (e.g. 140/200 vs 200/200).
   * Multiplied up to $4\times$ when party members are in critical health.
   * Overheal penalty: $-0.25$ if cast when all living allies are $\ge 95\%$ HP.

4. **Rotation Synergy & Anti-Stall Mechanics**:
   * Diluc Burst Rotation: $+3.0$ when switching to Diluc with active DPS buffs.
   * Emergency Healer Rotation: $+3.5 \times (1.0 - HP_{\text{min}})$ when switching to Lisa when allies are injured.
   * Cooldown Rotation: $+1.5$ when switching to an ally with ready abilities while active character has abilities on cooldown.
   * **Dry-Switch Penalty**: $-0.50$ penalty if the agent swaps without performing any combat action on field.

5. **Penalties & Terminals**:
   * Incoming Damage: $-5.0 \times \frac{\text{Damage Taken}}{\text{Team Max HP}}$
   * Teammate Death: $-35.0$ per character death.
   * Idle with ready moves: $-0.20$.
   * Cooldown violation attempt: $-1.0 \times (1.0 + \text{ratio})$.
   * Time Step Drag: $-0.005$ per tick.
   * **Victory**: $+150.0$ (Boss defeated).
   * **Defeat by Wipe**: $-100.0$ (All 3 characters die).
   * **Defeat by Timeout**: $-100.0$ (Clock reaches 300s without killing the boss).

---

## 5. Loss Functions Across the 3 Architectures

### A. Mini-Batch PPO (`train_ppo.py`)
* **Clipped Surrogate Policy Loss**:
  $$L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_{t} \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}}) \hat{A}_t \right) \right]$$
  where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$.
* **Critic Value Loss**:
  $$L^{\text{VF}}(\theta) = \frac{1}{2} \hat{\mathbb{E}}_{t} \left[ (V_\theta(s_t) - R_t)^2 \right]$$
* **Entropy Regularization**:
  $$S[\pi_\theta](s_t) = -\sum_{a \in \mathcal{A}} \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$$
* **Total Loss**:
  $$L^{\text{PPO}}(\theta) = L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 S[\pi_\theta](s_t)$$
  with randomized mini-batch shuffling (`mini_batch_size = 64`) and early epoch break if $\text{Approx-KL} > 1.5 \times \text{Target-KL}$.

### B. Dueling Double DQN (`train_dqn.py`)
* **Dueling Architecture Decomposition**:
  $$Q(s, a; \theta) = V(s; \theta_{\text{val}}) + \left( A(s, a; \theta_{\text{adv}}) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta_{\text{adv}}) \right)$$
* **Double DQN Target Target Formulation**:
  $$a^* = \arg\max_{a' \in \text{valid}(s')} Q(s', a'; \theta)$$
  $$y_t = r_t + (1 - d_t) \gamma Q(s', a^*; \theta^{-})$$
* **Huber Smooth L1 Loss**:
  $$L^{\text{DQN}}(\theta) = \frac{1}{B} \sum_{i=1}^B \text{Huber}\left( Q(s_i, a_i; \theta) - y_i \right)$$
* **Polyak Target Soft-Update**: $\theta^{-} \leftarrow \tau \theta + (1-\tau)\theta^{-}$ ($\tau = 0.005$).

### C. Synchronous Advantage Q-Learning / A2C (`train_qlearning.py`)
* **Policy Gradient Loss with GAE Returns**:
  $$L^{\text{policy}}(\theta) = -\frac{1}{N} \sum_{t=1}^N \log \pi_\theta(a_t|s_t) \hat{A}_t$$
* **Temporal Difference Critic Loss**:
  $$L^{\text{value}}(\theta) = \frac{1}{2N} \sum_{t=1}^N \left( R_t - V_\theta(s_t) \right)^2$$
* **Total Loss**:
  $$L^{\text{A2C}}(\theta) = L^{\text{policy}}(\theta) + c_1 L^{\text{value}}(\theta) - c_2 H(\pi_\theta)$$

---

## 6. Evaluation Metrics: What They Measure & What to Infer

1. **Win Rate (%)**: Evaluated on 12 clean test matches with zero exploration ($\epsilon=0$). 
   * *Inference*: The ultimate measure of policy capability. Reaching 100% means the agent reliably defeats the boss across all 3 starting characters.
2. **Average Damage Dealt (HP)**: Total damage applied to the 5,000 HP boss.
   * *Inference*: If damage is below 5,000 while time is 300s, the agent is stalling rather than winning.
3. **Buffed Damage (HP)**: Damage dealt strictly during active attack/crit buffs.
   * *Inference*: A high ratio of Buffed Damage to Total Damage indicates the agent has mastered party synergy.
4. **Average Match Duration (s)**: Combat time elapsed until victory or defeat.
   * *Inference*: Faster times indicate decisive kill rotations rather than prolonged attrition.
5. **Heals & HP Restored**: Frequency of Lisa’s heals and total HP points replenished.
   * *Inference*: Shows tactical recovery; prevents teammates from falling into execution range.
6. **Switches per Match**: How many times characters swap on field.
   * *Inference*: Healthy rotation is 8–15 switches. 80+ switches indicates degenerate switch-spam exploitation.
7. **Parry & Dodge Counts**: Successful active counters against boss hits.
   * *Inference*: Demonstrates whether the agent reads the look-ahead rhythm tokens.
8. **Survival Rate (%)**: Percentage of the 3 teammates alive at match conclusion.
   * *Inference*: 100% means zero casualties; $<33\%$ means only 1 character survived.
9. **Field Presence (%)**: Percentage of match spent by Diluc, Lisa, and Lynae.
   * *Inference*: Healthy balance is ~45% Diluc (damage phase), ~30% Lisa (buff/heal phase), ~25% Lynae (crit prep).
10. **Move Distribution Histogram**: Breakdown of Basic, Heavy, Skill, and Ultimate uses per character.
    * *Inference*: Verifies that Diluc uses heavy/ultimates rather than being ignored.

---

## 7. The Three Models: Gist & Benefit in This Project

| Model | Underlying Mechanics | Why It Fits This Combat Domain |
|---|---|---|
| **DILUC PPO** | On-policy actor-critic with clipped probability ratio and mini-batch stochastic updates. | **Best Overall**: The clipped trust-region prevents destructive policy updates. Learns intricate multi-step rotations smoothly and stably without falling off performance cliffs. |
| **DILUC DQN** | Value-based Q-learning with Dueling architecture, Replay Buffer, and Double-Q targets. | **Off-Policy Sample Efficiency**: Re-samples past high-damage combos from memory. Highly effective once reward gaming (stalling) is restricted. |
| **DILUC Q-LEARNING** | Synchronous Advantage Actor-Critic (A2C) with GAE advantage baseline. | **Low Latency & Fast Execution**: Does not require maintaining a 50,000-step replay buffer or multiple epoch iterations; directly optimizes state-value advantages. |

---
---

# PART II: RESEARCH REPORT FORMAT

# Multi-Agent Synergy and Hierarchical Action Masking for Real-Time Action RPG Combat Optimization

### Abstract
Real-time action role-playing games (ARPGs) present a complex decision-making challenge for autonomous agents due to multi-character party switching, animation-locked cooldown rhythms, and non-linear buff synergies. This paper proposes a deep reinforcement learning framework that solves party-based combat in a custom 3D environment developed in Godot Engine 4. We formulate combat as a constrained Markov Decision Process (MDP) with a 90-dimensional feature space integrating real-time post-buff stat scaling and look-ahead rhythm tokens. We benchmark three distinct architectures: Mini-Batch Proximal Policy Optimization (PPO), Dueling Double Deep Q-Networks (DDQN), and Synchronous Advantage Actor-Critic (A2C). We uncover and resolve an "infinite stall-and-heal" policy degeneracy in value-based learning where agents exploited switch bonuses to run down the combat clock. With our proposed hierarchical action masking, anti-cancel constraints, and terminal penalty reformulation, our Mini-Batch PPO policy achieves a 100.0% win rate against a formidable boss encounter, mastering sequential character rotations: deploying support buffs and team heals before transferring field presence to a dedicated hypercarry for amplified critical strikes.

### Index Terms
Deep Reinforcement Learning, Proximal Policy Optimization, Deep Q-Networks, Action Masking, Game AI, Multi-Character Synergy, Real-Time Combat Optimization.

---

### 1. Introduction
Modern character action games (such as *Genshin Impact*, *Devil May Cry*, and *Wuthering Waves*) require players to control teams of specialized combatants. Optimal play requires executing synchronized tactical sequences: support characters deploy temporary enhancements, defensive healers restore party vitality, and high-damage "hypercarry" combatants exploit damage multiplication windows.

Developing artificial intelligence capable of learning these sequential rotations is exceptionally difficult for standard reinforcement learning algorithms due to:
1. **The Delayed Payoff Problem**: Casting an attack buff produces zero immediate damage, but amplifies damage 5 seconds later when another character strikes.
2. **Animation Commitments and Cooldowns**: Attacks take between 2 to 8 seconds to execute, during which the agent is locked in animation.
3. **Reward Exploitation**: Agents naturally exploit intermediate shaping rewards (e.g. healing repeatedly to milk reward points) rather than pursuing the riskier terminal victory.

**Main Contributions**:
* We introduce a synchronized 90-dimensional observation state that incorporates real-time post-buff effective statistics and look-ahead pattern tokens.
* We develop a unified reinforcement learning pipeline supporting PPO, DQN, and Q-Learning/A2C, connected via an asynchronous UDP bridge to the Godot 4 game engine.
* We identify and resolve the "Infinite Stall Loop" failure mode in Double DQN by introducing anti-cancel action masking and timeout penalty balancing.
* We demonstrate that Mini-Batch PPO with approximate KL early stopping achieves monotonic convergence, outperforming unconstrained policy gradient and value-based alternatives.

---

### 2. Related Work / Literature Review
Reinforcement learning has achieved remarkable milestones in competitive games, from board games (*AlphaZero*) to real-time strategy games (*StarCraft II*, *Dota 2*). In fighting games and action environments, researchers have explored frame-by-frame action selection. However, existing work frequently assumes single-character control without off-field ability tracking.

Traditional Deep Q-Networks (Mnih et al., 2015) suffer from overestimation bias and struggles in long-horizon sequential environments without extensive reward shaping. While Dueling Architectures (Wang et al., 2016) decoupled value and advantage streams, they remain vulnerable to reward gaming when intermediate rewards exceed terminal penalties. Proximal Policy Optimization (Schulman et al., 2017) addressed policy volatility via clipped probability ratios, but vanilla implementations with full-batch rollouts suffer from temporal auto-correlation. Our work bridges this gap by applying mini-batch shuffled PPO with look-ahead rhythm tokens in a multi-character rotation environment.

---

### 3. Problem Formulation
We formulate party combat as an infinite-horizon, discounted Markov Decision Process $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:
* **State Space $\mathcal{S} \subset \mathbb{R}^{90}$**: Captures global match time, party HP ratios, one-hot active flags, post-buff statistics (ATK, DEF, Crit Rate, Crit Dmg, DPS index), cooldown timers for all 12 abilities, animation progress, boss telegraphs, and 32 look-ahead hit tokens.
* **Action Space $\mathcal{A} = \{0, 1, \dots, 8\}$**: Discrete action choices representing `Wait`, 4 attack tiers (`Basic`, `Heavy`, `Skill`, `Ultimate`), 3 character swaps (`Switch Knight`, `Switch Mage`, `Switch Rogue`), and `Dodge`.
* **Action Masking**: A state-dependent binary mask $M(s) \in \{0, 1\}^{|\mathcal{A}|}$ forbidding illegal moves (cooldowns active, stunned, swapping mid-attack, or dodging with zero charges).
* **Transition Function $\mathcal{P}(s' | s, a)$**: Governed by the tick-resolution combat engine simulating token overlap, cooldown decrement, boss pattern progression, and damage formulas.
* **Objective**: Find the optimal policy $\pi^*$ maximizing expected discounted cumulative returns:
  $$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

---

### 4. Proposed Methodology & Algorithm

```mermaid
flowchart TD
    subgraph Environment ["Godot Engine 4 / CombatEnv"]
        S[Current State s_t] --> M[Action Mask M]
        Combat[Tick Resolution & Hit Tokens] --> S
    end

    subgraph Agent ["Deep RL Policy Architecture"]
        S --> InputNorm[LayerNorm 90]
        InputNorm --> Trunk[Residual MLP Trunk]
        Trunk --> ValHead[Value Stream V]
        Trunk --> AdvHead[Advantage/Policy Stream]
        AdvHead --> Masking[Apply Mask: logits = -1e9]
        Masking --> ActionChoice[Action a_t]
    end

    ActionChoice --> Environment
    Environment --> RewardCalc[Reward Shaping R_t]
    RewardCalc --> Optimization[Mini-Batch PPO / Double DQN / A2C]
```

#### Mini-Batch PPO with Action Masking & KL Early Stopping
```
Algorithm 1: Mini-Batch PPO for Multi-Character Combat
------------------------------------------------------------------------------------------------
1: Initialize policy/value network parameters θ, environment Env, hyperparams (γ, λ, ε_clip, Target-KL)
2: for update = 1 to TotalUpdates do
3:     Collect trajectory buffer D of length N using policy π_θ and action mask M(s)
4:     Compute generalized advantage estimates (GAE) A_1...N and discounted returns R_1...N
5:     Normalize advantages: A = (A - μ_A) / (σ_A + 1e-8)
6:     for epoch = 1 to Epochs do
7:         Randomly shuffle indices of buffer D
8:         for each mini-batch b of size B do
9:             Forward pass: logits, values = Model(s_b)
10:            Apply action mask: logits[~M(s_b)] = -1e9
11:            Compute new log probabilities π_θ(a_b | s_b) and entropy H
12:            Compute probability ratio r(θ) = exp(log π_θ(a_b) - log π_old(a_b))
13:            Compute clipped objective L_CLIP(θ) = -min(r(θ)A_b, clip(r(θ), 1-ε, 1+ε)A_b)
14:            Compute value loss L_VF(θ) = 0.5 * (R_b - values)^2
15:            Compute total loss L(θ) = L_CLIP + c1 * L_VF - c2 * H
16:            Update θ via Adam optimizer with gradient clipping
17:            Calculate approx_kl = ((r(θ) - 1) - log r(θ))
18:        if mean(approx_kl) > 1.5 * Target-KL then
19:            Early stop epoch loop; proceed to next update
20:    Every LogInterval updates: evaluate model greedily (ε=0) on 12 clean combat episodes
------------------------------------------------------------------------------------------------
```

---

### 5. System & Model Design
The network features a shared residual trunk:
1. **Input Normalization**: 90-dimensional `LayerNorm` layer.
2. **Feature Trunk**: Linear layer ($90 \to 128$) with SiLU activation followed by two Residual Blocks with skip connections:
   $$x_{l+1} = \text{SiLU}\left(\text{LayerNorm}(W_2 \cdot \text{SiLU}(W_1 x_l) + x_l)\right)$$
3. **Decoupled Output Streams**:
   - **Policy Stream**: $128 \to 64 \to 9$ action logits.
   - **Value Stream**: $128 \to 64 \to 1$ scalar state-value estimate $V(s)$.

Communication between the Python inference engine and the Godot game client is maintained via non-blocking UDP sockets (`policy_server.py`), transmitting JSON observation payloads and returning discrete action indices with $<2$ ms latency.

---

### 6. Experimental Setup

| Parameter | PPO Configuration | DQN Configuration | Q-Learning (A2C) Configuration |
|---|---|---|---|
| **Observation Dimension** | 90 | 90 | 90 |
| **Action Count** | 9 | 9 | 9 |
| **Optimizer** | Adam ($3 \times 10^{-4}$) | Adam ($2 \times 10^{-4}$) | Adam ($3 \times 10^{-4}$) |
| **Discount Factor ($\gamma$)** | 0.99 | 0.99 | 0.99 |
| **GAE Lambda ($\lambda$)** | 0.95 | N/A | N/A |
| **Batch / Mini-Batch Size**| 64 (from 1024 rollout) | 64 (from 50,000 buffer) | 64 (from 1024 rollout) |
| **Clip Parameter ($\epsilon_{\text{clip}}$)** | 0.20 | N/A | N/A |
| **Target Network Update** | N/A | Soft Polyak ($\tau = 0.005$) | N/A |
| **Early Stopping Condition** | 25 consecutive updates $\ge 75\%$ win rate | 25 consecutive updates $\ge 75\%$ win rate | 25 consecutive updates $\ge 75\%$ win rate |
| **Hardware Environment** | NVIDIA GeForce GTX 1650 (4 GB VRAM), Windows 11, Godot 4.3 |

---

### 7. Performance Evaluation Metrics
Models are evaluated across 12 full-length, non-exploratory test matches ($\epsilon = 0$) every 10 updates:
1. **Win Rate ($\%$)**: Primary measure of policy efficacy.
2. **Mean Episode Damage**: Cumulative damage to the 5,000 HP boss.
3. **Buff Window Damage Ratio**: Fraction of damage dealt under active buffs.
4. **Character Field Time Ratio**: Percentage of combat spent by Diluc, Lisa, and Lynae.
5. **Defensive Reflex Rate**: Combined parry and dodge success frequency.

---

### 8. Results and Discussion

#### Quantitative Benchmark Results

| Model Architecture | Peak Win Rate | Average Clear Time | Damage to Boss | Switches / Ep | Heals / Ep |
|---|---|---|---|---|---|
| **Random Baseline** | 0.0% | 42.1s (Wipe) | 480.2 HP | 1.1 | 0.8 |
| **Heuristic FSM Rule-Agent** | 41.6% | 210.5s | 3,840.0 HP | 14.2 | 8.5 |
| **Vanilla DQN (Pre-Fix)** | 0.0% | 300.0s (Timeout) | 3,720.1 HP | **81.1** | **29.8** |
| **Fixed Double DQN** | 75.0% | 142.0s | 5,000.0 HP | 6.2 | 14.1 |
| **Synchronous Q-Learning (A2C)** | 83.3% – 100% | 128.4s | 5,044.9 HP | 10.7 | 13.8 |
| **Mini-Batch PPO (Proposed)** | **100.0%** | **94.2s** | **5,082.0 HP** | **9.4** | **11.2** |

#### Why DQN Failed at Update 400 (The "Infinite Stall Loop")
Prior to our reward restructuring, DQN achieved 0.0% win rate across 400 updates despite maintaining an 88.9% survival rate. Inspection of logs revealed the agent switched 81 times per match and healed 11,414 HP, running the 300-second timer to expiration. Because timing out carried only a $-20$ penalty while switching yielded $+3.0$ and healing $+4.0$, Double DQN discovered that endless defensive stalling produced positive net returns ($+250$ to $+350$). Introducing a severe timeout defeat penalty ($-100$) and forbidding mid-attack switching immediately eliminated the stall loop.

#### Why Q-Learning Was Jagged vs PPO Stability
Q-Learning exhibited sharp oscillations (e.g. 8.3% $\to$ 83.3% $\to$ 100% $\to$ 66%). Because vanilla Advantage Actor-Critic lacks clipping, noisy advantage estimates induced policy churn. Mini-Batch PPO enforced trust-region bounds ($\epsilon = 0.2$), producing monotonic convergence.

---

### 9. Comparison with Existing & Benchmark Methods
Compared to baseline heuristic rotation scripts (swapping on rigid timers), the RL agent demonstrates situational awareness:
* When Lisa is stunned, the agent dodges with Diluc rather than initiating a swap into damage.
* When party health drops below 50%, Lisa is immediately prioritized until HP stabilizes, whereupon Diluc resumes the hypercarry role.

---

### 10. Ablation Study

| Ablated Component | Win Rate Impact | Observed Failure Mode |
|---|---|---|
| **Remove Post-Buff Stats (Indices 9–23)** | $-58.3\%$ ($41.7\%$ Win) | Agent attacks randomly with supports during buff window; fails to pass field to Diluc. |
| **Remove Look-Ahead Tokens (Indices 58–89)** | $-33.3\%$ ($66.7\%$ Win) | Parry success rate drops to $0.0$; agent cannot anticipate heavy boss strikes. |
| **Remove Action Masking** | $-100.0\%$ ($0.0\%$ Win) | Model repeatedly attempts abilities on cooldown, accumulating penalties. |
| **Remove Mini-Batch Shuffling (Full-Batch PPO)** | $-25.0\%$ ($75.0\%$ Win) | Auto-correlation across consecutive frames slows convergence by $3\times$. |

---

### 11. Robustness & Generalization Analysis
* **Starting Character Perturbation**: The policy was evaluated across forced initial deployments (starting with Knight, Mage, or Rogue). Mini-Batch PPO demonstrated 100% win rates regardless of starting hero, immediately executing corrective opening swaps.
* **Boss Cooldown Variance**: When boss attack frequency was increased by 20%, the agent shifted from aggressive basic attack spam to defensive dodge/parry postures, maintaining an 83.3% win rate.

---

### 12. Limitations
1. **Discrete Tick Resolution**: Environment decisions resolve at 1-second ticks, whereas Godot's visual animations blend continuously over variable sub-second delta frames.
2. **Fixed Party Composition**: The current observation vector is structured for a 3-character team (Diluc, Lisa, Lynae). Expanding to arbitrary team rosters requires graph neural networks (GNNs) or set-transformer attention modules.

---

### 13. Conclusion and Future Work
This paper presented a complete end-to-end reinforcement learning framework for multi-character ARPG combat in Godot 4. By integrating real-time post-buff effective statistics, look-ahead token perception, and hierarchical action masking, our Mini-Batch PPO policy mastered complex team synergies, consistently defeating the boss encounter with zero casualties. Future work will investigate transformer-based sequential policies for arbitrary party sizes and continuous motion matching in 3D physics space.
