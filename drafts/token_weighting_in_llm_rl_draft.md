# Every Token Gets a Weight: From PPO to OPSD in LLM RL

Draft status: working markdown draft, not yet a Quarto post.

Reference style target: use the compact formula-first exposition from [State of RL for reasoning LLMs](https://aweers.de/blog/2026/rl-for-llms/) as a model, especially its treatment of PPO, GRPO, DAPO, and CISPO as small but consequential changes to the same policy-gradient template. Keep implementation-mismatch material as an appendix thread, with references such as IcePop, MiMo, verl rollout correction, and related training-inference mismatch notes.

## Working Thesis

This post argues that many modern LLM RL algorithms can be read as answers to one question:

> For this sampled token, how much should the gradient count?

The actor update often has the same skeleton:

$$
\mathcal{L}_{\text{actor}}
=
-\sum_{i,t}
m_{i,t}\,
w_{i,t}\,
\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
$$

where:

- $m_{i,t}$ says whether the token participates.
- $w_{i,t}$ says how much the token counts.
- $\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t})$ is the trainable log-prob handle.

The post can explain each method by decomposing $w_{i,t}$:

$$
w_{i,t}
\approx
\text{policy-ratio control}
\times
\text{reward advantage}
\times
\text{normalization}
\times
\text{teacher or trust gate}.
$$

The main post should move through five public-facing layers:

1. **Thesis and lens:** every token gets a weight.
2. **Minimum policy-gradient background:** why token log-probs are the update surface, why PPO reuses sampled data, and why the first importance ratio appears.
3. **PPO to CISPO:** how policy-gradient weights are built from reward advantage, policy ratios, normalization, token-level aggregation, and clipped importance weights.
4. **SDPO/OPSD teacher weighting:** how teacher signals enter as distillation losses, advantage-style gradient interpretations, routing, or reward-anchored token multipliers.
5. **Conditional teacher trust:** why the newest OPD/OPSD variants increasingly ask whether a teacher signal should be trusted, purified, routed, delayed, or expressed through another interface.

The appendix carries the longer machinery: full policy-gradient derivation, PPO minibatch/microbatch details, clipping cases, loss-value interpretation, rollout/training mismatch, async policy versions, training-inference importance sampling, and token discard.

## Why This Post Is Different From The Survey

The survey asks: where does the teacher advantage come from, and what optimization role does it play?

This blog asks a lower-level optimization question: once we have a sampled response, what scalar or mask multiplies each token's gradient?

That lets the post talk naturally about:

- PPO's current/old policy ratio as an importance-weighted token update.
- GRPO's group-relative advantage as a critic-free response-level weight.
- RLOO, Dr. GRPO, and REINFORCE++ as changes to baselines and normalization.
- DAPO as token-level aggregation plus asymmetric clipping.
- CISPO as clipped importance sampling weight with a live log-prob gradient path.
- SDPO, SRPO, and RLSD as ways to inject teacher information into RL token weights.
- Recent OPD/OPSD variants as conditional-trust designs for deciding when the teacher signal should count.

The post should avoid private experiment details. Internal results can inform the intuition, but the publishable narrative should rely on public papers, public blogs/docs, and high-level engineering patterns.

## Proposed Structure

### 1. Every Token Gets A Weight

Start with the thesis rather than with PPO mechanics.

For a sampled response $y_i=(y_{i,1},\ldots,y_{i,T})$, actor training usually touches the model through token log-probs:

$$
\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

A useful generic loss is:

$$
\mathcal L_{\text{actor}}
=
-\sum_{i,t}
m_{i,t}\,
w_{i,t}\,
\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

Interpretation:

- $m_{i,t}$ is the mask: padding, response-only masking, routing, or token selection.
- $w_{i,t}$ is the token's effective update weight.
- The trainable object is the log-probability of the sampled token under the current actor.

This is the lens for the entire post:

```text
PPO      changes the policy-ratio control.
GRPO     changes the advantage estimator.
DAPO     changes token aggregation, clipping, and sample filtering.
CISPO    changes how the ratio enters the gradient path.
SDPO     adds dense teacher signal.
SRPO     routes where teacher signal is allowed.
RLSD     keeps reward direction but lets teacher gaps scale token magnitude.
Recent OPD/OPSD methods decide when teacher-derived weights should be trusted.
```

Suggested paragraph:

> When people describe LLM reinforcement learning, they usually talk about rewards: did the answer pass the unit tests, solve the math problem, follow the instruction, or satisfy the verifier? But inside the actor update, the object that actually touches the model is smaller and more mechanical. It is a scalar multiplying a token log-probability. PPO, GRPO, DAPO, CISPO, and reward-anchored OPSD can all be read as different ways of constructing that scalar.

### 2. The Minimum Policy-Gradient Background

This section should be compact. It should give readers enough machinery to understand PPO-to-CISPO without swallowing the opening.

#### 2.1 From Reward To A Token Gradient

The true RL objective is expected reward:

$$
J(\theta)
=
\mathbb E_{y\sim \pi_\theta}
\left[
R(y)
\right].
$$

The policy-gradient trick gives:

$$
\nabla_\theta J(\theta)
=
\mathbb E_{y\sim \pi_\theta}
\left[
R(y)\nabla_\theta \log \pi_\theta(y)
\right].
$$

For an autoregressive LLM:

$$
\log \pi_\theta(y)
=
\sum_t
\log \pi_\theta(y_t\mid x,y_{<t}).
$$

So the update surface is a sum of token-logprob gradients:

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\sum_t
R(y)
\nabla_\theta \log \pi_\theta(y_t\mid x,y_{<t})
\right].
$$

The key point:

> RL for LLMs begins as reward-weighted learning on the model's own sampled tokens.

#### 2.2 Why PPO Reuses Rollouts

Classic REINFORCE can be described as:

```text
generate rollout with current policy
score rollout
update once
discard rollout
generate again
```

PPO changes the amortization:

```text
freeze policy as pi_old
generate rollout batch with pi_old
score rollout batch
compute advantages
train on the same rollout batch for K minibatch epochs
discard rollout batch
generate a fresh rollout batch
```

The reason is practical: rollout generation and reward evaluation are expensive. In LLM RL, generation may involve many samples per prompt, code execution, tool calls, environment interaction, verifier calls, or judge-model calls. PPO extracts several optimizer steps from the same expensive rollout batch before paying for new rollouts.

During these PPO epochs, the data record stays fixed:

```text
prompt / state
sampled response tokens
old logprobs under pi_old
reward
advantage
response mask
```

The reward and advantage usually do not change inside the PPO epochs. What changes is the current actor's log-probability:

$$
\log \pi_\theta(y_t\mid s_t).
$$

That is enough to produce a different gradient on each optimizer step.

#### 2.3 The First Importance Weight

Because PPO trains a changing actor on data generated by $\pi_{\text{old}}$, it introduces:

$$
\rho_t(\theta)
=
\frac{\pi_\theta(y_t\mid s_t)}
{\pi_{\text{old}}(y_t\mid s_t)}
=
\exp
\left(
\log \pi_\theta(y_t\mid s_t)
-
\log \pi_{\text{old}}(y_t\mid s_t)
\right).
$$

At the start of PPO training:

$$
\pi_\theta=\pi_{\text{old}},
\qquad
\rho_t(\theta)=1.
$$

After the first optimizer step:

$$
\pi_\theta\ne\pi_{\text{old}},
$$

but the rollout batch was still sampled from $\pi_{\text{old}}$.

This is PPO's intended "slightly off-policy" mismatch. It is not a systems bug. It is the data-reuse trick PPO was designed to control.

The main-text version can end here, because the next section uses $\rho_t$ directly. More detailed questions such as minibatch vs microbatch, exact clipping cases, and rollout/training mismatch can move to the appendix.

### 3. Policy-Gradient Weights: From PPO To CISPO

This is the main algorithmic lineage:

```text
PPO -> GRPO -> RLOO / Dr. GRPO / REINFORCE++ -> DAPO -> CISPO
```

The shared question is: how should sampled-token updates be weighted before any teacher signal is added?

In this section, the effective token weight has the rough form:

$$
w_{i,t}
=
f(
\hat A_i,
\rho_{i,t},
\text{clip},
\text{normalization}
).
$$

Here $\hat A_i$ is reward-derived, while $\rho_{i,t}$ is the current/old policy ratio. So this is not purely "reward-side"; it is the policy-gradient weight created by combining reward direction with importance-ratio control.

#### 3.1 PPO: Ratio Times Advantage

PPO maximizes the clipped surrogate objective:

$$
J^{\text{PPO}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\rho_t(\theta)\hat A_t,\,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

The actor loss is the negative:

$$
\mathcal L^{\text{PPO}}
=
-J^{\text{PPO}}.
$$

For the token-weight view:

$$
w_t^{\text{PPO}}
\approx
\rho_t(\theta)\hat A_t,
$$

unless clipping removes or caps the update.

Intuition:

- $\hat A_t$ says whether the sampled token came from a good or bad trajectory.
- $\rho_t$ says how far the current actor has moved from the actor that generated the token.
- clipping prevents old data from pushing the current actor too far in the reward-improving direction.

Suggested prose:

> PPO is not just "RL with clipping." It is an importance-weighted token update with a trust-region mask.

#### 3.2 GRPO: Same Ratio, Critic-Free Advantage

GRPO keeps the PPO-style ratio but replaces the learned value-model advantage with a group-relative advantage.

For prompt $x$, sample $G$ responses with rewards $\{r_1,\ldots,r_G\}$:

$$
\hat A_i
=
\frac{r_i-\mu_G}{\sigma_G+\epsilon},
\qquad
\mu_G=\frac{1}{G}\sum_j r_j.
$$

The advantage is response-level, then broadcast over tokens:

$$
\hat A_{i,t}=\hat A_i.
$$

The token weight becomes:

$$
w_{i,t}^{\text{GRPO}}
\approx
\rho_{i,t}(\theta)\,\hat A_i.
$$

This removes the critic, but token-level credit assignment is still coarse: every token in the response inherits the same group-relative reward direction.

#### 3.3 RLOO, Dr. GRPO, And REINFORCE++: Baselines And Normalization Are Weights Too

This should be a short bridge, not a long survey.

RLOO uses the reward of other samples for the same prompt as the baseline:

$$
\hat A_i
=
r_i
-
\frac{1}{K-1}\sum_{j\ne i}r_j.
$$

Dr. GRPO removes standard-deviation normalization and changes loss normalization to reduce length bias:

$$
\hat A_i = r_i-\mu_G.
$$

REINFORCE++ and related critic-free variants argue that much of PPO's machinery is not always needed in LLM post-training, but normalization and stability tricks remain important.

Suggested prose:

> These methods are easy to dismiss as bookkeeping. They are not. In long-form generation, dividing by sequence length, group standard deviation, or batch token count changes which prompts and which tokens dominate the gradient.

#### 3.4 DAPO: Token-Level Aggregation And Asymmetric Ratio Clipping

DAPO makes the token-level nature of the update explicit:

$$
J^{\text{DAPO}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{\sum_i |y_i|}
\sum_i\sum_t
\min\left(
\rho_{i,t}(\theta)\hat A_i,\,
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\hat A_i
\right)
\right].
$$

Why it matters for the weight story:

- Token-level aggregation changes which tokens dominate the batch update.
- Clip-Higher relaxes the upper ratio bound, so rare but useful positive-advantage tokens are not clipped too early.
- Dynamic sampling filters prompt groups with all-correct or all-wrong outcomes, increasing the fraction of prompts with useful nonzero relative signal.
- Overlong reward shaping changes the scalar reward before it becomes an advantage.

Suggested prose:

> DAPO is where the token-weighting view becomes unavoidable. The algorithm is not only changing the advantage; it is changing the denominator, the clipping asymmetry, and the set of prompt groups allowed to contribute gradient.

#### 3.5 CISPO: Clip The Weight, Keep The Gradient Path

PPO clipping can behave like a hard gradient mask. CISPO changes the role of clipping:

$$
\hat \rho_t(\theta)
=
\operatorname{clip}\left(
\rho_t(\theta),
1-\epsilon_l,
1+\epsilon_h
\right).
$$

Then:

$$
J^{\text{CISPO}}(\theta)
=
\mathbb{E}
\left[
\operatorname{sg}\!\left(\hat\rho_t(\theta)\right)
\hat A_t
\log \pi_\theta(y_t\mid s_t)
\right].
$$

Now the effective token weight is:

$$
w_t^{\text{CISPO}}
=
\operatorname{sg}\!\left(\hat\rho_t(\theta)\right)\hat A_t.
$$

The gradient still flows through $\log \pi_\theta$ even when the importance weight is clipped.

Suggested prose:

> PPO says: if the ratio has moved too far, stop trusting this token's update. CISPO says: keep the update, but cap how much the ratio can scale it.

This is a natural bridge to RLSD, because both CISPO and RLSD separate "the quantity used as a weight" from "the trainable log-prob path."

### 4. Teacher-Augmented Weights: SDPO/OPSD

This section starts after the reader understands the baseline policy-gradient weight from Section 3.

The key transition is:

```text
PPO-to-CISPO builds the policy-gradient token weight.
SDPO/OPSD adds teacher-derived token information.
RLSD and SRPO make teacher augmentation safer by separating direction, magnitude, and routing.
```

In this section, the effective token weight becomes:

$$
w_{i,t}
=
f(
\hat A_i,
\rho_{i,t},
\text{clip},
\text{normalization},
\text{teacher signal}
).
$$

The point is not that teacher signal has one universal role. In SDPO it can replace the reward-derived advantage with a teacher-defined objective; in RLSD and SRPO it more often augments, rescales, or routes a reward-grounded policy-gradient update.

#### 4.1 Two Views Of Teacher Signal: Loss Form vs. Update Interpretation

A small bridge is needed here because distillation papers often start from a loss, not from an advantage. These are not equivalent formulations. The distillation loss is the objective being optimized; the advantage-style form is usually an interpretation of the resulting gradient, or an implementation interface for reusing an RL training scaffold.

In Section 3, every method had a policy-gradient form:

$$
\mathcal L_{\text{PG}}
\approx
-
\sum_{i,t}
m_{i,t}
w_{i,t}
\log \pi_\theta(y_{i,t}\mid s_{i,t}).
$$

The trainable term is the sampled token log-prob, and $w_{i,t}$ decides how much that sampled token counts.

Distillation starts from a different-looking object:

$$
\mathcal L_{\text{KD}}
=
\sum_{i,t}
m_{i,t}
D(
\pi_T(\cdot\mid c_T,y_{i,<t})
\;\|\;
\pi_\theta(\cdot\mid c_S,y_{i,<t})
).
$$

For forward KL / cross-entropy distillation, the student-dependent part is easy to see:

$$
D_{\text{KL}}(\pi_T\|\pi_\theta)
=
\sum_a
\pi_T(a\mid c_T,y_{<t})
\left[
\log \pi_T(a\mid c_T,y_{<t})
-
\log \pi_\theta(a\mid c_S,y_{<t})
\right].
$$

The part depending on the student is:

$$
-
\sum_a
\pi_T(a\mid c_T,y_{<t})
\log \pi_\theta(a\mid c_S,y_{<t}).
$$

This resembles a weighted log-prob update, but it is not the same object as the policy-gradient loss above. The weight is a teacher probability over possible next tokens, not a reward-derived scalar advantage on a sampled token.

For on-policy distillation / OPSD, it helps to keep two levels separate:

1. **Loss-level distillation:** optimize a KL, cross-entropy, or JSD-style teacher loss directly.
2. **Advantage-style interpretation:** inspect the gradient and read teacher-student log-ratios as dense advantage-like coefficients.

The second view is useful, but it should be treated as a lens, not as an identity. A full-distribution KL, a top-$K$ approximation, a sampled-token surrogate, and a PPO/GRPO-style advantage loss can share similar-looking terms while having different fixed points, support behavior, and stability properties.

This distinction is the reason SDPO, SRPO, and RLSD should not be collapsed into one formula:

- **SDPO** starts as a dense teacher loss; the advantage view explains its gradient.
- **SRPO** decides which samples should use a teacher branch at all.
- **RLSD** keeps the policy-gradient form and uses the teacher gap to modulate reward-derived advantages.

This section moves from the loss-level view to the advantage-style view, while keeping the warning in place: similar weights do not imply equivalent objectives.

#### 4.2 Why Reward-Only Weights Are Too Coarse

In GRPO-style RL, a whole response gets one scalar advantage:

$$
\hat A_{i,t}=\hat A_i.
$$

That is often enough for short verifiable tasks. But long reasoning and agentic tasks need more local credit assignment:

- Which reasoning step was useful?
- Which token began the wrong branch?
- Which tool-call argument should be suppressed?
- Which part of a failed response still deserves credit?

Teacher signals try to answer these questions by making the update more local: either by changing the token weight $w_{i,t}$, or by replacing the sampled-token view with a full-distribution teacher loss.

#### 4.3 SDPO: Teacher Distribution As Dense Token Signal

SDPO is the cleanest example of why the distinction above matters. At the loss level, SDPO is a self-distillation method: the student samples a rollout, the same model is reprompted with feedback as a self-teacher, and training matches the student's next-token distribution to the feedback-conditioned teacher.

$$
\pi_S^t(a)
=
\pi_\theta(a\mid c_S,y_{<t}),
\qquad
\pi_T^t(a)
=
\operatorname{sg}\!\left[
\pi_\theta(a\mid c_T,y_{<t})
\right].
$$

One SDPO-style objective is:

$$
\mathcal L_{\text{SDPO}}
=
\sum_t
D_{\text{KL}}
\left(
\pi_S^t
\;\|\;
\pi_T^t
\right),
$$

with the teacher side detached. The paper also discusses stabilizing variants such as symmetric divergence choices, but the important point for this blog is simpler: the actual objective is still a distribution-matching loss.

The advantage-style view appears when we look at the negative gradient. Ignoring constants whose expectation has zero gradient, the update can be read as:

$$
-\nabla_\theta \mathcal L_{\text{SDPO}}
\approx
\sum_t
\sum_a
\pi_S^t(a)
\left[
\log \pi_T^t(a)
-
\log \pi_S^t(a)
\right]
\nabla_\theta \log \pi_S^t(a).
$$

So the teacher-induced, advantage-like coefficient is:

$$
A_t^{\text{teacher}}(a)
=
\log \pi_T^t(a)
-
\log \pi_S^t(a).
$$

Positive means the feedback-conditioned teacher assigns more probability to candidate token $a$ than the student does; negative means the teacher assigns less. This is the sense in which SDPO has an "advantage format." It is an interpretation of the distillation gradient, not a claim that SDPO is the same objective as PPO or GRPO with a different advantage estimator.

A sampled-token projection of the same idea would evaluate the gap only at the generated token:

$$
\Delta_{i,t}
=
\log \pi_T(y_{i,t}\mid c_T,y_{i,<t})
-
\log \pi_S(y_{i,t}\mid c_S,y_{i,<t}).
$$

That projection is useful for connecting SDPO to token weighting, RLSD, and GRPO-style infrastructure, but it is lower fidelity than the full logit-level objective. This is where many engineering variants live: full vocabulary, top-$K$, sampled token, entropy weighting, routing, and clipping all expose similar-looking "weights," but they are not mathematically interchangeable.

This gives dense token-level signal, but it can over-trust the teacher. A teacher may be helpful on task logic while still introducing style, length, safety, or format preferences that the verifier does not want.

#### 4.4 SRPO: Route Before Trusting The Teacher

SRPO should be explained in two layers, just like SDPO.

At the loss level, SRPO routes each sample to one of two branches. Correct rollouts use reward-aligned GRPO. Incorrect rollouts with available teacher information use an SDPO correction branch:

$$
z_i^{\text{SDPO}}
=
(1-c_i)m_i,
\qquad
z_i^{\text{GRPO}}
=
1-z_i^{\text{SDPO}}.
$$

Here $c_i$ marks correctness and $m_i$ marks whether usable teacher information exists. The routed objective is:

$$
\mathcal L_{\text{SRPO}}
=
\frac{
\sum_{i,t}
z_i^{\text{GRPO}}
\ell_{i,t}^{\text{GRPO}}
+
\sum_{i,t}
z_i^{\text{SDPO}}
\ell_{i,t}^{\text{SDPO}}
}{
\sum_{i,t}
z_i^{\text{GRPO}}
+
\sum_{i,t}
z_i^{\text{SDPO}}
}.
$$

SRPO also has a teacher-confidence weighting detail inside the SDPO branch. The entropy formula belongs in Appendix D, because the main concept here is routing: which samples use reward-derived GRPO, and which samples use teacher-derived SDPO.

At the advantage-style level, the GRPO branch still looks like a sampled-token policy-gradient update:

$$
g_i^{\text{GRPO}}
\approx
\sum_t
\rho_{i,t}
\hat A_i^{\text{GRPO}}
\nabla_\theta
\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

The SDPO branch can be interpreted through the teacher-induced logit-level advantage:

$$
A_{i,t}^{\text{SDPO}}(a)
=
\log q_{i,t}(a)
-
\log p_{i,t}(a),
\qquad
p_{i,t}(a)
=
\pi_\theta(a\mid x_i,y_{i,<t}).
$$

So the routed SRPO update can be read as:

$$
\begin{aligned}
-\nabla_\theta \mathcal L_{\text{SRPO}}
\approx
\;&
\sum_{i,t}
z_i^{\text{GRPO}}
\rho_{i,t}
\hat A_i^{\text{GRPO}}
\nabla_\theta
\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
\\
&+
\sum_{i,t}
z_i^{\text{SDPO}}
\sum_a
p_{i,t}(a)
A_{i,t}^{\text{SDPO}}(a)
\nabla_\theta
\log p_{i,t}(a).
\end{aligned}
$$

This is the clean way to describe SRPO in the token-weighting story:

> SRPO is not a new teacher. It is a routing decision about when teacher weighting should be allowed to replace or augment reward-only RL.

The caveat is important: this is an advantage-style interpretation of the SDPO branch's gradient, not a claim that SRPO literally turns the SDPO loss into the same sampled-token objective as GRPO. SRPO's key design is routing; the paper's own ablation suggests that direct advantage-level mixing is less stable over long horizons than routing samples into different branches.

#### 4.5 RLSD: Reward Direction, Teacher Magnitude

RLSD is the cleanest bridge for this blog.

Define a detached teacher-student gap:

$$
\Delta_{i,t}
=
\operatorname{sg}
\left[
\log \pi_T(y_{i,t}\mid c_T,y_{i,<t})
-
\log \pi_S(y_{i,t}\mid c_S,y_{i,<t})
\right].
$$

Let $\operatorname{sign}(\hat A_i)$ decide whether the sampled response should be reinforced or suppressed. A simple RLSD-style multiplier is:

$$
u_{i,t}
=
\operatorname{clip}
\left(
\exp(\operatorname{sign}(\hat A_i)\Delta_{i,t}),
1-\epsilon_w,
1+\epsilon_w
\right).
$$

Then:

$$
\hat A_{i,t}^{\text{RLSD}}
=
\hat A_i
\left[
(1-\lambda)
+
\lambda u_{i,t}
\right].
$$

The effective token weight becomes:

$$
w_{i,t}^{\text{RLSD}}
\approx
\rho_{i,t}(\theta)
\hat A_i
\left[
(1-\lambda)
+
\lambda u_{i,t}
\right].
$$

Plain-English point:

> RLSD does not ask the teacher to decide correctness. Reward decides direction. The teacher only says which tokens deserve more or less magnitude inside that reward direction.

### 5. Conditional Teacher Trust: The New Direction

Section 4 introduced the core mechanics: SDPO turns teacher context into dense supervision, SRPO routes the teacher branch, and RLSD keeps reward direction while letting teacher gaps modulate token magnitude.

The next direction is broader:

> Once teacher signal can become part of the token update, the central question becomes when that teacher signal deserves trust.

A useful generic form is:

$$
\mathcal L_{\text{teacher}}
=
\sum_{i,t}
z_{i,t}
\tau_{i,t}
D_{\text{teacher}}(i,t),
$$

or, in a reward-anchored policy-gradient view:

$$
w_{i,t}^{\text{eff}}
\approx
\rho_{i,t}
\hat A_i
\left[
1+\lambda\,\tau_{i,t}\,h_{i,t}^{\text{teacher}}
\right].
$$

Here $z_{i,t}$ decides whether the teacher branch is active, $\tau_{i,t}$ is a trust or reliability weight, and $h_{i,t}^{\text{teacher}}$ is the teacher-student signal. The papers below differ in how they estimate $z$, $\tau$, or the teacher signal itself.

#### 5.1 Purify The Teacher Signal

The simplest teacher-student gap mixes several things together:

$$
\log \pi_T(y_t\mid c_T,y_{<t})
-
\log \pi_S(y_t\mid c_S,y_{<t}).
$$

Some of that gap may be task-bearing. Some may be style, length, oracle leakage, or unreachable hindsight.

**RLCSD** makes this explicit. It argues that privileged hints can shift the teacher toward shorter, more direct style tokens, so the raw teacher-student gap may emphasize style rather than correctness. Its fix is contrastive: compare the gap under a correct hint with the gap under a wrong hint, then use the difference to cancel the hint-induced style component and retain more task-bearing signal. For the blog, RLCSD is the natural follow-up to RLSD: not only "teacher changes magnitude," but "first clean the teacher feature being used as magnitude."

**ARG-OPD / AR-OPD** attacks a related problem from the support side. Full privileged imitation may ask the student to jump toward an oracle-conditioned distribution that is not locally reachable from its current prefix. Anchored residual guidance separates a locally compatible anchor from a residual oracle signal. The teacher is no longer an absolute target; it becomes a residual correction around behavior the student can plausibly support.

Blog thesis:

> Teacher-student gaps are not pure credit signals. They contain task signal plus artifacts from the privileged context. Good weighting sometimes starts by subtracting or anchoring away those artifacts.

#### 5.2 Gate Teacher Updates Against Reward Direction

RLSD already separates reward direction from teacher magnitude. **SG-OPD** pushes this idea further by asking whether teacher token preferences agree with verifier-correct direction. It uses sign-consistency gating: teacher guidance is amplified when it agrees with the verifier direction and softened when it conflicts. It also uses phased teacher sampling to help cold-start regions where student rollouts are far from teacher support.

This is one of the cleanest examples for the post because it looks like a trust variable:

$$
\tau_{i,t}
=
\mathbf 1[
\operatorname{sign}(\text{teacher preference}_{i,t})
=
\operatorname{sign}(\text{verifier direction}_i)
].
$$

The exact method is more nuanced, but the intuition is enough:

> The teacher is useful only where its local preference points in the same direction as the reward evidence.

This connects directly to the practical lesson from Sections 3-4: token weights should not be large merely because a teacher is confident. They should be large when teacher confidence is compatible with the objective.

#### 5.3 Match Feedback To The Decision Context

For multi-turn agents, the problem is not only token-level trust; it is whether the feedback is aligned with the decision being updated.

**HERO** targets this agentic version of mismatch. Instead of applying a terminal outcome or full successful trajectory to every intermediate turn, it uses next environment observations as locally aligned hindsight. After the rollout, it converts observations into compact turn-level diagnoses about the original action: whether it was necessary, valid, or caused a failure. This makes the teacher signal closer to the local state-action pair being trained.

**SocraticPO** changes the rollout process rather than the distillation loss. The student first answers independently; when it fails, a teacher provides Socratic natural-language guidance and the student continues. The key weighting idea is reward decay: if a correct answer required teacher intervention, it should not receive the same reward as an unaided success. This treats teacher help as useful exploration support but prevents the policy from learning that assistance is a free path to full credit.

Blog thesis:

> In long-horizon settings, teacher trust is local. The right question is not "was the final answer correct?" but "was this feedback relevant to this decision at this point in the trajectory?"

#### 5.4 Change The Supervision Interface

Some papers question whether token-logit distillation is the right interface at all.

**DistIL** frames rich-feedback RL as distributional DAgger. Instead of reverse-KL or JSD-style self-distillation, it uses forward cross-entropy from a feedback-conditioned expert distribution and argues that this gives stronger improvement guarantees. For this post, DistIL is important because it keeps the on-policy/rich-feedback spirit but changes the loss geometry: the teacher signal is no longer merely an advantage-like token log-ratio.

**OPRD** moves supervision from output logits into hidden representations. Rather than matching next-token probabilities over a huge vocabulary, it aligns student and teacher representations across selected layers on the same on-policy rollouts. This makes it a different answer to the same trust problem: if output-space token weights are noisy or expensive, use teacher internals as the supervision surface.

Blog thesis:

> Conditional trust is not only about multiplying a token loss by a scalar. Sometimes the trusted object is a representation, a forward-CE target, a rubric, or a black-box guidance trace rather than a next-token logit gap.

#### 5.5 Control Context And Teacher Timing

Two newer papers make the "teacher" itself more dynamic.

**Context Returns** studies context internalization failures. A student may learn to perform well without privileged context, but then degrade when that same context is reintroduced at inference. The proposed fix, no-context anchoring, regularizes the context-conditioned output to stay consistent with the no-context behavior. In the blog's language, this is a compatibility check: internalized teacher signal should not make the policy brittle when the original context returns.

**Teacher-Move Timing** studies the self-teacher refresh schedule. It finds that teacher freshness is not automatically good; moving teachers can destabilize learning if the student is copied into the teacher during a transient drift. The key stability variable is the isolation period between teacher refreshes, and the proposed consolidation-gated refresh updates the teacher only when reward improvement and length-tail safety suggest the student has actually consolidated.

Blog thesis:

> Teacher trust is temporal too. A teacher can be too stale, too fresh, or refreshed at the wrong moment.

#### 5.6 Compact Map

| Method | Trust Question | Mechanism | Use In Blog |
|---|---|---|---|
| RLCSD | Is the teacher gap task signal or style drift? | Contrast correct-hint and wrong-hint gaps. | Purify teacher-derived token weights. |
| ARG-OPD / AR-OPD | Is full privileged imitation locally reachable? | Anchor to partial privilege, add oracle residual. | Residual guidance, not absolute imitation. |
| SG-OPD | Does teacher preference agree with verifier direction? | Sign-consistency gate and phased teacher sampling. | Reward-compatible teacher trust. |
| HERO | Is feedback aligned with this agent turn? | Next-observation hindsight and turn-level diagnosis. | Local trust for multi-turn agents. |
| SocraticPO | Did the model solve unaided or with teacher help? | Interactive guidance plus reward decay. | Assisted success should get discounted credit. |
| DistIL | Is reverse-KL/JSD the right rich-feedback objective? | Distributional DAgger with forward cross-entropy. | Rich feedback as imitation, not only RL-style advantage. |
| OPRD | Are output logits the right supervision surface? | Hidden-state representation alignment. | Teacher signal can live below token logits. |
| Context Returns | Does internalized context remain compatible when context returns? | No-context anchoring / context removability. | Test both with and without privileged context. |
| Teacher-Move Timing | When should the self-teacher refresh? | Isolation periods and consolidation-gated refresh. | Teacher versioning is part of the algorithm. |

Suggested paragraph:

> The June papers make the same point from different angles. The teacher is no longer treated as a uniformly correct dense signal. It is a signal source that needs support checks, reward-direction checks, context checks, temporal checks, and sometimes a different interface entirely. In that sense, "teacher weighting" is becoming conditional teacher trust.

### 6. Conclusion: Better Rewards Are Not Enough

Close by returning to the thesis:

- Rewards decide broad direction.
- Ratios control how far the current policy can move from the policy that produced the sample.
- Normalization controls which examples and tokens dominate.
- Teacher signals improve local credit assignment.
- Systems mismatch can silently corrupt all of the above.

Suggested closing:

> The future of LLM RL is not only better rewards. It is better accounting for how much each sampled token should count, under which policy generated it, which advantage assigned it credit, and which teacher signal, if any, should be trusted.

## Appendix A: Policy-Gradient Derivation

This appendix can hold the derivation details so Section 2 stays readable.

Start from expected reward:

$$
J(\theta)
=
\mathbb E_{y\sim \pi_\theta}
\left[
R(y)
\right]
=
\sum_y \pi_\theta(y)R(y).
$$

Take the gradient:

$$
\nabla_\theta J(\theta)
=
\sum_y
\nabla_\theta \pi_\theta(y)R(y).
$$

Use:

$$
\nabla_\theta \pi_\theta(y)
=
\pi_\theta(y)\nabla_\theta\log\pi_\theta(y).
$$

Then:

$$
\nabla_\theta J(\theta)
=
\sum_y
\pi_\theta(y)R(y)
\nabla_\theta\log\pi_\theta(y)
=
\mathbb E_{y\sim\pi_\theta}
\left[
R(y)\nabla_\theta\log\pi_\theta(y)
\right].
$$

To do gradient ascent:

$$
\theta
\leftarrow
\theta
+
\eta R(y)\nabla_\theta\log\pi_\theta(y).
$$

But optimizers usually minimize losses, so define the REINFORCE-style surrogate:

$$
\mathcal L_{\text{RF}}
=
-R(y)\log\pi_\theta(y).
$$

Then:

$$
\nabla_\theta \mathcal L_{\text{RF}}
=
-R(y)\nabla_\theta\log\pi_\theta(y),
$$

and gradient descent gives:

$$
\theta
\leftarrow
\theta
-
\eta\nabla_\theta\mathcal L_{\text{RF}}
=
\theta
+
\eta R(y)\nabla_\theta\log\pi_\theta(y).
$$

Usually $R(y)$ is replaced by an advantage estimate $\hat A$:

$$
\mathcal L_{\text{RF}}
=
-\hat A\log\pi_\theta(y).
$$

Important note for readers:

> The actor loss is a gradient surrogate. Its numeric value is often much less meaningful than its gradient, ratio statistics, KL, entropy, reward, gradient norm, and evaluation metrics.

This explains why policy losses can be small, near zero, positive, or negative without directly meaning "good" or "bad" in the supervised-learning sense.

## Appendix B: PPO Training Loop Details

### B.1 Rollout Batch, Minibatch, Microbatch, Optimizer Step

These terms are easy to blur in LLM training code:

| Term | Meaning |
|---|---|
| Rollout batch | The full generated data batch collected from $\pi_{\text{old}}$. |
| PPO minibatch | A shuffled subset of the rollout batch used for one logical actor update. |
| PPO epoch | One full pass over the rollout batch. |
| Microbatch | A memory chunk inside a PPO minibatch, used for gradient accumulation. |
| Optimizer step | The actual parameter update, e.g. `optimizer.step()`. |

Example:

```text
rollout batch size = 1024 responses
PPO minibatch size = 256 responses
PPO epochs = 4
```

Then:

```text
1024 / 256 = 4 minibatches per epoch
4 epochs * 4 minibatches = 16 optimizer steps
```

Each rollout response is reused 4 times, once per PPO epoch.

If the PPO minibatch is too large for memory:

```text
PPO minibatch size = 256
microbatch size = 64
```

then one logical minibatch update is implemented as:

```text
microbatch 1: forward/backward, accumulate gradients
microbatch 2: forward/backward, accumulate gradients
microbatch 3: forward/backward, accumulate gradients
microbatch 4: forward/backward, accumulate gradients
optimizer.step()
```

That is one optimizer step, not four. Microbatching is a systems trick; PPO minibatching is an algorithmic data-reuse choice.

### B.2 What PPO Clipping Does To Gradients

PPO uses:

$$
\min
\left(
\rho_t\hat A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right).
$$

For $\hat A_t>0$:

- if $\rho_t \le 1+\epsilon$, PPO can keep increasing the sampled token probability;
- if $\rho_t > 1+\epsilon$, the clipped branch is constant and gives no more upward gradient.

For $\hat A_t<0$:

- if $\rho_t \ge 1-\epsilon$, PPO can keep decreasing the sampled token probability;
- if $\rho_t < 1-\epsilon$, the clipped branch is constant and gives no more downward gradient.

PPO's clipping is not a generic "ignore all out-of-range tokens" rule. It removes gradients that would make an already-large reward-improving policy movement even larger.

## Appendix C: Mismatch And Rollout Correction

### C.1 PPO's Intended Mismatch

PPO's intended mismatch is:

$$
\pi_\theta
\ne
\pi_{\text{old}}
$$

after one or more optimizer steps on a fixed rollout batch.

The control is:

$$
\rho_t^{\text{PPO}}
=
\frac{\pi_\theta(y_t\mid s_t)}
{\pi_{\text{old}}(y_t\mid s_t)}.
$$

This is sample-reuse mismatch: the current actor has moved away from the actor that generated the batch.

### C.2 Rollout/Training Mismatch

The PPO mismatch above assumes:

$$
\pi_{\text{rollout}} = \pi_{\text{old}}.
$$

In real LLM RL systems, that assumption can fail. The rollout engine may be vLLM or SGLang, while the training logprob engine may be FSDP, Megatron, or another distributed actor implementation. Async queues can also make generated trajectories stale.

Then there may be three policies:

$$
\pi_{\text{rollout}}
\quad
\pi_{\text{old}}
\quad
\pi_\theta.
$$

PPO controls the training ratio:

$$
\rho_{t}^{\text{train}}
=
\frac{\pi_\theta(y_t\mid s_t)}
{\pi_{\text{old}}(y_t\mid s_t)}.
$$

Rollout correction controls a separate rollout/training discrepancy:

$$
\omega_t^{\text{rollout}}
=
\frac{\pi_{\text{old}}(y_t\mid s_t)}
{\pi_{\text{rollout}}(y_t\mid s_t)}.
$$

Together:

$$
\frac{\pi_\theta(y_t\mid s_t)}
{\pi_{\text{rollout}}(y_t\mid s_t)}
=
\rho_t^{\text{train}}
\cdot
\omega_t^{\text{rollout}}.
$$

Clean distinction:

| Mismatch | Cause | Typical control |
|---|---|---|
| PPO sample-reuse mismatch | Reusing data after actor optimizer steps | PPO ratio and clipping |
| Rollout/training mismatch | Generation backend, training backend, precision, async staleness, tokenization/logprob semantics | rollout IS, clipping, masking, token discard, version tracking |

Blog-friendly phrasing:

> PPO's ratio answers: "How far has the trainable actor moved from the old actor that this batch assumes?" Rollout correction answers: "Did the system that generated this token and the system that scores its logprob agree on what policy produced it?"

### C.3 Token Discard And Training-Inference Importance Sampling

The practical tip from MiMo/IcePop-style training-inference importance sampling belongs to rollout/training mismatch.

If the rollout engine and trainer strongly disagree about a token's probability, that token may have an extreme correction weight:

$$
\log \omega_t
=
\log \pi_{\text{old}}(y_t\mid s_t)
-
\log \pi_{\text{rollout}}(y_t\mid s_t).
$$

Then truncated token IS is:

$$
\tilde \omega_t
=
\operatorname{clip}
\left(
\exp(\log \omega_t),
0,
C
\right).
$$

or sequence-level IS:

$$
\tilde \omega_i
=
\operatorname{clip}
\left(
\exp\left(\sum_t \log \omega_{i,t}\right),
0,
C
\right).
$$

Token-level IS has lower variance but can be biased when mismatch is large. Sequence-level IS is more faithful but much more sensitive to long-response outliers.

Clipping or discarding such tokens is not the same as PPO clipping. It is saying the probability accounting for that token is unreliable or too high-variance to trust.

### C.4 Practical Checklist

Every rollout row should know:

- rollout policy version
- rollout backend and precision
- sampling parameters
- reward/verifier version
- teacher version, if teacher scoring is used
- old-logprob source: rollout logprobs, recomputed actor logprobs, or snapshot model

Do not silently repair sampled tokens after generation.

Track:

- timeout
- invalid special token
- generated token count
- prompt token count
- response-length cap
- first-token BOS/EOS
- blank output
- backend errors

Why:

> If rollout repair changes the sampled token IDs after the policy sampled them, the log-prob ratio is now describing a different trajectory from the one the environment saw.

For OPSD/SRPO/RLSD-style runs, version the teacher too:

$$
\pi_T^{(k)}
\neq
\pi_S^{(k)}
\neq
\pi_\theta^{(k+1)}.
$$

Practical tips:

- log teacher checkpoint/version
- log teacher prompt template hash
- log teacher context type
- separate teacher target construction from rollout generation
- route inactive rows to zero teacher target and zero self-distillation mask
- keep teacher scoring active-row-only only after equivalence tests

## Appendix D: Teacher-Trust Weighting

The main text should keep SRPO focused on sample routing. The entropy detail belongs here because it is one instance of a broader pattern: teacher-derived signals are increasingly weighted by confidence, reliability, or agreement with reward.

In SRPO, the SDPO branch can be additionally weighted by teacher entropy. If $q_{i,t}$ is the feedback-conditioned teacher distribution, then:

$$
\tilde \alpha_{i,t}
=
\exp(-\beta H(q_{i,t})),
\qquad
\alpha_{i,t}
=
\frac{\tilde \alpha_{i,t}}
{\frac{1}{|\Omega_{\text{SDPO}}|}
\sum_{(j,s)\in \Omega_{\text{SDPO}}}
\tilde \alpha_{j,s}}.
$$

The weighted SDPO token loss is:

$$
\ell_{i,t}^{\text{DW-SDPO}}
=
\alpha_{i,t}\ell_{i,t}^{\text{SDPO}}.
$$

Intuition:

> Low-entropy teacher distributions are treated as more reliable corrective targets; high-entropy teacher distributions are downweighted because the teacher itself is uncertain.

For the blog, this can be mentioned briefly in the main text and revisited when discussing newer conditional-trust methods. SRPO's headline idea is routing; entropy weighting is a trust-weighting refinement on the SDPO branch.

## Suggested Figures

Keep the main post to four or five visuals. The figures should do different jobs: introduce the token-weight lens, compare policy-gradient variants, clarify the distillation-vs-advantage distinction, explain the SDPO/SRPO/RLSD family, and then summarize conditional teacher trust. The async mismatch diagram belongs in the appendix.

### Figure 1: The Token Weight Stack

Placement: Section 1, right after the generic actor loss.

Purpose: Make the thesis visible before the formulas get dense.

```text
sampled token
   |
mask m_{i,t}
   |
policy ratio rho_{i,t}
   |
reward advantage A_i
   |
normalization / clipping / filtering
   |
teacher gate or trust multiplier
   |
token gradient
```

Caption idea:

> Most LLM RL variants differ less in the final trainable handle than in the stack of weights and masks applied before the token log-prob gradient.

### Figure 2: PPO To CISPO As Weight Edits

Placement: Section 3, before or after the DAPO/CISPO subsections.

Purpose: Prevent the PPO-to-CISPO section from feeling like a list of unrelated algorithms.

```text
PPO
  ratio rho * advantage A

GRPO
  PPO ratio + group-relative advantage

RLOO / Dr. GRPO / REINFORCE++
  change baseline, normalization, and loss scaling

DAPO
  change token aggregation, sample filtering, and clipping asymmetry

CISPO
  clip the importance weight while keeping the live log-prob gradient path
```

Alternative format: a compact table with columns `method`, `advantage source`, `ratio handling`, `normalization`, `token effect`.

### Figure 3: Distillation Loss vs. Advantage-Style Interpretation

Placement: Section 4.1, where the draft warns that the forms are not equivalent.

Purpose: This is probably the most important conceptual figure. It should show that the loss-level objective and the advantage-style gradient interpretation are connected but not identical.

```text
Loss-level view
---------------
teacher distribution q_t(.)
student distribution p_t(.)
KL / CE / JSD over next-token distribution
actual optimized objective

Gradient interpretation
-----------------------
teacher-student log ratio
log q_t(a) - log p_t(a)
advantage-like coefficient
useful lens for token weighting
```

Caption idea:

> SDPO begins as distribution matching. The teacher-induced advantage appears when we inspect the gradient; it is an interpretation and implementation bridge, not an equivalent PPO objective.

### Figure 4: Three Ways To Use Teacher Signal

Placement: Section 4, after RLSD or as a summary before Section 5.

Purpose: Show SDPO, SRPO, and RLSD as different roles for the same broad teacher-student signal.

```text
student rollout y
      |
      +--> teacher context c_T
      |       |
      |       +--> teacher distribution q_t
      |
      +--> verifier reward r

SDPO:
  q_t defines dense distillation target

SRPO:
  verifier outcome routes samples:
    correct -> GRPO
    failed + teacher info -> SDPO branch

RLSD:
  verifier reward decides direction
  teacher-student gap scales token magnitude
```

Caption idea:

> SDPO, SRPO, and RLSD are not one objective. They differ in whether teacher signal defines the target, chooses the branch, or modulates a reward-grounded update.

### Figure 5: Conditional Teacher Trust Map

Placement: Section 5, replacing part of the compact map if the final post needs less text.

Purpose: Give readers a one-screen view of the newer papers without turning the section into a literature dump.

| Trust Problem | Representative Methods | Visual Keyword |
|---|---|---|
| Purify teacher signal | RLCSD, ARG-OPD | subtract / anchor |
| Check reward direction | SG-OPD, RLSD | sign gate |
| Match local decision context | HERO, SocraticPO | turn-level feedback |
| Change supervision interface | DistIL, OPRD | CE / representation |
| Control context and timing | Context Returns, Teacher-Move Timing | context / clock |

Caption idea:

> The newest OPD/OPSD papers treat the teacher as a conditional signal source, not as uniformly reliable dense supervision.

### Appendix Figure A: Two Ratios In Async RL

Placement: Appendix C, near rollout/training mismatch.

Purpose: Keep the systems material visible without interrupting the algorithmic story.

```text
pi_rollout -- rollout correction omega --> pi_old -- PPO ratio rho --> pi_theta
```

Caption idea:

> PPO's ratio controls reuse of old sampled data. Rollout correction controls mismatch between the generation backend and the training log-prob backend.

## Reference Spine

Foundational RL and RLHF:

- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

Critic-free and reasoning-LLM RL:

- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)

Token-ratio and token-weighting refinements:

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- [State of RL for reasoning LLMs](https://aweers.de/blog/2026/rl-for-llms/)

OPSD / OPD with RL focus:

- [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
- [Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing](https://arxiv.org/abs/2604.02288)
- [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128)
- [Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155)

Conditional teacher trust and recent OPD variants:

- [When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation](https://arxiv.org/abs/2606.03532)
- [Reinforcement Learning from Rich Feedback with Distributional DAgger](https://arxiv.org/abs/2606.05152)
- [OPRD: On-Policy Representation Distillation](https://arxiv.org/abs/2606.06021)
- [SG-OPD: Sign-Gated On-Policy Distillation via Sign-Consistency Gating and Phased Teacher Sampling](https://arxiv.org/abs/2606.09304)
- [SocraticPO: Policy Optimization via Interactive Guidance](https://arxiv.org/abs/2606.09887)
- [Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation](https://arxiv.org/abs/2606.10385)
- [HERO: Hindsight-Enhanced Reflection from Environment Observations for Agentic Self-Distillation](https://arxiv.org/abs/2606.11559)
- [When Context Returns: Toward Robust Internalization in On-Policy Distillation](https://arxiv.org/abs/2606.11627)
- [RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation](https://arxiv.org/abs/2606.11709)

Async mismatch and rollout correction:

- [IcePop: Training-Inference Importance Sampling](https://ringtech.notion.site/icepop)
- [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)
- [Why Off-Policy Breaks RL: An SGA Analysis Framework](https://richardli.xyz/post/rl-collapse-part1/)
- [All-In-One Solution to Training-Inference Mismatch with Miles](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md)
- [Training-Inference-Mismatch - ms-swift documentation](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/training_inference_mismatch.html)
- [TRL Paper Index: Training-Inference Distribution Shift](https://huggingface.co/docs/trl/en/paper_index)
- [Rollout Correction - verl documentation](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- [Mathematical Formulations of Rollout Correction Methods in verl](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
- [Missing Old Logits in Asynchronous Agentic RL](https://arxiv.org/abs/2605.12070)
- [How Far Can Off-Policy RL Reach with Stale Data on LLMs?](https://arxiv.org/abs/2510.01161)
- [A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation](https://arxiv.org/abs/2512.06547)

## Open Questions Before Turning This Into A Post

- Do we want a small code excerpt from `verl` showing `ratio = exp(log_prob - old_log_prob)` and CISPO's detached clipped ratio?
- Should the title use "Token Weighting" explicitly, or the more accessible "Every Token Gets a Weight"?

## Possible Opening Paragraph

When people describe LLM reinforcement learning, they usually talk about rewards: did the answer pass the unit tests, solve the math problem, follow the instruction, or satisfy the verifier? But inside the actor update, the object that actually touches the model is smaller and more mechanical. It is a scalar multiplying a token log-probability. PPO, GRPO, DAPO, CISPO, and reward-anchored OPSD can all be read as different ways of constructing that scalar. Once you look at the algorithms this way, the recent explosion of RL variants becomes less mysterious: everyone is arguing over which tokens deserve gradient, how much, and under which policy mismatch assumptions.
