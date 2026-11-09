# Mismatch mitigation in LLM RL: research and article outline

Updated: 2026-11-08. Status: [publication version](../posts/rl-mismatch-mitigation/index.qmd) completed after full-draft review; draft metadata removed. This document retains the skeleton and broader source map. The article has six main sections, compact sidebars, numbered main equations, four original explanatory SVGs, and one attributed original REINFORCE Zero figure.

Review checkpoint: the PPO/TIS baseline and its assumptions precede the LMSYS forward/backward bridge, SC plus sampler KL, and TII/REINFORCE Zero. The worked gradient expectations and sampler-KL derivative were numerically checked. Local preview, links, math rendering, and desktop/mobile layout were inspected; no training run was reproduced. The final pass makes ordinary IS explicit as the lag correction, separates anchor and direct ratios, distinguishes sampler/reference/diagnostic KL, qualifies TorchTitan's cache-retention behavior, and organizes Section 6 around the limits of token correction. MiMo is now related reading; repeated qualifications were shortened or moved to a footnote.

Section 5 review: expanded same-snapshot versus stale-policy agreement; added GEMM, cache-phase, and distributed-layout examples from the TII report; explained inference-defined forward values and independently validated backward support; added a compact diagram and the live-ratio gradient derivation. The IS discussion distinguishes mean correction from finite-batch variance. Experimental scope now explicitly identifies BF16 math runs and the aligned synchronous baseline behind the throughput comparison. Forward regression and backward/update validation remain separate requirements when changing kernels.

Companion: [Low-Precision RL: From Rollout to Update](../posts/low-precision-rl/index.qmd).

## Proposed argument

**Title: Rollout–Training Mismatch in LLM RL.**

**Subtitle: Importance sampling, score centering, and REINFORCE Zero.**

Lower-precision rollout can reduce memory and compute costs; asynchronous collection can reduce waiting. Both can separate the policy producing experience from the policy being updated. Neither mismatch nor staleness is mathematically necessary for RL, and BF16 on both sides does not guarantee matching probabilities. The practical question is how to retain useful efficiency gains while preserving effective learning.

The narrative is: **why probabilities disagree → what PPO/GRPO plus rollout correction already handles → what forward alignment changes → SC and sampler KL → TII and REINFORCE Zero → what remains under staleness.** Establish the existing correction baseline before presenting either featured method. SC and REINFORCE Zero are selected technical case studies, not asserted industry defaults or the only serious approaches.

Use **trainer–rollout forward-probability mismatch** for the concrete same-version problem. “Execution mismatch” is a useful umbrella only after naming quantization, kernel arithmetic, routing, and state handling. Forward agreement, backward accuracy, and optimizer-state precision are separate decisions. This framing is editorial synthesis; the cited methods do not share one objective or a single historical lineage.

The intended reader already knows policy gradients and has working familiarity with SC and sampler KL. The author's Miles experience supplies that context; it does not need an experiment vignette. REINFORCE Zero should be discussed through its published mechanism and evidence, without implying firsthand validation.

## What the literature contributes

Coverage below means selected mechanism, experiment, or implementation passages were checked. No LLM training was reproduced, no complete framework audit was performed, and source selection is not a citation-count ranking. Results remain author-reported. New papers receive space for a distinct mechanism, not simply because they are recent.

### Central references

| Source | Contribution to the article | Evidence boundary and reading coverage |
| :--- | :--- | :--- |
| **TIS: On the Rollout-Training Mismatch in Modern RL Systems** [1] | Establish the correction baseline: same weights can yield different probabilities; cap an importance ratio to limit large contributions. Its PPO extension separates rollout correction from current/old clipping. | Five-page workshop report inspected, including objectives and experiment setups. Quantized-rollout evidence is specific to its models and recipes; truncation introduces bias. |
| **Score Centering** [2] | Main additive-correction mechanism: remove conditional mean-score drift while retaining sampler-distributed covariance. Explain practical top-$k$ reconstruction and weighted centering. | Checked §4 and implementation/appendix material plus public reference code. Exact algebra is stronger evidence than a claim of universal long-run stability. Practical reconstruction is approximate. |
| **SFC plus trainer-generator KL** [3] | Develop the second part of the SC story: control the discrepancy between the retained sampler-relative update and the current policy, using $D_{\mathrm{KL}}(q\|p)$. | Author's derivation and selected implementation/experiment material checked. The local bound requires a bounded integrand; soft regularization does not enforce a hard KL constraint. Age-dependent coefficients are heuristics. |
| **Train-Inference Consistency** [4] | Explain the engineering route: inference kernels in the trainer forward, batch invariance, and matching phase/operator arithmetic. | Official October 14 engineering report and selected alignment guides checked. Reported costs are configuration-specific; this is not general support for every model, topology, or precision mode. |
| **REINFORCE Zero** [5] | Main combined recipe: TII suppresses execution mismatch, then unclipped token-level IS handles remaining weight lag. | Checked paper problem setting, correction comparison, Gaussian model, selected results/settings, and algorithm guide. The paper reports 20,000-update runs and 2.40× training-step throughput versus Sync TII on 16 GPUs, excluding evaluation. Its SC comparison uses aligned execution, top-32 centering, and no IS; it does not test the whole SC-plus-sampler-KL family. |
| **AReaL** [6] | Main staleness system example: bounded outstanding rollout work, interruptible generation, and separate behavior/proximal/current policy roles. | Checked §§4–5 and selected setup/results. In-flight refresh can create mixed-version responses; the actual token probabilities matter. A proximal snapshot scored at batch arrival need not be the historical trainer snapshot corresponding to generation. |
| **Predicting and Controlling Staleness** [7] | Explain where age comes from: generation/group completion before queue entry, then queue residence. Connect age to resource allocation and workload tails. | Checked the model, queue assumptions, and control discussion. Closed forms assume a particular GRPO pipeline and simplified arrival/service behavior; queue recommendations are not universal. |

### Supporting references that prevent an incomplete story

| Source | Keep in the article | Scope |
| :--- | :--- | :--- |
| **Unified FP8** [8] | Main bridge from the earlier post: use the four-way trainer ablation to distinguish forward-value alignment, backward quantization, and native FP8 GEMMs. | Rechecked QDQ/native-GEMM ablations. The proposed TIS combinations below are an organizing comparison, not a claim that the published four-way ablation enabled TIS in every case. Its clip-fraction diagnostic alone does not establish the applied loss. |
| **Defeating Nondeterminism in LLM Inference** [9] | Origin of the batch-invariance discussion: a deterministic kernel can still change arithmetic with batch shape. | Checked reduction-order discussion and true-on-policy experiment. Batch invariance within one runtime and agreement between two runtimes are distinct requirements. |
| **R3** [10] | One compact MoE example: replay selected expert IDs to address a discrete execution disagreement. | Checked mechanism/implementation. Trainer gate weights remain recomputed and differentiable. Matching routes alone does not match hidden states or expert arithmetic. |
| **FP16 mismatch study** [11] | Brief reminder that disagreement also appears in BF16/BF16 execution; reduced rounding error is another engineering lever. | Checked precision argument and selected settings. Improved agreement from FP16 is empirical, not a datatype-level guarantee of identical execution; range and loss scaling still matter. |
| **VeXact** [12] | A strong motivation source: use a zero-mismatch reference to isolate numerical effects from update drift and compare compensation choices. | Checked zero-mismatch construction, diagnostic experiments, and limitations. Evidence is primarily math RL; avoid presenting it as universal causal identification for all training failures. |
| **TorchTitan hybrid-attention analysis** [13] | Important counterweight: alignment can be useful for diagnosis while its training benefit and cost remain workload-dependent. | Checked shared-model/kernel construction and result discussion. Mostly single-seed workloads and implementation-specific slowdowns; do not transfer its costs to REINFORCE Zero or generalize either result. |
| **Miles SC documentation** [14] | One implementation sidebar on payload, weighted centering, filtered support, and diagnostic-versus-loss distinctions. | Current documentation checked. SC selection also enables FP32 rollout LM-head arithmetic, so a configuration comparison can change two mechanisms. A logged rollout/trainer KL is not automatically a sampler-KL penalty. |

### Further reading, with limited main-text space

- **IcePop / Ring-1T** [15]: include a short comparison with TIS. Checked the double-sided ratio gate and selected ablations. It masks out-of-range tokens rather than capping their weights. Its compounding-discrepancy theorem has explicit alignment assumptions and a discrepancy threshold; avoid quoting compounding as an unconditional law.
- **Asynchronous RLHF** [16] and **IMPALA** [17]: historical context for the utilization/off-policy tradeoff. Abstract-level screening here; their algorithms are not substitutes for an LLM token-IS derivation.
- **Align and Filter / VACO** [18]: useful continuation on stale advantage information, not just action ratios. Introductory mechanism screening; inspect the proof and LLM experiments before explaining its guarantees.
- **SAT** [19] and **confidence-aware IS / CIS** [20]: possible short pointers on adapting update control to realized discrepancy. Selected method passages screened; no full statistical or proof audit. CIS is distinct from CISPO. Keep these out of the core derivation unless a specific gap remains.
- **PPO and GRPO** [21,22]: authorities for the baseline surrogate and group-relative credit assignment. Their full derivations belong in background reading; explain the ratio roles needed here.
- **DAPO and MiniMax-M1/CISPO** [23,24]: selected objective and motivation passages checked. Use the history sidebar to distinguish asymmetric PPO bounds from detached clipped weights; neither is simply another backend correction.
- **GLM-5, SAO, and MiMo-V2.6** [25,26,27]: selected provenance, direct-ratio, masking, and credit-assignment passages checked. Use brief examples in Section 6; their complete training recipes are outside this article.
- **Practitioner account of SFC** [28]: reports equivalent industry techniques under other names. This is an attributed account, not verified evidence of widespread pre-publication use or a settled priority history.

## Proposed article flow

Target roughly 3,500–4,500 words, with longer derivations in a technical note. Sections 2 and 3 must establish a substantial baseline before the featured methods. Keep the history in a sidebar and introduce each formula where it answers the section's question.

### 1. Why rollout and training disagree

Connect directly to the previous article: precision choices determine the numerical policy sampled by rollout and scored by the trainer. This follow-up asks what the update should do when those policies differ.

Open with one concrete, source-backed example of same-weight probability disagreement [9,12], then introduce stale experience as a second source. Lower precision and overlap motivate accepting these discrepancies, but more generated tokens or higher utilization do not establish better learning. Avoid implying mismatch explains every collapse.

Briefly answer why BF16 is not a complete answer: it may be affordable for a smaller workload, becomes expensive in weight/state storage at larger scale, and still allows different arithmetic across runtimes. Treat model-size implications as engineering considerations, not a monotonic law of numerical error or version lag.

Define same-version forward-probability mismatch and weight-version lag. Show the log-ratio split without making it the article's sole organizing equation. Note that elapsed age, update count, and distribution distance are different quantities.

### 2. What the usual corrections assume

Establish the baseline family before discussing SC or REINFORCE Zero. “Usual” names an established family of recipes, not a measured adoption ranking.

- **PPO/GRPO update control:** show how current/frozen-anchor ratios enter the surrogate. Explain the conventional assumption that rollout data comes from the anchor policy, and how backend disagreement breaks that equality. PPO clipping gates selected contributions; it does not enforce a hard global probability or KL bound. GRPO changes credit assignment, not that numerical assumption. [21,22]
- **Separate rollout correction:** define actual behavior $q$, frozen trainer anchor $\pi_0$, and live trainer $p$. Distinguish $\pi_0/q$ from $p/\pi_0$. Derive PPO plus detached truncated correction; contrast a capped TIS weight with an out-of-range masking rule. [1,15]
- **Assumptions:** generation-time probabilities must describe the actual sampled token, history, version, and sampling transformation. Exact conditional IS requires support coverage. A recently scored proximal anchor need not be the historical trainer at the generating version. [1,6]
- **Accepted tradeoffs:** truncation/masking changes the estimator; token correction leaves prefix visitation and continuation-credit issues. Mean KL alone does not describe ratio tails or useful sample retention.
- **Freshness controls:** mention bounded outstanding work, sync cadence, and queue policy as complementary baseline measures. A maximum age does not establish a maximum distribution gap. [6,7]

Explain when this baseline can be sufficient: the measured disagreement and ratio tails remain manageable, correction preserves enough useful signal, and downstream learning is satisfactory at the available compute budget. No premise requires SC or exact numerical alignment in every run.

Introduce the later methods through the remaining questions: can we reduce the discrepancy before weighting it, remove a specific drift component, or make full weights practical?

### 3. From correction to forward alignment

Use the previous article's LMSYS discussion as the concrete bridge. Hold FP8 rollout fixed while separating three trainer decisions: forward operands, backward operands, and GEMM implementation.

| Organizing configuration | Trainer forward | Trainer backward | Question answered |
| :--- | :--- | :--- | :--- |
| TIS with BF16 trainer / FP8 rollout | Ordinary BF16 values and GEMMs | BF16 | Can correction make accepted forward disagreement manageable? |
| TIS plus forward QDQ | FP8-grid values reconstructed for BF16 GEMMs | BF16 | How much does matching forward operand values reduce the gap? |
| TIS plus forward and backward QDQ | Same quantized-forward target | Quantized operand values with BF16 GEMMs | What changes when the gradient computation is also quantized? |
| TIS plus native FP8 operations | Eligible GEMMs consume FP8 codes/scales | Eligible DGrad/WGrad use FP8 GEMMs | What performance and numerical changes come from native kernels? |

This table combines a correction layer with numerical choices for exposition. Cite the published four-way ablation as evidence for the numerical distinctions, not as proof that every published case used this TIS configuration. [8]

Make the implications explicit:

- FProp defines the probabilities in the current objective. Backward quantization changes gradient quality and subsequent updates; it cannot retroactively change the already-computed forward probabilities.
- Rollout has no backward path to match. Say “quantizing trainer backward operands,” not “matching rollout backward.” Native FP8 is an implementation/performance choice, not automatically a stronger agreement guarantee.
- Matching QDQ values can substantially reduce a discrepancy, while kernel arithmetic, reduction order, routing, phase differences, and state handling can leave a residual.
- Higher-precision master weights and optimizer moments remain a separate choice. They need not use serving precision. [4,8]

Place the update-path diagram here (Figure 2 in the full draft): sampled tokens and behavior probabilities enter the trainer loss beside the current forward scores; the loss signal passes through backward into the optimizer. Mark forward agreement separately from backward correctness and parameter publication. Use the chain rule to locate numerical changes versus loss-level interventions.

Transition: after baseline correction and forward-value alignment, what can be done about residual forward disagreement and policy movement? Sections 4 and 5 examine different answers rather than successive mandatory upgrades.

### 4. Score centering: correcting the residual update

Now derive the conditional drift/covariance identity. Explain which part of the uncorrected update SC cancels and which sampler-distributed covariance it retains. Group reward centering does not generally eliminate the prefix-conditional drift. [2]

Use one independent-logit calculation to connect SC and sampler KL: remove an accidental coefficient depending on the conditional mean advantage, then add a deliberate coefficient on the sampler-attraction direction. State the KL direction and frozen-sampler assumption; distinguish a soft penalty from a hard constraint. [3]

Keep the implementation discussion concrete and short: top-$k$ candidates/tail approximation, stored sampler metadata, stop-gradient placement, and centering weighted scores when composing with TIS or MIS. Distinguish a sampler-KL loss from a detached diagnostic and a fixed-reference penalty. [2,14]

End with the residual: covariance remains under the sampler, reconstruction is approximate, and the local identities do not correct prefix visitation or stale continuation credit. Source experiments should not be read as a universal ranking of complete training recipes.

### 5. REINFORCE Zero: tighter forward agreement, unclipped correction

Return to Section 3's residual forward mismatch. Describe TII's same-version agreement criterion, and the additional work beyond matching quantized values: batch/phase consistency, kernel arithmetic, routing, distributed reductions, and request-state behavior. Scope claims to validated operators, tasks, layouts, and numerical modes. [4]

Develop REINFORCE Zero's division of labor: aligned forward computation suppresses execution-dependent ratio excursions; unclipped token IS corrects the remaining weight-lag contribution at fixed histories. Explain why restoring a mean does not establish small variance, and why the Gaussian analysis is an explanatory model rather than a transformer convergence theorem. [5]

Treat backward accuracy, STE where relevant, optimizer state, and restore behavior as separately validated. Explain kernel maintenance: a replacement can preserve existing outputs or establish a new jointly validated trainer/rollout pair. Changed weight version, runtime arithmetic, and cache/state provenance must remain distinguishable.

Present reported results with exact comparators and timing boundaries. Contrast the evidence with VeXact's diagnostic role and TorchTitan's workload-dependent benefits/costs. An aligned SC-without-IS baseline is not a comparison against every SC-plus-sampler-KL configuration. [5,12,13]

### 6. What changes when experience becomes stale?

Bring the approaches together through a long, uneven agent trajectory. Follow generation, environment/group wait, queueing, learner updates, and weight publication. One response can span several generating versions; store actual token probabilities and version provenance rather than assigning an invented single old checkpoint. Figure 5 in the full draft illustrates this lifecycle. [6,7,25]

Explain the separate residuals: SC can remove conditional drift while distribution distance remains large; TII can remove same-version forward mismatch while weight lag remains; token-local IS still leaves trajectory-distribution and credit-assignment issues.

Use AReaL, GLM/SAO, and MiMo briefly to show concrete choices: a recent proximal anchor, one rollout per prompt with a learned critic, or a direct current/recorded-behavior ratio with masking. Single-rollout collection does not mean learner batch size one. Their full algorithms belong in the sidebar/readings. [6,25,26,27]

Close with a compact comparison of baseline correction, SC plus sampler KL, and aligned unclipped IS: assumptions, required metadata, residual bias/variance, implementation cost, and validation evidence. Include age distributions alongside ratio/KL tails, rejected contributions, and held-out learning versus GPU-hours or wall time with a defined resource pool. Neither a lower KL nor zero mismatch alone decides the tradeoff. No universal safe lag or method ranking follows.

### Sidebar: how update control, rollout correction, and alignment evolved

Keep this as a short sidebar, not a main section or a single succession of replacements.

| Development | Distinction worth preserving |
| :--- | :--- |
| PPO (2017) → GRPO (2024) [21,22] | Proximal update control persists while the baseline/credit estimator changes. |
| DAPO (March 2025) [23] | Decouples lower and upper PPO bounds; retains a clipped surrogate. |
| CISPO (June 2025) [24] | Clips a detached weight multiplying log-probability, preserving contributions PPO can gate off. |
| TIS and IcePop (2025) [1,15] | Separate behavior correction from proximal control; capping and masking are distinct. |
| Asynchronous systems and agentic RL [6,16,25,26,27] | AReaL, GLM/SAO, and MiMo make different choices about policy anchors, direct behavior ratios, collection barriers, and critics. |
| Numerical alignment and SC [2,4,5,8,9,10] | Forward-value agreement, routing replay, centering, and bitwise alignment target different boundaries and can be combined. |

Avoid saying DAPO/CISPO followed TIS, or that direct current/rollout weighting merely moves an unchanged PPO correction elsewhere. Distinguish three meanings of “decouple”: asymmetric bounds, correction versus proximal ratios, and clipped weights versus clipped surrogate gradients. If mentioning earlier industrial SC use, attribute the practitioner account and preserve uncertainty about adoption and priority. [28]

## Mathematical content to retain

### Policy roles and fixed-history separation (Sections 1–2)

Let $\mu_v$ be the actual rollout distribution at generating version $v$, and $\pi_k,\pi_v$ trainer evaluations at versions $k,v$. Hold the history and trainer execution context fixed. With matching sampling transformations and positive probabilities,

$$
\log\frac{\pi_k(a\mid h)}{\mu_v(a\mid h)}
=\underbrace{\log\frac{\pi_k(a\mid h)}{\pi_v(a\mid h)}}_{\text{weight movement}}
+\underbrace{\log\frac{\pi_v(a\mid h)}{\mu_v(a\mid h)}}_{\text{same-version execution difference}}.
$$

This is an exact factorization [5], not a uniquely identifiable causal attribution. Using current-version rollout as the intermediate policy gives a different valid split. Neither term is automatically a KL divergence.

### PPO plus rollout correction (Section 2)

Keep actual behavior $q$, a frozen proximal trainer $\pi_0$, and live trainer $p$ distinct:

$$
w=\frac{\pi_0}{q},\qquad r=\frac p{\pi_0},\qquad wr=\frac pq.
$$

With detached advantages and $\bar w=\operatorname{sg}[\min(w,C)]$, a token-level baseline loss is

$$
L=-\mathbb E_q\!\left[
\bar w\min\!\left(rA,
\operatorname{clip}(r,1-\epsilon_l,1+\epsilon_h)A\right)
\right].
$$

Specify masks and reduction separately. Truncating $w$ and clipping $r$ does not equal clipping their product. The proximal anchor can intentionally be recent; only a historical anchor at the generating version isolates same-version numerical mismatch. Direct current/behavior weighting is a different surrogate. A fixed reference model would be an additional policy role. [1,6]

Temperature and top-$p$/top-$k$ transforms must be specified when defining these distributions. Truncated behavior support can prevent recovery of an untruncated target. A logprob field is useful only if its distribution and version are known.

### Conditional correction and its scope (Sections 2 and 5)

At one fixed history, write $p=\pi_k$, $q=\mu_v$, $s(a)=\nabla_\theta\log p(a)$, with $q$ and $A(a)$ fixed during differentiation. If $q$ covers $p$,

$$
\mathbb E_q\left[\frac pq A s\right]=\mathbb E_p[A s].
$$

This identity changes the action distribution at that history. It does not replace the visited-prefix distribution or recompute the advantages of continuations under the current policy. Full-response correction involves products of ratios; token correction is not automatically an unbiased gradient of current-policy expected sequence reward.

A useful second-moment identity is

$$
\mathbb E_q\left[\left(\frac pq\right)^2\right]
=\sum_a\frac{p(a)^2}{q(a)}
=\exp D_2(p\|q).
$$

Use this to explain potential IS variance, without claiming the ratio moment alone determines gradient variance or that small ordinary KL bounds it. Correlation with advantages and scores matters.

### Forward, backward, and the loss signal (Section 3)

For trainer logits $z_\theta$ and a differentiable forward,

$$
\nabla_\theta L
=\left(\frac{\partial z_\theta}{\partial\theta}\right)^\top
\frac{\partial L}{\partial z}.
$$

Forward quantization changes the scored policy; backward precision or STE changes the computed derivative; SC/weighting changes the loss signal. For a discrete quantizer, identify the chosen surrogate derivative rather than presenting STE as the literal derivative of rounding. Master weights and optimizer moments govern update retention and publication independently of these choices. [4,8]

### SC and a controlled sampler anchor (Section 4)

The key identities [2] are

$$
g_{\rm raw}=\operatorname{Cov}_q(A,s)+\mathbb E_q[A]\mathbb E_q[s],
\qquad
g_{\rm SC}=\operatorname{Cov}_q(A,s).
$$

SC removes the second term, not the difference between $\operatorname{Cov}_q(A,s)$ and $\operatorname{Cov}_p(A,s)$. Group reward centering does not generally make $\mathbb E_q[A\mid h]=0$ at every prefix.

For independent softmax logits, $\mathbb E_q[s]=q-p$. Our explanatory calculation gives

$$
g_{{\rm SC},j}=q_j(A_j-\bar A_q),
\qquad
\nabla_{z_j}\left[-\beta D_{\rm KL}(q\|p)\right]=\beta(q_j-p_j).
$$

Thus the raw accidental component $\bar A_q(q-p)$ can be removed while a chosen $\beta(q-p)$ is added. With constant $A$, raw scoring gives $A(q-p)$, SC gives zero, and SC plus sampler KL leaves only the explicit regularizer. This is a local algebraic illustration, not an empirical training result or an Adam update model. The sampler is frozen for this gradient. [3]

For an action-dependent detached weight $w$, the correctly centered score is

$$
w(a)s(a)-\mathbb E_q[w s],
$$

not $w(a)(s(a)-\mathbb E_q[s])$. Exact $w=p/q$ makes the expectation term zero. Clipped or masked weights generally do not. This formula concerns token-local weights; a response-dependent gate needs the corresponding joint expectation. [2,14]

## Figures and scope decisions

The full draft contains three explanatory diagrams, a compact numerical callout, and one empirical source figure:

1. **One token, two gaps (Section 1, Figure 1).** Illustrative fixed-history probabilities separate same-version numerical disagreement from weight movement. Neither bracket is a measured KL.
2. **From rollout probability to trainer update (Section 3, Figure 2).** Rollout and trainer scores feed the corrected loss, then backward and the optimizer. Forward agreement, gradient precision, and parameter publication are distinct boundaries.
3. **TII forward values and backward validation (Section 5, Figure 3).** The comparison fixes weights, history, and sampling settings; the trainer uses aligned inference arithmetic and supplies independently validated backward support. Master weights and moments belong to the optimizer.
4. **Original REINFORCE Zero Figure 1 (Section 5, Figure 4).** Unmodified image from the November 6 announcement, preserving the original reasoning/KL panels and an explicit model/comparator caption. [Original image](https://vllm-project.github.io/assets/figures/2026-11-06-zero-mismatch-async-rl/figure1.png), downloaded 2026-11-08; SHA-256: 509e204e10d32076ef3344c0fcfcc40f8563005397096e9903d501a697dd75ae. No data were digitized or curves generated.
5. **Where a response becomes stale (Section 6, Figure 5).** Generation, environment/reward wait, queue, and learning, with optional in-flight refresh and token-level provenance. Durations are illustrative.

Section 4 contains a calculated gradient comparison table. The Section 3 configuration table is the compact link to the previous article's four-way ablation. Never merge curves from unrelated experiments into a head-to-head plot.

Maintain depth on the established correction baseline, forward/backward distinctions, SC/sampler KL, TII/REINFORCE Zero, and age provenance. Compress format basics, R3, FP16, and individual gating variants. The algorithm history stays in its sidebar. Leave OPD, a full PPO/GRPO course, framework installation, runnable recipes, and personal experiment reporting outside this article.

Before publication: review the prose and figure captions, pin any additional implementation revisions, recheck changing docs, and retain every performance claim's comparator and timing boundary. No additional personal trial details are required.

## References

Numbered references follow the earlier posts' format. Dates are included only where verified; living documentation was checked on 2026-11-08.

1. Yao et al. (2025). *On the Rollout-Training Mismatch in Modern RL Systems.* [Workshop report](https://opt-ml.org/papers/2025/paper116.pdf); [author's exposition](https://fengyao.notion.site/off-policy-rl).
2. Marek and Ryabinin (2026). *Score Centering Stabilizes Off-policy Reinforcement Learning.* [Paper](https://arxiv.org/html/2609.20807v1); [reference implementation](https://github.com/martin-marek/score-centering).
3. Zhang (2026). *Score Function Centering: Why a Stale Rollout Does Not Have to Be Importance Sampled.* [Derivation](https://normaluhr.github.io/en/p/score-function-centering-kl/); [experiments](https://normaluhr.github.io/en/p/score-function-centering-experiments/); [implementation](https://github.com/NormalUhr/score-function-centering-trainer-generator-kl).
4. Bai and Shen (2026-10-14). *Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven.* [Engineering report](https://vllm-project.github.io/2026/10/14/train-inference-consistency.html).
5. Hu et al. (2026). *REINFORCE Zero: Stable Asynchronous RL with Importance Sampling.* [Paper, pinned repository version](https://github.com/hijkzzz/REINFORCE-Zero/blob/ac19d189c4caaa4a5946146bd3eeaef3db244719/REINFORCE-Zero.pdf); [algorithm guide](https://github.com/hijkzzz/REINFORCE-Zero/blob/ac19d189c4caaa4a5946146bd3eeaef3db244719/ALGORITHM.md); [November 6 announcement](https://vllm-project.github.io/2026/11/06/zero-mismatch-async-rl.html).
6. Fu et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* [Paper v1](https://arxiv.org/html/2505.24298v1).
7. Dong, Goel, and Yu (2026-07-03). *Predicting and Controlling Staleness in Fully Asynchronous RL Training.* [Applied Compute analysis](https://appliedcompute.network/research/staleness-in-fully-async-rl).
8. InfiXAI, Ant Group AQ, SGLang RL, and Miles teams (2025-11-25). *Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL.* [Blog post](https://www.lmsys.org/blog/2025-11-25-fp8-rl/).
9. He and Thinking Machines Lab (2025-09-10). *Defeating Nondeterminism in LLM Inference.* [Technical article](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
10. Ma et al. (2025). *Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers.* [R3 paper v1](https://arxiv.org/html/2510.11370v1).
11. Qi et al. (2025). *Defeating the Training-Inference Mismatch via FP16.* [Paper v1](https://arxiv.org/html/2510.26788v1).
12. Zhong et al. (2026). *Diagnosing Training-Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference.* [VeXact paper v2](https://arxiv.org/html/2605.14220v2).
13. Wang and TorchTitan team (2026-08). *Defending Against the Training–Inference Numeric Mismatch in RL (Especially Linear Attention) — and Whether It Helps Async RL.* [Engineering analysis](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/).
14. Miles. *Score Centering.* [Living documentation](https://miles.radixark.com/docs/examples/infra-features/score-centering). Checked 2026-11-08.
15. Ling Team et al. (2025). *Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model.* [Ring-1T / IcePop report v1](https://arxiv.org/html/2510.18855v1).
16. Noukhovitch et al. (2024). *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models.* [Paper](https://arxiv.org/abs/2410.18252).
17. Espeholt et al. (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* [Paper](https://arxiv.org/abs/1802.01561).
18. Honari et al. (2026). *Align and Filter: Improving Performance in Asynchronous On-Policy RL.* [VACO paper v1](https://arxiv.org/html/2603.01365v1).
19. Yang et al. (2026). *Stale but Stable: Staleness-Adaptive Trust Regions for Stabilizing Asynchronous Reinforcement Learning.* [SAT paper v1](https://arxiv.org/html/2607.18722v1).
20. Yu et al. (2026). *Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where it Arises and How to Correct it.* [CIS paper v1](https://arxiv.org/html/2609.32444v1).
21. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* [Paper](https://arxiv.org/abs/1707.06347).
22. Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [GRPO paper v1](https://arxiv.org/html/2402.03300v1).
23. Yu et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* [Paper v1](https://arxiv.org/html/2503.14476v1).
24. MiniMax (2025). *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.* [CISPO report v1](https://arxiv.org/html/2506.13585v1).
25. GLM-5 Team (2026). *GLM-5: from Vibe Coding to Agentic Engineering.* [Report v1](https://arxiv.org/html/2602.15763v1).
26. Hou et al. (2026). *Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning.* [SAO paper v1](https://arxiv.org/html/2607.07508v1).
27. Xiaomi MiMo (2026). *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement.* [Report v1](https://arxiv.org/html/2610.11959v1).
28. Zhang (2026). *The Most Intuitive Way to Understand Score Function Centering: Understand Trainer-Generator Mismatch in Async RL from the Score Function.* [Practitioner account](https://huggingface.co/blog/NormalUhr/sfc).
