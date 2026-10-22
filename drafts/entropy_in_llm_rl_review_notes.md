# Entropy in LLM RL — structure and review notes

Updated: 2026-10-21.

Main text: [Entropy in LLM RL: Dynamics and Control](entropy_in_llm_rl_draft.md).

Status: converted to a [Quarto post](../posts/entropy-in-llm-rl/index.qmd), with this Markdown draft retained for review. The post uses the previous blog posts' numbered citation format. The source and rendered site are prepared for publication; drafts remain excluded from the site. No study-repository files are part of this revision.

## The argument

RL changes its own future training evidence. Concentration can make correct behavior reliable or make useful alternatives practically inaccessible. Added randomness can reveal better solutions or produce uniformly poor responses. The objective is effective learning, not a particular entropy curve.

The sequence is:

**Training failure → probability-update mechanism → three intervention points → evidence of better learning.**

The opening separates productive concentration, premature concentration, productive exploration, and unproductive dispersion. These are interpretations of joint behavior/performance evidence, not diagnoses from entropy direction or a universal healthy range.

## Structure and review priorities

| Section | Purpose | Main material |
| :--- | :--- | :--- |
| [1. When concentration helps—and when it stalls RL](entropy_in_llm_rl_draft.md#opening) | Establish why the topic matters before introducing solutions | On-policy feedback loop; Skywork Figures 19–20; joint entropy/performance patterns |
| [2. How RL updates change entropy](entropy_in_llm_rl_draft.md#dynamics) | Derive the mechanism rather than summarize slide results | Log-softmax derivative, four binary updates, softmax Jacobian, entropy derivative, covariance expansion |
| [3. Shaping the policy update](entropy_in_llm_rl_draft.md#task-update) | Explain which useful feedback survives | PPO → DAPO Clip-Higher versus CISPO; compact weighting/selection comparisons |
| [4. Entropy regularization: bonuses, targets, and feedback](entropy_in_llm_rl_draft.md#regularization) | Explain the extra preference and why its strength matters | Changed optimum, changing gradient balance, Skywork controller; brief SIREN contrast |
| [5. Exploration through rollout sampling](entropy_in_llm_rl_draft.md#collection) | Address missing training evidence | Temperature derivative, Nemotron-Cascade ablation, mixed-reward groups and DAPO sampling |
| [6. What better entropy dynamics should achieve](entropy_in_llm_rl_draft.md#outcomes) | Return to learning quality rather than curve shape | Per-prompt reliability and coverage, Pass@k example, measurement/strategy limitations |
| [Technical note](entropy_in_llm_rl_draft.md#measurement) | Preserve implementation distinctions without delaying the main argument | Cross-entropy under sampler mismatch; entropy-value versus entropy-gradient estimation |

For the next review, read Sections 1 and 6 together first. Then check whether Section 2 supplies enough derivation and whether each intervention section states the limitation it addresses.

## Source emphasis

The hierarchy is based on relevance, inspectable mechanisms, reported ablations, public implementation, and use in substantial training recipes. It is **not a verified citation-count ranking**. No numerical bibliometric claims are made. Vendor reports remain author-reported evidence, not independent replications.

| Treatment | Sources | Why they receive this space |
| :--- | :--- | :--- |
| Main empirical anchor | Skywork-OR1 | Paired coefficient/performance ablations and pinned controller code connect failure to intervention |
| Main update comparison | PPO, DAPO, MiniMax-M1 | Clear objective/gradient differences with concrete LLM training use |
| Mathematical connection | The Entropy Mechanism | Connects local entropy dynamics to proposed targeted updates; assumptions and batch-proxy boundary remain explicit |
| Main sampling evidence | Nemotron-Cascade | A specific temperature/stability/performance ablation |
| Brief foundations/comparisons | Dr. GRPO, SAC, GSPO | Explain normalization, sequential MaxEnt, and update granularity without adding full method chapters |
| Compact research extensions | Revisiting Entropy/PAR, high-entropy position selection, SIREN, EAD | Each supplies a distinct contrast; none is presented as a universal best practice |

The main article has 13 paper references plus three numbered references to pinned implementation code. Extensions are grouped by mechanism rather than given equal-length mini-tutorials.

## What was shortened or moved

- Removed the standalone measurement chapter. The main definition now begins the mathematical analysis.
- Moved sampled-surprisal/KL distinctions into the closing technical note and linked it from sampling.
- Replaced the formal strategy-label and sequence-entropy decompositions with a short evaluation discussion.
- Expanded the central derivatives while removing separate GSPO/SAPO/PAR method derivations and repeated numerical traces.
- Kept a compact controller recurrence, but removed its arithmetic trace and SIREN's separate candidate-distribution derivation.
- Reduced EAD to a research example; removed its exact schedule and full importance-sampling derivation.
- Removed ProRL/JustRL, MiMo, GLM-5, CURE, BAPO, and Pass@k-training digressions from the narrative.
- Retained four calculated comparisons: binary updates, clipping coefficients, mixed-group probability, and aggregate Pass@k.

These cuts prioritize a coherent argument; removal is not a claim that a source is invalid or low quality.

## Parked references

These remain available for later focused discussion, not as a required reading list for the article:

- [ProRL](https://arxiv.org/html/2505.24864v1) and [JustRL](https://arxiv.org/html/2512.16649v1): reference/optimizer resets versus a fixed recipe; not a matched comparison.
- [MiMo-V2.6](https://arxiv.org/html/2610.11959v1): sign-dependent eligibility as part of a broader training recipe.
- [SAPO](https://arxiv.org/html/2511.20347v2) and [BAPO](https://arxiv.org/html/2510.18927v1): smooth weighting and adaptive clipping.
- [CURE](https://arxiv.org/html/2508.11016v1): selected-prefix continuations and their sampling/cost distinctions.
- [Pass@k Training](https://arxiv.org/html/2508.10751v1): reward construction for group success.
- [GLM-5](https://arxiv.org/html/2602.15763v1): execution inconsistency as a separate cause of apparent collapse.
- [DeepSeekMath](https://arxiv.org/html/2402.03300v1): original GRPO background, unnecessary as another method explanation here.

## Technical boundaries to preserve

- All local entropy derivatives hold at a fixed prefix; the covariance uses actual logit displacement across vocabulary actions.
- The added PG/NPG comparison assumes true centered advantages and independent tabular logits: ordinary PG gives expected displacement proportional to pA, while natural PG gives A up to a common shift. Entropy changes remain first-order approximations.
- The logged-entropy decomposition is exact for normalized logging distributions with consistent weighting; estimated curves also contain sampling noise.
- The binary updates assume independently trainable logits and plain ascent, not Adam on a shared transformer.
- A fixed-prefix auxiliary bonus does not include the entire sequential MaxEnt gradient.
- PPO clipping removes a direct loss contribution, not all parameter changes affecting that token.
- CISPO's stop-gradient is part of its definition. The comparison cap is illustrative.
- First-update unit ratios require matched policies and scoring/execution; optimizer minibatches differ from physical accumulation.
- High-entropy selection and covariance-based suppression use different statistics. Neither guarantees semantic diversity.
- Skywork's stored state, gate, monitored estimator, differentiable entropy, and delayed application remain distinct.
- The controller formula follows the pinned code, including its strict below-target gate; it is not a guaranteed hard bound.
- Temperature monotonicity requires fixed logits and untruncated support. EAD is an empirical schedule, not a positional entropy theorem.
- Raw-policy update ratios alone do not correct temperature sampling. Exact response correction includes prefix likelihoods and requires support coverage; truncation generally introduces bias.
- Near binary saturation, both reward and entropy gradients vanish; entropy's relative contribution can grow even while absolute motion slows.
- All-correct and all-incorrect groups share zero centered advantages but imply different collection problems.
- At one fixed prompt and IID sampler, true Pass@1 and Pass@k are monotone together; aggregate trends can differ across prompts.
- Reported empirical findings and this article's explanatory synthesis are distinct. No LLM experiment was reproduced.

## Verification and publication

Selected source passages revisited for this rewrite: Skywork §4.5 and pinned controller/driver/actor code; DAPO §3.1 and surrounding objective; MiniMax-M1 §3.1; Nemotron-Cascade §5.2; Revisiting Entropy §5.1; the Entropy Mechanism covariance/intervention discussion. Other retained short pointers preserve their previously reviewed versioned scope. No claim of a fresh full-paper audit or citation census.

The follow-up review also checked The Entropy Mechanism §3.2 and EAD Appendix B. Reference [15] now covers actor update and controller update at pinned lines 1148–1169, replacing the unrelated rejection-sampling range.

Before publication:

- Review the new opening and reduced method selection.
- Figure 1 uses the original Skywork Figure 19 without modification, with attribution and adjacent analysis. The caption says “Original figure from” to avoid suggesting that the experiments were rerun. Source: https://arxiv.org/html/2505.22312v2/figures/different_ent_coeff_comparison.png. The arXiv license does not directly grant reuse rights; inclusion follows the user's approval of the commentary/fair-use approach, not a claim of explicit permission or a legal determination.
- Figures 2 and 3 are original SVGs: an RL feedback loop and an analytical gradient-coefficient comparison. The latter is calculated, not measured training data. Regenerate them with `python3 drafts/entropy_blog_figures.py`.
- Quarto conversion is complete; edit the post source for publication and keep the review draft synchronized.
- Render and inspect math, tables, navigation, and mobile widths in the final theme.
- Keep this companion note out of the published page.

The current checks cover local navigation, reference resolution, MathML conversion, and numerical/finite-difference validation of the retained derivations and examples. They do not replace final visual inspection.
