# Low-Precision RL: From Rollout to Update

Draft status: working markdown draft, not yet a Quarto post.

Working description: How precision choices across rollout, training, and
synchronization shape the policy optimized by LLM reinforcement learning.

## Scope

This became concrete for me while working through post-training support for
GLM-5.2 and Kimi K3. The difficult questions were not simply whether a model
could be loaded in FP8 or MXFP4. They were whether the trainer and rollout
engine represented the intended policy, whether an update became visible in
the correct form, and whether an observed mismatch came from quantization,
stale weights, routing, model state, or an implementation bug.

Low precision is often introduced as a hardware optimization. Smaller weights
use less memory, lower-precision Tensor Cores can execute more arithmetic, and
compressed checkpoints are easier to place across a cluster. Those benefits
matter, but they do not capture what makes low precision unusual in language
model reinforcement learning.

An RL system normally executes the policy in at least two places. A rollout
engine generates tokens and records their probabilities. A training engine
later scores those tokens, constructs an objective, and updates the model. Even
when both engines use the same weight version, differences in precision,
scaling, kernels, routing, or weight conversion can make their output
probabilities differ.

That difference does not automatically make the RL update invalid. It does
mean that we need to identify which policy generated the data, which policy the
trainer evaluates, and how a new weight or adapter version reaches rollout.

This post develops the topic in three parts:

1. the conceptual foundation: why RL has separate rollout and training
   executions, and why numerical differences matter;
2. the systems view: how precision enters rollout, FProp, DGrad, WGrad,
   optimizer state, and live synchronization;
3. the practical view: what GLM-5.2 and Kimi K3 reveal about correctness,
   memory, hardware, and model-specific support.

The narrative follows two objects around one RL cycle:

> A token moves from rollout to training. An updated weight moves from training
> back to rollout.

Low-precision RL works when we can identify, measure, and control both paths.

## One Token, Two Executions

For a prefix \(s_t\), the rollout engine samples a token from a behavior policy:

$$
a_t\sim\mu(\cdot\mid s_t).
$$

The generated token is now part of the training data. The trainer feeds the
same token and prefix through its current policy and recomputes:

$$
\log\pi_\theta(a_t\mid s_t).
$$

Many RL objectives compare the two policies through a ratio:

$$
\rho_t
=
\frac{\pi_\theta(a_t\mid s_t)}
     {\mu(a_t\mid s_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\mu(a_t\mid s_t)
\right).
$$

If the policy has changed because of an optimizer update, a ratio away from one
is expected. In an asynchronous system, the rollout policy may also be several
versions behind the trainer. But the trainer and rollout engine can disagree
even at the same weight version:

    weight version W(t)
       ├─ rollout format, scales, layout, and kernels  → rollout policy μ(t)
       └─ trainer format, scales, layout, and kernels  → trainer policy π(t)

Let \(\pi_t\) be the current trainer policy, and let \(\mu_\tau\) be the
behavior policy that generated the token, where \(\tau\leq t\). To separate
two common causes of disagreement, define \(\mu_t\) as the policy the rollout
engine would execute after receiving weight version \(t\). For the same token
and prefix:

$$
\begin{aligned}
\log\pi_t(a_t\mid s_t)
-
\log\mu_\tau(a_t\mid s_t)
=
\underbrace{
\log\pi_t(a_t\mid s_t)
-
\log\mu_t(a_t\mid s_t)
}_{\text{same-version precision mismatch}}
+
\underbrace{
\log\mu_t(a_t\mid s_t)
-
\log\mu_\tau(a_t\mid s_t)
}_{\text{rollout staleness}}.
\end{aligned}
$$

The first term is the **same-version precision mismatch**. The second is
**rollout staleness**. The algebra is exact only when all three scores refer to
the same token and prefix. It is still a debugging decomposition rather than a
promise that the two causes can be measured independently. In a nonlinear
model, an early numerical difference can change a router decision or recurrent
state, which then changes the later computation path.

MoE models make this especially visible. If two router scores lie close to a
top-\(k\) boundary, a small quantization change can select a different expert.
The resulting difference is no longer only a slightly perturbed matrix
multiplication. The two executions have entered different branches of the
model.

The first low-precision RL question is therefore:

> How much of the trainer–rollout log-probability gap comes from the precision
> configuration, and how much comes from rollout staleness?

Three related diagnostics should remain separate:

| Diagnostic | Comparison | What it answers |
| :--- | :--- | :--- |
| Direct trainer–rollout log-probability difference | \(\log\pi_t(a_t\mid s_t)-\log\mu_t(a_t\mid s_t)\) for identical tokens and prefixes | How large is the same-version mismatch? Report absolute differences, tails, token-position dependence, and routing agreement rather than only a signed mean. |
| Trainer–rollout KL diagnostic | The same trainer and rollout policies, aggregated over sampled tokens | Does the end-to-end gap remain near an established baseline or grow over time? This is a diagnostic unless the objective explicitly uses it. |
| Reference-policy KL | The trained or quantized policy versus a separate reference policy, often BF16 | How far has the policy moved from that reference? It does not directly measure trainer–rollout agreement. |

The importance ratio is constructed from the first comparison. A clipping or
truncation statistic then reports how often that ratio reaches the objective's
control boundary. It measures a consequence of mismatch; it does not identify
whether the cause is precision or staleness.

<!-- TODO: Add an original figure showing the rollout and trainer executions,
the weight version, and the synchronization path. -->

## From One Linear Layer to an RL Update

Before following a full RL update, it helps to make the word *precision* more
specific. A number format is only one part of a quantization scheme.

### Format, scale, and execution

Consider a high-precision value \(x\) represented on a lower-precision grid:

$$
\widehat{x}
=
s\,Q\left(\frac{x}{s}\right),
$$

where \(Q\) rounds and clamps to the available codes and \(s\) determines which
real values those codes cover.

The format determines the local number system. The scale moves that number
system to a useful range. Scale grouping determines which values must share
that range. One tensor-wide scale, one scale per row, one scale per block, and
one scale per token can produce different reconstructed values even when the
payload format is unchanged.

In this post, **format** means the encoded datatype, such as FP8 or MXFP4. A
**quantization scheme** includes the format, scale grouping, rounding, and
exceptions. A **precision configuration** says where those schemes are used in
rollout and training. The **synchronization path** says how a new weight or
adapter version reaches the rollout engine.

A small example makes the trade-off visible. Suppose \(1\), \(2\), and \(100\)
share one scale chosen to preserve \(100\). The smaller values receive much
less of the available resolution. Placing them in a separate group can
represent them more accurately, but requires more scale metadata and more
work to compute and apply it.

Low-precision execution also needs several distinctions:

| Question | Examples |
| :--- | :--- |
| Format | BF16, FP8, MXFP8, MXFP4, NVFP4, INT4 |
| Scale grouping | per tensor, row, block, token, or expert tensor |
| GEMM inputs | low-precision operands or dequantized BF16 values |
| Accumulation and output | FP32 or BF16 even when inputs are lower precision |
| Persistent storage | master weight, trainer parameter, rollout artifact |
| Communication | BF16, FP8, packed low-bit values, or adapter tensors |

This is why an FP8 GEMM does not imply an entirely FP8 model. The persistent
trainer weight, optimizer moments, residual stream, normalization layers, GEMM
outputs, and selected sensitive operations may all remain wider.

The main configurations discussed below use these formats and scale layouts.
The rows describe the cited implementations, not universal definitions of the
format names.

| Configuration | Payload and scaling | Role in this post |
| :--- | :--- | :--- |
| Blockwise FP8 | E4M3 values with FP32 scales; the Unified FP8 setup uses \(1\times128\) activation tiles and \(128\times128\) weight blocks | SGLang rollout and eligible Transformer Engine trainer GEMMs on Hopper |
| INT4 QAT / W4A16 | grouped INT4 weights with BF16 activations; the trainer uses quantize–dequantize fake quantization | weight-resident rollout on Hopper with BF16 trainer GEMMs |
| MXFP8 | E4M3 values with one E8M0 scale for every 32 consecutive values | rollout and eligible trainer GEMMs on Blackwell |
| MXFP4 | packed microscaling FP4 weights; the exact packing and kernel layout are model-specific | K3's frozen rollout weights |
| NVFP4 | E2M1 values, one E4M3 scale per 16 values, and an outer FP32 scale | selective routed-expert rollout and FProp on Blackwell |

### FProp, DGrad, and WGrad

For one trained linear layer:

$$
Y=XW^\top.
$$

The forward pass and the two main backward matrix multiplications consume
different operand pairs:

$$
\operatorname{FProp}(X,W)\rightarrow Y,
$$

$$
\operatorname{DGrad}(G_Y,W)\rightarrow G_X,
$$

$$
\operatorname{WGrad}(G_Y,X)\rightarrow G_W.
$$

FProp determines the numerical function used to score the current tokens.
DGrad passes loss sensitivity to earlier layers. WGrad produces the parameter
gradient consumed by the optimizer.

These operations do not have to use the same precision. A trainer can match a
quantized rollout in FProp while keeping DGrad and WGrad in BF16, provided the
trainer forward reproduces the rollout's quantizer, grouping, layout,
exceptions, and relevant execution semantics closely enough. That aligns the
policy used for the current objective without claiming four-bit or eight-bit
backward computation.

Fake quantization makes the distinction even clearer:

$$
\widehat W=DQ(Q(W)).
$$

A BF16 GEMM can use \(\widehat W\). The computation contains real quantization
error because it sees the reconstructed grid values, but the GEMM itself does
not consume low-bit operands. Quantization-aware training then needs a
separate choice for how gradients cross the rounding operation, commonly a
straight-through estimator.

### Tracing one RL update

The complete RL update adds rollout, optimizer state, and synchronization to
the three trained-linear operations:

| Component | Main question |
| :--- | :--- |
| Rollout | Which behavior policy generated the tokens? |
| Trainer FProp | How does the current trainer policy score them? |
| DGrad | Which weight representation propagates sensitivity backward? |
| WGrad | Which activation representation forms the weight gradient? |
| Optimizer | Which parameter and moment representations create \(W_{t+1}\)? |
| Convert and synchronize | How does \(W_{t+1}\) become the next rollout policy? |

The weights are not the only state in this process. The trainer maintains
optimizer state. Forward and backward passes create temporary activations.
Autoregressive generation maintains request state, including the accepted
token prefix, KV cache, recurrent or convolution state, and sometimes routing
or draft-model state. A request must remain paired with the weight or adapter
version used to score and extend it.

The token travels through the first half of this loop. The updated weight
travels through the second:

    rollout
       │ sampled token and behavior log-probability
       ▼
    trainer FProp
       ├─ DGrad
       └─ WGrad
             │
             ▼
          optimizer creates W(t+1)
             │
             ▼
       convert and synchronize
             │
             └──────────────→ next rollout version

The same \(W_t\) may exist simultaneously as an FP32 optimizer master, a BF16
trainer parameter, rowwise and columnwise quantized GEMM operands, and a packed
rollout tensor. These are representations of one weight version, not four
different updates. The version changes when the optimizer creates
\(W_{t+1}\).

<!-- TODO: Replace the text diagram with an original figure. -->

## Where Precision Enters the RL Step

There is no single low-precision RL configuration. Different designs place
low precision in different parts of the update.

::: {.column-page}
| Design | Rollout | FProp | DGrad | WGrad | Optimizer | Synchronization path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified FP8 | matching blockwise FP8 | native blockwise FP8 for eligible GEMMs | native blockwise FP8 for eligible GEMMs | native blockwise FP8 for eligible GEMMs | higher-precision master and moments | rebuild and transfer the matching FP8 policy |
| INT4 QAT | real W4A16 | BF16 GEMM over fake-quantized weights | BF16 using reconstructed weights | BF16 from saved activations; STE maps the gradient through weight quantization | higher precision | quantize and pack updated INT4 weights |
| MXFP8 | matching MXFP8 | native MXFP8 for eligible GEMMs | native MXFP8 for eligible GEMMs | native MXFP8 for eligible GEMMs | higher precision | rebuild codes, scales, layouts, and exceptions |
| Published NVFP4 RL configuration | NVFP4 routed experts with exceptions | native NVFP4 expert GEMMs | BF16 using original or dequantized weights | BF16 using original or dequantized activations | higher precision | reproduce the quantizer, layout, and specified exceptions |
| LoRA over frozen base weights | packed low-bit or mixed-precision base plus BF16 adapter | BF16 frozen weights plus BF16 adapter | BF16 sensitivity through the frozen weights and adapter paths | adapter tensors only | adapter state only | synchronize the adapter; leave the rollout base unchanged |
:::

The hardware generation filters which native paths are available:

| Hardware family | Natural starting point | Important boundary |
| :--- | :--- | :--- |
| Hopper, including H200 | blockwise FP8 training; W4A16 rollout when weight residency is the goal | no native Blackwell microscaling path |
| Blackwell, including B200/B300/GB300 | MXFP8 and NVFP4 training paths; packed MXFP4 rollout where model kernels support it | hardware support does not guarantee model, kernel, framework, or complete RL support |

The table should not be read as a ranking or as a promise of equal treatment
in the rest of the post. It is a map of the design space. Unified FP8 is traced
in detail below because it provides a clean example of how to follow precision
through the entire RL step. The other rows remain comparison points and return
at different depth in the model-specific sections.

Unified FP8 and MXFP8 try to align rollout and trainer execution while also
using native low-precision trainer GEMMs. INT4 QAT adapts a higher-precision
trainer to a low-bit rollout function without claiming INT4 backward speed.
The published NVFP4 example uses four-bit expert execution in the forward path
while keeping a wider backward for stability. LoRA over frozen base weights
changes the update path: instead of converting and moving the full model after
every optimizer step, it moves only the trainable adapter.

Three hardware lessons carry across these designs:

1. low-bit storage is not the same as low-bit computation;
2. native hardware support is not the same as complete model support;
3. theoretical Tensor Core throughput is not full RL-step throughput.

Scale construction, transposes, packing, communication, small RL
microbatches, ragged expert GEMMs, and CPU launch gaps can dominate the
measured system.

### A closer look at Unified FP8

Unified FP8 is a useful first deep dive because it exposes three questions that
are easy to collapse into one: which values define the forward policy, which
GEMMs actually run in FP8, and which representation is stored or communicated
between workers.

#### Matching the forward comes first

The published Hopper setup uses an SGLang blockwise FP8 rollout policy and
Transformer Engine blockwise FP8 for eligible trainer GEMMs. Both use E4M3
values with FP32 scales, \(1\times128\) activation tiles, and
\(128\times128\) weight blocks. Accumulation, normalization, residual paths,
optimizer state, and unsupported operations can remain wider.

Four trainer ablations help separate policy alignment from native execution.
The rollout remains blockwise FP8 in every case:

| Trainer case | FProp | DGrad and WGrad | Question isolated |
| :--- | :--- | :--- | :--- |
| BF16 baseline | BF16 values and GEMMs | BF16 | How large is the original same-version trainer–rollout gap? |
| Forward fake quantization | quantized–dequantized FP8-grid values in BF16 GEMMs | BF16 | Is matching the rollout's quantized forward values sufficient? |
| Forward and backward fake quantization | quantized–dequantized values in BF16 GEMMs | quantized–dequantized operands in BF16 GEMMs | What changes when the backward also sees FP8-grid values? |
| Real FP8 | blockwise FP8 operands and FP8 GEMMs | FP8-capable DGrad and WGrad | Does native FP8 execution behave like the fake-quantized configuration? |

The important result is conceptual as much as empirical. Matching the
quantized forward is the first-order requirement for same-version
trainer–rollout agreement, because FProp defines the probabilities used in the
current RL objective. Backward precision instead changes the next parameter
update, so its effects appear through optimization stability, throughput, and
future policies.

This is also where the probability diagnostics need precise names. Direct
same-token log-probability differences and trainer–rollout KL measure the
forward agreement being targeted. Reference-policy KL measures movement from
a separate reference. Importance-ratio clipping reports how the objective
responds to the disagreement; it does not explain its source.

The setup itself provides a useful test of this workflow. In Miles,
`--transformer-impl transformer_engine`, `--fp8-format e4m3`, and
`--fp8-recipe blockwise` enable real FP8 execution for supported operators.
The optional `--fp8-param-gather` flag changes a different lifecycle point.
Without it, eligible primary weights and their data-parallel AllGather can
remain BF16 before the weights are quantized online for real FP8 GEMMs. With
it, supported primary weights and the AllGather use FP8. The master parameters
and optimizer moments remain wider in both cases. The distinction is a small
but representative example: a flag that sounds like “more FP8” may affect
parameter storage and communication without changing the precision of FProp,
DGrad, or WGrad. The current implementation also ties this option to
Transformer Engine's FusedAdam path rather than the commonly used Megatron CPU
Adam offload path, so the exact compatibility constraint should be rechecked
as the software changes.

#### Higher-precision state and conversion boundaries remain

In the current integration, FP8 coverage applies to eligible Transformer
Engine `Linear` and `GroupLinear` operations. Embeddings, the language-model
head, unsupported operators, optimizer master state, and selected sensitive
paths remain in higher precision. FP8 codes and scales, transposed operand
copies, workspaces, saved activations, and higher-precision state also explain
why end-to-end memory does not simply fall in proportion to the payload width.

The live synchronization path is another separate boundary. The current Miles
example can dequantize the trainer representation to BF16 for transfer and let
SGLang quantize it into its rollout layout; its distributed checkpoint remains
BF16. These conversions may be worth optimizing, but they do not decide
whether the trainer's eligible GEMMs use FP8.

In this post, then, *Unified FP8* means that rollout and trainer FProp target
the same blockwise quantized policy while eligible trainer forward and backward
GEMMs execute in FP8. It does not mean that every tensor, operation,
communication, and persistent state has become FP8.

## GLM-5.2 and Kimi K3: Two Different Update Paths

GLM-5.2 and Kimi K3 both use low precision in rollout, but they close the RL
loop in different ways.

### GLM-5.2: BF16 training and FP8 rollout

The current GLM-5.2 Miles practice, checked on 2026-09-01, uses a BF16 Megatron
trainer and an FP8 SGLang rollout policy on separate GPU pools. The rollout
side also uses an FP8 E4M3 KV cache. Model-weight precision and cache precision
are separate decisions: one changes the executed network, while the other
changes how attention state is stored during generation.

Its precision configuration is approximately:

| Component | GLM-5.2 practice |
| :--- | :--- |
| Rollout | FP8 SGLang behavior policy with FP8 KV cache |
| Trainer FProp | BF16 Megatron |
| DGrad | BF16 Megatron |
| WGrad | BF16 Megatron |
| Optimizer | higher-precision full-parameter update |
| Convert and synchronize | asynchronous full-weight refresh |

Applying the earlier decomposition, the trainer–rollout log-probability gap
contains same-version BF16-versus-FP8 precision mismatch and rollout
staleness caused by asynchronous updates. Both can appear in the same
importance ratio, even though they require different diagnostics and
interventions.

Truncated importance sampling can limit the effect of extreme ratios on the
update. It does not make the BF16 trainer equal to the FP8 rollout engine,
restore the stale behavior policy, or prove that both systems selected the
same experts. Precision mismatch and rollout staleness still need separate
diagnostics.

### Kimi K3: packed MXFP4 rollout weights and a BF16 adapter

Kimi K3 uses a different update path. The disclosed RL practice keeps frozen
BF16 weights in the Megatron trainer, while SGLang executes the packed MXFP4
checkpoint directly on Blackwell. Only a BF16 LoRA adapter is updated and
synchronized.

For one adapted projection:

$$
Y_T
=
\mathcal L_T(X_T;W_{\mathrm{BF16}})
+
\frac{\alpha}{r}X_TA_t^\top B_t^\top,
$$

$$
Y_R
=
\mathcal L_R(X_R;W_{\mathrm{MXFP4}})
+
\frac{\alpha}{r}X_RA_t^\top B_t^\top.
$$

The separate \(X_T\) and \(X_R\) matter. Earlier BF16 and MXFP4 operations may
already have produced different hidden states. The equations describe one
adapted projection, not an additive decomposition of the complete nonlinear
model. Here \(\mathcal L_T\) is the trainer's BF16 linear operation and
\(\mathcal L_R\) is SGLang's packed MXFP4 linear operation.

Its precision configuration is:

| Component | Kimi K3 practice |
| :--- | :--- |
| Rollout | packed MXFP4 frozen weights plus a separate BF16 LoRA path |
| Trainer FProp | frozen BF16 weights plus BF16 LoRA |
| DGrad | BF16 sensitivity passing through the frozen weights and adapter paths |
| WGrad | adapter tensors only |
| Optimizer | BF16 LoRA state |
| Convert and synchronize | install the adapter update; do not rebuild the frozen weights |

The base still participates in DGrad because sensitivity must pass through it
to reach earlier adapted layers. It does not receive WGrad or optimizer state.

This design avoids merging an adapter into a complete BF16 base, requantizing
the full 2.8-trillion-parameter model, and redistributing a new MXFP4 artifact
after every step. The low-bit base remains unchanged; the adapter is the
moving part.

“Only the adapter moves” still describes a distributed model update. Megatron
adapter shards must be reconstructed into the rollout TP/EP ownership and
tensor layouts, transferred in bounded groups, installed, and acknowledged.
The rollout must activate adapter version \(t+1\) only after all of its tensors
have been installed; otherwise one trajectory can observe a mixture of adapter
versions \(t\) and \(t+1\). Successful byte transfer alone does not prove that
the rollout kernel reads the new adapter.

That changes the validation target. Same-version BF16-versus-MXFP4 precision
mismatch is expected at initialization. The question is whether the
trainer–rollout gap remains near a measured baseline or grows because of
missing adapter tensors, mixed versions, execution differences, or
request-state errors.

K3 also makes the distinction between persistent request state and temporary
activations concrete. KDA recurrent state, the ShortConv window, and MLA KV
records persist as a request is decoded. The attention-residual bank is a
temporary activation carried across the relevant pipeline boundary. Correct
weights paired with the wrong prefix, request state, or adapter version still
produce the wrong continuation probabilities.

Memory accounting is similarly phase-based:

$$
M_{\mathrm{peak}}
=
M_{\mathrm{persistent}}
+
\max\left(
M_{\mathrm{rollout\text{-}only}},
M_{\mathrm{trainer\text{-}only}},
M_{\mathrm{sync\text{-}only}}
\right).
$$

Persistent memory includes the packed MXFP4 rollout weights, installed
adapter, process groups, and runtime metadata. Trainer and rollout phase-local
peaks should not be added when the system alternates between the two phases,
but neither should the idle yet resident rollout weights disappear from the
accounting. Transition buffers and adapter transport can create a peak that
neither steady-state number exposes.

<!-- TODO: Add a side-by-side GLM-5.2 versus K3 lifecycle figure. -->

### Side-by-side comparison

::: {.column-page}
| Question | GLM-5.2 | Kimi K3 |
| :--- | :--- | :--- |
| What changes during training? | all BF16 parameters | BF16 LoRA adapter |
| What generates rollouts? | asynchronously refreshed FP8 policy | frozen packed MXFP4 weights plus current adapter |
| What moves after an update? | all converted rollout weights | adapter tensors |
| Placement and schedule | dedicated trainer and rollout GPU pools; fully asynchronous | colocated GPUs; rollout-active and trainer-active phases alternate |
| Version boundary | asynchronous full-parameter refresh | atomic adapter installation; frozen weights remain unchanged |
| Expected mismatch | same-version precision mismatch plus rollout staleness | stable BF16-versus-MXFP4 gap plus adapter or runtime errors |
| Main control | truncated importance sampling and separate staleness/mismatch measurements | versioned adapter installation and fixed-prefix same-version comparisons |
| Main systems pressure | full-weight refresh and asynchronous lag | phase memory, tensor mapping, and adapter correctness |
:::

These are not two implementations of one configuration. They place precision
and synchronization differently within the same RL update.

## When the Loop Breaks

Low-precision bugs rarely identify themselves by format. They appear as
changes in probabilities, routes, gradients, memory, or utilization.

| Symptom | First places to inspect |
| :--- | :--- |
| Trainer and rollout disagree immediately | token alignment, weight version, quantizer, scales, layout |
| Difference begins at one MoE layer | selected routes, router inputs, expert packing |
| Initial load matches but a live update fails | shard reconstruction, fusion, packing, version switch |
| Adapter transfer completes but behavior is unchanged | tensor manifest, active adapter version, kernel read path |
| Mismatch grows with context length | KV cache, recurrent state, convolution window |
| Gradients spike under a matched forward | backward operands, optimizer, sensitive-layer exceptions |
| Expected memory saving disappears | retained BF16 copies, transpose caches, transition buffers |
| Low-precision training is slower | small GEMMs, scale kernels, launch gaps, communication |

Validation should move from the smallest deterministic object to the full RL
system:

1. **Record the precision configuration.** State the format, scales, grouping,
   exceptions, trainable parameters, synchronization method, and allowed
   staleness.
2. **Test the quantizer and converter.** Compare values, scale bytes, grouping,
   padding, packing, and tensor names.
3. **Test one operator.** Separate quantization error from kernel and layout
   error.
4. **Score a fixed prefix.** Make both engines score identical tokens before
   independently sampling.
5. **Test a live update.** Use a known synthetic change and prove that the
   rollout kernel executes the new version.
6. **Replay one deterministic training step.** Hold the trajectory fixed while
   changing forward or backward precision.
7. **Run bounded RL.** Track reward, entropy, gradients, direct
   trainer–rollout differences, clipping, routing, and rollout staleness.
8. **Profile performance.** Measure the actual RL shapes and phase transitions
   only after correctness is understood.

A checksum can show that bytes arrived. It cannot show that a kernel used
them. A stable same-version trainer–rollout KL diagnostic can show that the
end-to-end gap remains near an established baseline. A reference-policy KL
answers a different question. Neither localizes a remaining difference to
quantization, routing, state, or versioning. The evidence has to be layered.

## Why One Precision Configuration Does Not Transfer Automatically

Larger models rarely apply one format uniformly. Routed experts may use
NVFP4, while attention, routers, shared experts, final layers, and custom
recurrent operators remain wider. The actual operator coverage matters more
than the format named in the checkpoint.

Qwen3.8 is a useful complement to K3. Its rollout checkpoint uses native
NVFP4 only for linear layers in routed-expert MLPs, while LoRA trains attention
projections. This is a concise example of why the quantized operator scope and
the trainable operator scope do not have to match.

Other model-specific details matter as well. KV caches, recurrent state,
convolution windows, routes, and temporary pipeline activations can affect the
next-token probabilities. Full-model conversion, packed layouts, transition
buffers, and version activation can determine whether an update fits in memory
and becomes visible at the right time.

A new model should therefore be evaluated by tracing its actual RL update:
which behavior policy generates tokens, how trainer FProp scores them, which
representations DGrad and WGrad use, what the optimizer changes, and how that
new version reaches rollout. A shared format label is not enough to transfer
K3's tensor mapping, memory assumptions, or expected same-version precision
mismatch to another model.

## Summary

Low precision becomes an RL problem because the system that generates tokens
and the system that trains on them may use different precision configurations.
The same weight version does not guarantee the same probabilities, routes, or
state transitions.

A practical analysis traces precision through rollout, trainer FProp, DGrad,
WGrad, optimizer state, and synchronization. It also keeps request state and
weight or adapter versions explicit. A format name describes only part of this
process.

GLM-5.2 and Kimi K3 illustrate two different ways to close the loop. GLM-5.2
updates all BF16 parameters and asynchronously exposes an FP8 rollout version.
K3 keeps frozen BF16 trainer weights and packed MXFP4 rollout weights, then
synchronizes only a BF16 adapter. Their mismatch, memory, and validation
targets are therefore different.

The token moves from rollout to training. The updated weight moves back. The
engineering task is to know which policy exists at every step of that cycle.

## Working References {.unnumbered}

The reference list and evidence boundaries will be completed as the draft is
developed.

1. [Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL](https://www.lmsys.org/blog/2025-11-25-fp8-rl)
2. [Squeezing 1TB Model Rollout into a Single H200](https://www.lmsys.org/blog/2026-01-26-int4-qat)
3. [Towards Blackwell-Native 8-bit and 4-bit RL](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/)
4. [The 4-bitter Lesson: Balancing Stability and Performance in NVFP4 RL](https://humansand.ai/blog/nvfp4-rl)
5. [SGLang and Miles Add Day-0 Support for Kimi K3](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)
6. [Miles GLM-5.2 reference recipe](https://github.com/radixark/miles/tree/main/examples/experimental/openenv/glm52_tbench2)
7. [Transformer Engine FP8 and FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
8. [Qwen3.8 day-zero support](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support)
9. [RadixArk Qwen3.8 NVFP4 checkpoint card](https://huggingface.co/RadixArk/Qwen3.8-2.4T-A95B-NVFP4)
10. [Miles FP8 training examples](https://github.com/radixark/miles/blob/main/examples/infra_features/low_precision/README.md)
