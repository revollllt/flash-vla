# The official Pi0.5 gate, on a machine without the training stack

Status: implemented.

## Problem

`eval.pi05.reference` runs OpenPI's forward and the Target's in one process.
That is the right shape when the two can share an interpreter, and on the RTX
5090 they cannot. The deployment environment there is the pinned flash-vla one
-- torch, TileLang, safetensors, sentencepiece -- and OpenPI's import path
reaches `openpi.training.config`, which pulls JAX, Flax, LeRobot, `tyro` and a
`transformers` fork. OpenPI also pins `numpy<2` against the environment's 2.5.2.

Installing that into the pinned environment would move the environment every
latency number on this machine is recorded against, to obtain a number about
accuracy. The two interpreters on the machine that *do* have OpenPI cannot run
the model instead: one is torch 2.7.1+cu126, whose arch list stops at `sm_90`,
and the other is a fork whose `Pi0Config` lacks the field this adapter's loader
passes.

## Decision

Split the comparison across two interpreters, as `eval.lingbot` already splits
its oracle. `eval.pi05.parity capture` runs the official forward and writes the
tensors *and the fixture it used*; `eval.pi05.parity compare` replays that
fixture through the Target and judges it against `eval.tolerances`. Neither step
needs the other's imports.

`OPENPI_PI05_MODULE` names the module the official forward comes from. The
default is upstream OpenPI's own path. A vendored snapshot of the same PyTorch
subtree -- the JAX-free half of OpenPI, which is all the forward needs -- is
selected by pointing it elsewhere, and `official.module_provenance` records
which module, file and commit actually ran rather than assuming a repository.

Two smaller decisions follow from the same constraint:

- `models.pi05.openpi.converted_checkpoint` loads a checkpoint OpenPI's
  converter has already written, without OpenPI. The conversion is a one-off
  that happened elsewhere; what remains is a state dict, and
  `target_checkpoint` already rejects one whose keys or shapes are not this
  architecture's. The factory keeps it a separate argument from `checkpoint`,
  because that path resolves and validates a *named* upstream config and this
  one cannot -- it states the tensor ABI and the caller states the identity.
- Inputs are rounded to bf16-representable values in the fixture. Both sides
  consume them in bf16, and an input the two rounded differently reads as a
  model difference that is really a fixture difference.

## Consequences

The official gate runs on a machine that hosts no training stack, and the
environment a latency number is recorded against stays the one the README pins.
The cost is that the oracle is a file rather than a live comparison: a capture
and a compare can drift apart if the checkpoint or the fixture changes under
them, so the oracle carries its own fixture, its config, its valid prefix
length and the provenance of the module that produced it, and `compare` refuses
to proceed when the Target tokenizes a different prompt than the capture did.

## Verification

Three weight sets, all at 3 views, a 200-token prompt with 135 valid tokens,
chunk 50, 10 steps, both sides on one RTX 5090 (`rtx5090/pi05`, all-torch route).
Layer 0 is held to the registry's tight pair, the deepest layer to its loose one,
and the action chunk -- read at full depth -- to the loose one as well.

| weights | prefix KV L0 | prefix KV L17 | worst step | action chunk | gate |
|---|---:|---:|---:|---:|---|
| random, seed 0 | 0.9999172 | 0.9965790 | 0.00027 | 0.9999855 | pass |
| `pi05_belt_cup` | 0.9998225 | 0.9992419 | 0.00019 | 0.9999766 | pass |
| `pi05_base` | 0.9996728 | 0.9871325 | 0.00815 | 0.9999193 | **fail** |

`pi05_belt_cup` is this Target's checkpoint of record and passes the registry's
thresholds as they stand. Random weights establish that the two implementations
agree, not that the policy is good.

`pi05_base` misses the layer-0 pair, and the section below is why that is not an
accuracy deficiency: against an fp32 run of the same weights this route is the
closer of the two bf16 implementations to exact. It is kept as a data point, not
as the gate.

## What the difference is made of

The gap was bisected rather than attributed, by capturing the official SigLIP
tower layer by layer on the same fixture (`eval.pi05.parity` writes the KV cache
and the chunk; this used forward hooks beside it) and comparing the Target's
residual stream after every layer:

| stage | rel_rms |
|---|---:|
| patch embedding | 0.0019 |
| vision layer 0 | 0.0057 |
| vision layer 12 | 0.0165 |
| vision layer 18 | 0.0204 |
| vision layer 26 | 0.0161 |

The input agrees to 0.19% and no layer steps. Each layer adds
`sqrt(0.0057^2 - 0.0019^2)` = 0.54%, which is 2.8 bf16 half-ulps (2^-9 = 0.195%)
and so about eight independent roundings -- the number of places a SigLIP layer
rounds. That accumulates like the square root of depth to 1.61% at the tower's
output.

The hand-off then amplifies it: the tower's own final LayerNorm takes 1.61% to
2.60%, and backbone layer 0 inherits that. Per-token sigma heterogeneity
accounts for about two thirds of the step (sigma spans 7.8 to 27.0 with a median
of 11.3, and a uniform absolute error through that spread predicts 1.75%); the
remaining factor of ~1.5 is not decomposed. It does not change where the error
comes from, which is the 27 layers ahead of it. 2.60% is just outside the ~2.1%
the layer-0 cosine gate implies, which is the criterion this route misses.

Three candidate causes were checked and two were eliminated by measurement:

- **Attention.** Upstream SigLIP runs SDPA at scale `72**-0.5` with
  `gelu_pytorch_tanh`, which is what this backend runs. No implementation
  difference.
- **LayerNorm epsilon.** Upstream's `layer_norm_eps` is 1e-6 and this project's
  vision LayerNorm -- H100's TileLang kernel included -- uses 1e-5. That is a
  real deviation, and it is not this one: at the measured per-token variance
  (median 0.447 at the embeddings, higher everywhere after) it moves
  `rsqrt(var + eps)` by 1e-5 relative, against the 5.7e-3 being explained.
- **Rounding count.** Upstream rounds twice at each residual site and applies
  GELU to the BF16-rounded pre-activation. Spelling those as `x @ w + bias + res`
  rounds a third time. Matching the count -- bias in the GEMM epilogue, GELU on
  the rounded pre-activation, residual added once -- moved the whole curve down
  by 2.5% at every depth and improved the action chunk on both real checkpoints.
  That is the form the backend now ships.

What remains is rounding *order*: cuBLAS split-k, the SDPA kernel choice, and
this Target batching three views where upstream embeds one at a time. Two
independent bf16 implementations of a 27-layer residual tower agree to about
0.5% per layer, and closing that means replicating the other side's kernel
decomposition, not removing roundings.

That distinction matters for what happens next, and it was measured rather than
argued: the official tower was run again in fp32 on the same weights and fixture,
as a third point.

| | vision output | after post_layernorm |
|---|---:|---:|
| OpenPI bf16 vs fp32 | 0.01408 | 0.02296 |
| this route bf16 vs fp32 | 0.01222 | 0.01978 |
| this route vs OpenPI bf16 | 0.01606 | 0.02595 |

Neither side is wrong and this route is the closer of the two to exact. The
distance between them exceeds either one's distance from fp32, which is what two
largely independent rounding paths give. So **a more precise kernel moves this
number the wrong way**: the gate measures distance to a target that is itself
1.4% from exact, and rounding less converges to exact, not to upstream. The
hand-written kernels the optimization loop will write should be judged on latency
and on the in-engine reference, not on closing this.

The post_layernorm amplification applies to both routes (0.01408 to 0.02296, and
0.01222 to 0.01978), so it is a property of the tensor, not of this
implementation.

One consequence for how the registry's layer-0 pair reads here. Its calibration
assumes layer 0 carries no accumulated error -- `eval.pi05.reference` says as
much, that a structural error shows up there at full size "because nothing has
accumulated yet". That holds for Pi0, whose prefix is the vision tower's output
read directly. It does not hold for Pi0.5's *backbone* layer 0, which is 27
vision layers deep, so the tight pair is being applied to a tensor that already
carries most of the drift the loose pair was written for. This note records the
observation; the thresholds are unchanged and nothing here argues for relaxing
them.

## A metric that is not stable on this checkpoint

On `pi05_base`, layer 17's K and V carry Gemma's massive activations -- |v| up
to 15, against a bf16 quantum of ~0.125 at that magnitude -- so a handful of
extreme elements decide the cosine and a 1e-5 perturbation upstream can move one
of them by a whole quantum. Across three builds of this backend that differ only
in vision rounding, layer-17 V measured 0.9948, 0.9883 and 0.9871 while layer 16
moved by 6e-6 and `pi05_belt_cup`'s layer 17 stayed at 0.9992. The shipped form
is the one that is better on every stable metric -- layer 0 and the chunk on both
checkpoints -- and worse only on that one. Read a layer-17 change as a bf16
bucket flip until something rules that out.
