# Pi0.5 / RTX 5090: integrated route and experiment traces

The architecture, model-reference, workload-generalization, replay-bucket and
MXFP8 worktree commits are all ancestors of `main`. The active implementation
is the Target's `shipped` plan with workload-specific geometry selection and
replay-time buckets. For the fastest recorded Pi0.5 route, select the approved
`mxfp8-llm-ffn` quantization recipe explicitly. BF16 remains the numerical
control; the two precisions have different reference math and their latency
gap is a latency/quality trade-off, not a same-recipe kernel gain.

| Workload / recipe | Latest recorded deployed median | Evidence |
|---|---:|---|
| robodojo / BF16 | 29.46 ms | [replay extent](replay-extent/README.md) |
| robodojo / MXFP8 FFN | 21.00 ms | [replay extent](replay-extent/README.md) |
| LIBERO / BF16 | 21.61 ms | [replay extent](replay-extent/README.md) |
| LIBERO / MXFP8 FFN | **16.23 ms** | [replay extent](replay-extent/README.md) |

These are the final B legs of separate ABBA comparisons at `7735c9f`, rounded
to 0.01 ms. Do not compute a controlled BF16-to-MXFP8 speedup from this table.
The paired N and T effects are in the replay result; its start-to-end figures
cross sessions and are approximate. The subsequent `e0e3954` commit changed
numerical checks, and this index does not claim a fresh latency measurement on
that commit. The [MXFP8 run](mxfp8-llm-ffn/README.md) contains the earlier
2,000-episode paired LIBERO quality comparison, not a new evaluation of the
latest replay route.

## Main implementation and ablation evidence

| Worktree line | Main-line outcome | Retained experiment evidence |
|---|---|---|
| `exp/pi05-5090/mxfp8-llm-ffn` | Explicit performance recipe, with BF16 as a separate precision | [candidate decisions and progress](mxfp8-llm-ffn/README.md), including the reverted single-launch bucket trial and raw measurements |
| `workload/generalization` | Geometry-specific routes for LIBERO; robodojo's routes stay at their measured shapes | [G1/G2/total ABBA, screens and progress](workload-generalization/README.md) |
| `replay/buckets` | Bucketed capture and exact-length attentions; per-bucket GEMM tiles | [N/A/T ABBA, rejected alternatives and progress](replay-extent/README.md) |
| `reference/end-to-end`, `refactor/architecture` | Shared model reference, source/measurement boundaries | [model and workload references](../../ARCHITECTURE.md), [workflow](../../.agents/skills/model-optimization/SKILL.md) |

The [original GPT-6 BF16 optimization run](gpt6-run-01/README.md) and its
`progress.svg` remain separately preserved. Each run's JSON files are the raw
measurement record; the SVGs display those records and are not additional
performance evidence. An optimization rejected in a local screen or ABBA stays
in its run's notes and files rather than becoming a default backend route.
