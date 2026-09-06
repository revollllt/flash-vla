# lab: the optimization workspace

Everything here is a trial, not a deployment configuration. The deployment
path (`src/`, `eval/`, `benchmarks/`) carries exactly one shipped plan and one
reference plan per Target and never imports this directory; scripts here may
import the deployment path.

| path | what |
|---|---|
| `plans/` | candidate call-site plans for A/B runs, passed as `--plan lab/plans/<name>.json` |
| `gemma_backbone/` | experiment and per-kernel checks of the shared Gemma backbone component: `parity.py` (each backbone kernel against its ABI mirror), `baselines.py` (cuBLAS at the production GEMM shapes of both Targets), `ncu_driver.py` (one call site, a fixed number of launches, for a profiler) |
| `pi05/` | experiment and per-kernel checks of the Pi0.5 Target: `kernels.py` (AdaRMSNorm kernels vs torch), `enc_attn.py` (the fused CUDA encoder attention vs the torch chain), `attention_block.py`, `ffn_taskloop.py`, `ffn_full_chain_pdl.py`, `xfs_producer.py`, `xfs_real_chain.py` |
| `sbatch/` | the Slurm wrappers of those scripts; submit from the repository root (`sbatch lab/sbatch/<name>.sh`) |
| `stage_dump.py` | dump a Target's declared stage outputs on one plan, and compare two dumps bit for bit (`python -m lab.stage_dump`); `sbatch/bit_identity.sh` runs it from two trees on one node, the proof a promotion PR gives when a kernel moves without a numerical change |

A candidate that is promoted ships its kernel and its check into the
deployment path in the same change; its trial script stays here as the record
of how it was measured. The `kernel-design` skill's candidate loop works from
this directory.
