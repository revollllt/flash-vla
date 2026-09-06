# Agent Note: the Gemma action expert becomes a device component package

Status: implemented

Date: 2026-09-07

## Problem

[Device component packages](2026-09-06-device-component-packages.md) declared
that a model component both H100 Targets share gets one package holding its
kernels, and that existing kernels move there with the promotion that first
shares them, bit-identically. The action-expert CUDA/PDL chain
([decoder PDL chain](2026-09-02-decoder-pdl-chain.md),
[cooperative XFS producer](2026-08-28-cooperative-xfs-pdl.md)) was the first
such move to come up: lane D0 of the
[optimization campaign](../../implemented/performance/2026-09-06-optimization-campaign-plan.md)
set out to run it on Pi0 as well, and lane D1 needs it addressable from
outside the Pi0.5 Target.

Two things stood in the way. The chain is not only CUDA: its FFN input comes
from a cooperative TileLang producer that lives in Pi0.5's TileLang backend,
so a package holding only the `.cu` files would still have to import a Target,
which the dependency rule forbids. And the TileLang JIT conventions that
producer needs -- the two `pass_configs` sets, the decorator that compiles a
builder under a name, the raw-builder registry the autotuner re-wraps from --
existed as two verbatim copies, one in each Target's `kernels/base.py`, so
there was no non-Target place to import them from.

## Decision

1. **`hardware/nvidia/h100/gemma_expert/` owns the expert chain.** Its `cuda`
   backend holds the five action-expert call sites, the attention and FFN task
   loops, their ABI header and their torch reference mirrors; its `tilelang`
   half holds the RMS-factor kernel and the XFS producers the CUDA route
   launches. The package declares the route constraints and the graph contract
   for its own call sites; a Target says which of its backend names mean the
   package and keeps every routing decision, plan and constraint name.
2. **The backbone attention stays with Pi0.5.** `enc_attn` belongs to
   `gemma_backbone` and moves with the promotion that shares it, which is lane
   C's. Pi0.5's `cuda` and `cuda-pdl` backends therefore provide the backbone
   attention they own plus the package's expert wrappers, under unchanged
   names, call sites, constraints and graph contract.
3. **`hardware/<vendor>/tilelang/` is a vendor-level library**, beside
   `hardware/<vendor>/cuda/tile/`: the TileLang JIT conventions, specific to
   neither a model nor a device. Both Targets and the package import it, and
   the duplicate in each Target's `kernels/base.py` is gone.
4. **Each declaring module keeps its own raw-builder registry.** The registry
   is a property of a `KernelSet` instance rather than of the shared module,
   because the two Targets declare kernels under the same names with bodies
   that have diverged -- `tl_rms_factor` carries a PDL-trigger parameter on
   Pi0.5 and not on Pi0 -- and one shared registry would hand the autotuner
   whichever module imported last, silently.
5. **Pi0.5 re-exports what its lab scripts and its own TileLang route still
   name**: the expert's RMS-factor kernel and the three producer wrappers,
   which are not call sites and are reached directly by
   `lab/pi05/{attention_block, ffn_full_chain_pdl, ffn_taskloop, xfs_producer,
   xfs_real_chain}.py`. Those scripts import the moved CUDA modules from the
   package.

The move changes nothing about what runs. Every moved file is byte-identical
except the XFS kernel module, whose module docstring and one import line
follow the move, and the C++ namespaces keep their `pi05` segment so the
preprocessed kernel source is unchanged. Pi0's `kernels/base.py` keeps its own
`tl_rms_factor`: its signature differs from the expert's, and re-pointing it
would change that Target's reference-route JIT key for no gain.

## Alternatives considered

- **Give the package its own copy of the JIT decorator.** Rejected: a third
  verbatim copy of code the documentation rule says not to duplicate.
- **Let the package import Pi0.5's TileLang backend.** Rejected: it is the
  dependency the architecture forbids, and a relative import would evade the
  grep the rule is checked with rather than satisfy it.
- **One shared raw-builder registry.** Rejected: it raises or silently
  shadows the moment both Targets are constructed in one process, which
  `eval.smoke` does on every run.
- **Move `enc_attn` and the AdaRMS kernels too.** Rejected: `enc_attn` is
  `gemma_backbone`'s and lane C's; the AdaRMS kernels are on Pi0.5's TileLang
  route only and are not part of the chain.
- **Rename the C++ namespaces in the same commit.** Rejected: it would change
  the preprocessed source and cost the move its by-construction bit identity,
  for a naming improvement that belongs with the next header change.

## Consequences

- Both H100 Targets can register the expert chain; whether either should is a
  separate decision, priced for Pi0 in
  [the Pi0 expert chain is priced out](../../rejected/architecture/2026-09-07-pi0-expert-cuda-chain.md).
- `ARCHITECTURE.md` carries the vendor-level TileLang library in its
  dependency direction; the grep
  `grep -rn "h100\.pi0\b\|h100\.pi05" src/flash_vla/hardware/nvidia/h100/gemma_expert`
  must stay empty, and so must the same search for relative Target imports.
- A Target's `autotune.rewrap` still reads `RAW_KERNELS`, `FAST_MATH` and
  `NO_WARP_SPEC` from its own `kernels/base.py`; the expert's kernels are in
  the package's registry and are re-wrapped from there.
- Registering the package on a second Target additionally needs the row count
  and the prefix length to stop being one profile's compile-time constants;
  the ABI header names that migration and nothing in this note performs it.

## Verification

- Login node: `python -m eval.smoke` passes on both Targets, all seven
  candidate plans and both route oracles; `python -c "import flash_vla"`;
  every `lab/pi05` module imports; both no-Target greps over the package are
  empty.
- Structural, login node: every `.cu` and `.cuh` file crosses the move as a
  rename with no line changed, and the nvcc compile input of both task loops
  -- kernel source, ABI header, shared tile headers and the CUTLASS version --
  hashes identically before and after. The two trees still build into
  different cache directories because the shipped cache tag also hashes the
  absolute CUTLASS path.
- GPU, job 599785 (ACD1-1), `lab/sbatch/bit_identity.sh` with `BASE_TREE` at
  `a02bc64`: all four Target/plan pairs bit-identical. Every declared stage
  output of both Targets on both plans, and the second forward's output,
  compare `torch.equal` between the tree before this change and the tree
  after, one node, one seed. `python -m eval.correctness --steps 1 --layers 1`
  passed on both Targets and `eval.smoke` passed, in the same job.

## Related notes

- [device component packages](2026-09-06-device-component-packages.md): the
  rule this move is the first instance of.
- [decoder PDL chain](2026-09-02-decoder-pdl-chain.md) and
  [cooperative XFS producer + PDL](2026-08-28-cooperative-xfs-pdl.md): what
  the moved kernels are and which launch contract they hold.
- [the Pi0 expert chain is priced out](../../rejected/architecture/2026-09-07-pi0-expert-cuda-chain.md):
  the measurement that decided the package's first cross-Target use.
