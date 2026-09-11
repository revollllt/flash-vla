"""A0 -- does TileLang 0.1.11 produce working sm_120 code?

pi0's entire fast path is TileLang and its `reference` plan is TileLang too, so
this is the go/no-go for the chosen Target. Three samples, chosen to separate
the failure modes rather than just to get a pass:

  1. tl_rms_norm      reduction only, no tensor core, no warp specialisation
  2. tl_matmul        tensor core, warp specialisation OFF
  3. tl_matmul_ws     tensor core, warp specialisation ON  <- the setmaxnreg risk

Hypothesis: TileLang emits sm_80-level primitives and plain `sm_120` suffices,
so (1) and (2) pass. Falsifier: (3) fails or miscompares, because warp
specialisation on Hopper uses `setmaxnreg`, which ptxas refuses on plain
`sm_120` and TileLang 0.1.11 has no `sm_120a`/`sm_120f` in its arch table.
"""
import traceback

import torch

from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.kernels import base as kernels


def report(name, fn):
    try:
        err = fn()
    except Exception as exc:  # noqa: BLE001 -- the failure mode IS the result
        head = traceback.format_exc().strip().splitlines()[-1][:150]
        print(f"  {name:22} FAIL   {type(exc).__name__}: {head}")
        return False
    ok = err < 2e-2
    print(f"  {name:22} {'PASS' if ok else 'WRONG'}   max rel err {err:.3e}")
    return ok


def rms_norm():
    M, K = 256, 1152
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    kernels.tl_rms_norm.compile(M=M, K=K, BLOCK_M=1, BLOCK_K=128, THREADS=128)(x, out)
    torch.cuda.synchronize()
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (out.float() - ref).abs().max().item() / ref.abs().max().item()


def _matmul(entry):
    M, N, K = 256, 256, 128
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    entry.compile(M=M, N=N, K=K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                  NUM_STAGES=2, THREADS=256)(a, b, c)
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    return (c.float() - ref).abs().max().item() / ref.abs().max().item()


print(f"torch {torch.__version__}  device {torch.cuda.get_device_properties(0).name}")
import tilelang
print(f"tilelang {tilelang.__version__}\n")

results = {
    "tl_rms_norm": report("tl_rms_norm", rms_norm),
    "tl_matmul (WS off)": report("tl_matmul (WS off)", lambda: _matmul(kernels.tl_matmul)),
    "tl_matmul_ws (WS on)": report("tl_matmul_ws (WS on)", lambda: _matmul(kernels.tl_matmul_ws)),
}
print()
if all(results.values()):
    print("A0 -> GO: TileLang works on sm_120, warp specialisation included")
elif results["tl_rms_norm"] and results["tl_matmul (WS off)"]:
    print("A0 -> PARTIAL: TileLang works with warp specialisation OFF; WS path is broken")
else:
    print("A0 -> NO-GO: TileLang cannot carry pi0 on this device")
