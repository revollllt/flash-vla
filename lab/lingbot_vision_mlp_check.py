"""Check the width-aligned vision feed-forward against the upstream expression.

    python -m lab.lingbot_vision_mlp_check

Random weights at the Target's real vision shapes; no checkpoint is needed.
The padded lanes carry zero weight and zero bias, so any difference is the
accumulation order of a different cuBLAS kernel, not a different value.
"""
import torch

from flash_vla.models.lingbot.spec import PATCH_ROWS_PER_VIEW, VIEWS, VISION_DIM, VISION_FFN


def main() -> int:
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    rows = VIEWS * PATCH_ROWS_PER_VIEW
    x = torch.randn(rows, VISION_DIM, dtype=dtype, device=device)
    gate = torch.nn.Linear(VISION_DIM, VISION_FFN, device=device, dtype=dtype)
    up = torch.nn.Linear(VISION_DIM, VISION_FFN, device=device, dtype=dtype)
    down = torch.nn.Linear(VISION_FFN, VISION_DIM, device=device, dtype=dtype)
    act = torch.nn.SiLU()

    want = down(act(gate(x)) * up(x))

    pad = (-(-VISION_FFN // 8)) * 8 - VISION_FFN
    packed_w = torch.cat([torch.nn.functional.pad(gate.weight, (0, 0, 0, pad)),
                          torch.nn.functional.pad(up.weight, (0, 0, 0, pad))], dim=0)
    packed_b = torch.cat([torch.nn.functional.pad(gate.bias, (0, pad)),
                          torch.nn.functional.pad(up.bias, (0, pad))], dim=0)
    down_w = torch.nn.functional.pad(down.weight, (0, pad))
    gate_up = torch.nn.functional.linear(x, packed_w, packed_b)
    g, u = gate_up.split(VISION_FFN + pad, dim=-1)
    got = torch.nn.functional.linear(act(g) * u, down_w, down.bias)
    torch.cuda.synchronize()

    error = (got.float() - want.float())
    scale = want.float().norm()
    print(f"  padded width {VISION_FFN} -> {VISION_FFN + pad}")
    print(f"  max_abs={error.abs().max().item():.3e} rel_rms={(error.norm()/scale).item():.3e}")
    print(f"  cosine={torch.nn.functional.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0).item():.10f}")
    # One bf16 rounding is 2^-8 relative; a different K blocking of the same sum
    # should stay within a small multiple of that.
    ok = (error.norm() / scale).item() < 1e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
