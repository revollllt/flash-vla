"""T2 kernel parity for the shared Gemma backbone backend, both Targets.

    python -m lab.gemma_backbone.parity --target h100/pi0
    python -m lab.gemma_backbone.parity --target h100/pi05 --kernel attention

Each kernel is run against its ABI mirror
(`gemma_backbone/backends/cuda/*_reference.py`) on the same tensors, in the
same buffers, with the same in-place mutation. This is the structural gate: a
wrong layout, mask, rotation or wiring shows here at full size, before any
in-engine check has a chance to average it away.

Both sides consume the SAME tensors, drawn from the engine's own seeded
buffers at the production shape, so a difference is an implementation
difference and nothing else. Metrics are the shared six of `eval/metrics.py`
in float64; `rel_rms` and `cosine_similarity` gate, at the acceptance
registry's `shallow` pair for the Target's precision policy -- read, never
invented.

An fp32 recomputation is reported beside the gate for attention. It is not a
gate: it bounds how much of the measured difference is bf16 rounding that both
implementations are entitled to.

Padded and masked regions are checked for finiteness separately, because
`0 * NaN` survives a mask and a NaN parked in a padded row eventually reaches
real data.
"""
from __future__ import annotations

import argparse
import json

import torch

from benchmarks.kernels import record_invocations
from benchmarks.targets import build, resolve
from eval.acceptance import tolerances
from eval.metrics import error_metrics
from flash_vla.hardware.nvidia.h100.gemma_backbone.backends.cuda import (
    enc_attn_reference,
    gated_ffn_reference,
    residual_gemm_reference,
)

SEGMENT = "llm_backbone"

#: kernel key -> the call site it implements.
KERNELS = {
    "attention": "llm_backbone_attention",
    "norm_gated_ffn": "llm_backbone_norm_gated_ffn",
    "out_proj_residual": "llm_backbone_out_proj_residual",
    "ffn_down_residual": "llm_backbone_ffn_down_residual",
}


def _plan(kernel: str) -> dict[str, str]:
    return {KERNELS[kernel]: "gemma-cuda"}


def _check_attention(engine, call_args) -> dict:
    Q, K, V, scale, mask, out = call_args
    expected = torch.empty_like(out)
    enc_attn_reference.attention_reference(Q, K, V, scale, mask, expected)
    got = out.clone()
    report = {"metrics": error_metrics(expected, got), "finite": bool(torch.isfinite(got).all())}
    fp32 = enc_attn_reference.attention_fp32_reference(Q, K, V, scale, mask)
    report["vs_fp32"] = error_metrics(fp32.to(torch.bfloat16), got)
    return report


def _check_gated_ffn(call_args) -> dict:
    x, gate_w, up_w, out, x_norm = call_args
    rows = x.shape[0]
    expected = torch.empty_like(out[:rows])
    expected_norm = torch.empty_like(x_norm[:rows])
    gated_ffn_reference.norm_gated_ffn_reference(x, gate_w, up_w, expected, expected_norm)
    got, got_norm = out[:rows].clone(), x_norm[:rows].clone()
    report = {"metrics": error_metrics(expected, got),
              "finite": bool(torch.isfinite(got).all())}
    # The normalized activation is an auxiliary output with no consumer, but the
    # projection must see the bf16-rounded values, so it is checked separately
    # and expected to be bit-identical: both sides round the same way.
    report["x_norm"] = error_metrics(expected_norm, got_norm)
    report["x_norm_bit_identical"] = bool(torch.equal(expected_norm, got_norm))
    return report


def _run_site(engine, call_site: str, invocation) -> dict:
    """Invoke one recorded call, with the residual sites re-run from a clean copy."""
    args, kwargs = invocation
    fn = getattr(engine.ops, call_site)
    if call_site == "llm_backbone_attention":
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        return _check_attention(engine, args)
    if call_site == "llm_backbone_norm_gated_ffn":
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        return _check_gated_ffn(args)

    # In-place residual: snapshot the residual, run the kernel, then run the
    # mirror from the same snapshot into a separate buffer.
    x, weight, out = args
    before = out.clone()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    got = out.clone()
    expected = before.clone()
    residual_gemm_reference.residual_gemm_reference(x, weight, expected)
    return {"metrics": error_metrics(expected, got),
            "finite": bool(torch.isfinite(got).all())}


def run(target: str, kernels: tuple[str, ...], seed: int = 0) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on a GPU node")
    # The mirrors contract in fp32 and refuse to run under TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    target = resolve(target)
    tol = None
    results: dict[str, dict] = {}
    for kernel in kernels:
        call_site = KERNELS[kernel]
        engine = build(target, _plan(kernel), seed=seed)
        if tol is None:
            tol = tolerances(engine.identity.precision)["shallow"]
        engine.forward(**engine.sample_inputs(seed))
        torch.cuda.synchronize()
        calls = record_invocations(engine, SEGMENT)
        if call_site not in calls:
            raise RuntimeError(f"{call_site} was not invoked; recorded {sorted(calls)}")
        # Layer 0's invocation: nothing upstream has accumulated into it.
        report = _run_site(engine, call_site, calls[call_site][0])
        metrics = report["metrics"]
        report["passed"] = bool(metrics["rel_rms"] < tol["rel_rms_max"]
                                and metrics["cosine_similarity"] > tol["cosine_min"]
                                and report["finite"])
        report["invocations"] = len(calls[call_site])
        results[kernel] = report
        line = (f"[{target} {kernel:18s}] rel_rms {metrics['rel_rms']:.3e}  "
                f"cosine {metrics['cosine_similarity']:.7f}  "
                f"max_abs {metrics['max_abs']:.3e}  "
                f"finite {report['finite']}  "
                f"{'PASS' if report['passed'] else 'FAIL'}")
        print(line, flush=True)
        del engine
        torch.cuda.empty_cache()
    return {"target": target, "tolerance": dict(tol or {}), "kernels": results,
            "passed": all(r["passed"] for r in results.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--kernel", action="append", default=None,
                        choices=sorted(KERNELS), help="default: every kernel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the full report here")
    args = parser.parse_args(argv)
    report = run(args.target, tuple(args.kernel or sorted(KERNELS)), seed=args.seed)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
