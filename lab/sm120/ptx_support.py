#!/usr/bin/env python3
"""Decide, per PTX instruction, which of sm_90a / sm_120 / sm_120a accept it.

Each case is a minimal .entry whose body is the instruction under test. ptxas is
the oracle: a case is SUPPORTED when ptxas returns 0, UNSUPPORTED when it says
"not supported on .target", and BROKEN otherwise -- a BROKEN case is a bug in
this file's PTX, not a fact about the machine, and must be fixed before its row
is read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Machine paths stay out of the repository: point CUDA_HOME at the toolkit, or
# put a matching ptxas on PATH. ptxas must be new enough to know sm_120a.
PTXAS = (
    os.path.join(os.environ["CUDA_HOME"], "bin", "ptxas")
    if os.environ.get("CUDA_HOME")
    else shutil.which("ptxas") or "ptxas"
)
TARGETS = ("sm_90a", "sm_120", "sm_120a")
VERSION = "9.1"

# (id, family, declarations, body). Declarations go before the body inside the
# entry; the harness supplies the entry, params and `ret`.
CASES: list[tuple[str, str, str, str]] = [
    # ---------------- TMA: tensor bulk copy ----------------
    ("tma.load.2d", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    """, """
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
      [%rd1], [%rd2, {%r1, %r2}], [%rd3];
    """),
    ("tma.load.3d", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    """, """
    cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes
      [%rd1], [%rd2, {%r1, %r2, %r3}], [%rd3];
    """),
    ("tma.load.2d.multicast", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<4>; .reg .b16 %h<2>;
    """, """
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
      [%rd1], [%rd2, {%r1, %r2}], [%rd3], %h1;
    """),
    ("tma.store.2d", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    """, """
    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd1, {%r1, %r2}], [%rd2];
    """),
    ("tma.store.3d", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    """, """
    cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%rd1, {%r1, %r2, %r3}], [%rd2];
    """),
    ("tma.prefetch.2d", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    """, """
    cp.async.bulk.prefetch.tensor.2d.L2.global [%rd1, {%r1, %r2}];
    """),
    ("tma.tensormap.prefetch", "tma", """
    .reg .b64 %rd<2>;
    """, """
    prefetch.tensormap [%rd1];
    """),
    ("tma.bulk.load.raw", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<2>;
    """, """
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%rd1], [%rd2], %r1, [%rd3];
    """),
    ("tma.bulk.store.raw", "tma", """
    .reg .b64 %rd<4>; .reg .b32 %r<2>;
    """, """
    cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%rd2], %r1;
    """),
    ("tma.commit_group", "tma", "", "cp.async.bulk.commit_group;"),
    ("tma.wait_group", "tma", "", "cp.async.bulk.wait_group 0;"),
    ("tma.wait_group.read", "tma", "", "cp.async.bulk.wait_group.read 0;"),

    # ---------------- cp.async (Ampere-style) ----------------
    ("cpasync.cg.16", "cpasync", """
    .reg .b64 %rd<4>;
    """, """
    cp.async.cg.shared.global [%rd1], [%rd2], 16;
    """),
    ("cpasync.commit_group", "cpasync", "", "cp.async.commit_group;"),
    ("cpasync.wait_group", "cpasync", "", "cp.async.wait_group 0;"),

    # ---------------- mbarrier ----------------
    ("mbarrier.init", "mbarrier", """
    .reg .b64 %rd<2>; .reg .b32 %r<2>;
    """, """
    mbarrier.init.shared::cta.b64 [%rd1], %r1;
    """),
    ("mbarrier.arrive.expect_tx", "mbarrier", """
    .reg .b64 %rd<3>; .reg .b32 %r<2>;
    """, """
    mbarrier.arrive.expect_tx.shared::cta.b64 %rd2, [%rd1], %r1;
    """),
    ("mbarrier.try_wait.parity", "mbarrier", """
    .reg .b64 %rd<2>; .reg .b32 %r<2>; .reg .pred %p<2>;
    """, """
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%rd1], %r1;
    """),
    ("mbarrier.expect_tx.cluster", "mbarrier", """
    .reg .b64 %rd<3>; .reg .b32 %r<2>;
    """, """
    mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%rd1], %r1;
    """),

    # ---------------- wgmma (Hopper warpgroup MMA) ----------------
    ("wgmma.fence", "wgmma", "", "wgmma.fence.sync.aligned;"),
    ("wgmma.commit_group", "wgmma", "", "wgmma.commit_group.sync.aligned;"),
    ("wgmma.wait_group", "wgmma", "", "wgmma.wait_group.sync.aligned 0;"),
    ("wgmma.mma_async.bf16.n64", "wgmma", """
    .reg .b64 %rd<3>; .reg .f32 %f<33>; .reg .pred %p<2>;
    """, """
    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
      {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11,%f12,%f13,%f14,%f15,%f16,
       %f17,%f18,%f19,%f20,%f21,%f22,%f23,%f24,%f25,%f26,%f27,%f28,%f29,%f30,%f31,%f32},
      %rd1, %rd2, %p1, 1, 1, 0, 0;
    """),
    ("wgmma.mma_async.fp8.n64", "wgmma", """
    .reg .b64 %rd<3>; .reg .f32 %f<33>; .reg .pred %p<2>;
    """, """
    wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3
      {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11,%f12,%f13,%f14,%f15,%f16,
       %f17,%f18,%f19,%f20,%f21,%f22,%f23,%f24,%f25,%f26,%f27,%f28,%f29,%f30,%f31,%f32},
      %rd1, %rd2, %p1, 1, 1;
    """),

    # ---------------- mma.sync (warp-level) ----------------
    ("mma.m16n8k16.bf16", "mma", """
    .reg .b32 %r<9>; .reg .f32 %f<5>;
    """, """
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
      {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};
    """),
    ("mma.m16n8k16.f16", "mma", """
    .reg .b32 %r<9>; .reg .f32 %f<5>;
    """, """
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
      {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};
    """),
    ("mma.m16n8k8.tf32", "mma", """
    .reg .b32 %r<9>; .reg .f32 %f<5>;
    """, """
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
      {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};
    """),
    ("mma.m16n8k32.fp8", "mma", """
    .reg .b32 %r<9>; .reg .f32 %f<5>;
    """, """
    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
      {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};
    """),
    ("mma.m16n8k32.blockscale.mxf8f6f4", "mma", """
    .reg .b32 %r<12>; .reg .f32 %f<5>; .reg .b16 %h<4>;
    """, """
    mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
      {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4},
      %r7, {0, 0}, %r8, {0, 0};
    """),
    ("mma.m16n8k64.blockscale.mxf4", "mma", """
    .reg .b32 %r<12>; .reg .f32 %f<5>; .reg .b16 %h<4>;
    """, """
    mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
      {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4},
      %r7, {0, 0}, %r8, {0, 0};
    """),

    # ---------------- ldmatrix / stmatrix ----------------
    ("ldmatrix.x4.b16", "ldst_matrix", """
    .reg .b64 %rd<2>; .reg .b32 %r<5>;
    """, """
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1,%r2,%r3,%r4}, [%rd1];
    """),
    ("ldmatrix.x4.trans.b16", "ldst_matrix", """
    .reg .b64 %rd<2>; .reg .b32 %r<5>;
    """, """
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r1,%r2,%r3,%r4}, [%rd1];
    """),
    ("stmatrix.x4.b16", "ldst_matrix", """
    .reg .b64 %rd<2>; .reg .b32 %r<5>;
    """, """
    stmatrix.sync.aligned.m8n8.x4.shared.b16 [%rd1], {%r1,%r2,%r3,%r4};
    """),
    ("ldmatrix.m16n16.trans.b8", "ldst_matrix", """
    .reg .b64 %rd<2>; .reg .b32 %r<5>;
    """, """
    ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%r1,%r2,%r3,%r4}, [%rd1];
    """),

    # ---------------- cluster / DSMEM ----------------
    ("cluster.mapa", "cluster", """
    .reg .b32 %r<4>;
    """, """
    mapa.shared::cluster.u32 %r1, %r2, %r3;
    """),
    ("cluster.barrier.arrive", "cluster", "", "barrier.cluster.arrive;"),
    ("cluster.barrier.wait", "cluster", "", "barrier.cluster.wait;"),
    ("cluster.ctarank", "cluster", """
    .reg .b32 %r<2>;
    """, """
    mov.u32 %r1, %cluster_ctarank;
    """),
    ("cluster.nctarank", "cluster", """
    .reg .b32 %r<2>;
    """, """
    mov.u32 %r1, %cluster_nctarank;
    """),

    # ---------------- warp specialization ----------------
    ("setmaxnreg.inc", "warpspec", "", "setmaxnreg.inc.sync.aligned.u32 240;"),
    ("setmaxnreg.dec", "warpspec", "", "setmaxnreg.dec.sync.aligned.u32 24;"),
    ("elect.sync", "warpspec", """
    .reg .b32 %r<2>; .reg .pred %p<2>;
    """, """
    elect.sync %r1|%p1, 0xffffffff;
    """),

    # ---------------- tcgen05 (Blackwell datacenter tensor core) ----------------
    ("tcgen05.alloc", "tcgen05", """
    .reg .b64 %rd<2>; .reg .b32 %r<2>;
    """, """
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd1], %r1;
    """),
    ("tcgen05.ld", "tcgen05", """
    .reg .b32 %r<5>;
    """, """
    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r1}, [%r2];
    """),

    # ---------------- misc execution ----------------
    ("pdl.griddepcontrol.wait", "misc", "", "griddepcontrol.wait;"),
    ("pdl.griddepcontrol.launch", "misc", "", "griddepcontrol.launch_dependents;"),
    ("red.global.add.v4.f32", "misc", """
    .reg .b64 %rd<2>; .reg .f32 %f<5>;
    """, """
    red.global.add.v4.f32 [%rd1], {%f1,%f2,%f3,%f4};
    """),
    ("red.release.gpu.add.u32", "misc", """
    .reg .b64 %rd<2>; .reg .b32 %r<2>;
    """, """
    red.release.gpu.global.add.u32 [%rd1], %r1;
    """),
    ("fence.proxy.async.shared", "misc", "", "fence.proxy.async.shared::cta;"),
    ("fence.proxy.async.global", "misc", "", "fence.proxy.async.global;"),
    ("shfl.sync.bfly", "misc", """
    .reg .b32 %r<5>; .reg .pred %p<2>;
    """, """
    shfl.sync.bfly.b32 %r1|%p1, %r2, 16, 31, 0xffffffff;
    """),
]

NOT_SUPPORTED = re.compile(r"not supported on \.target|requires \.target|Feature .* not supported")


def build(case_decls: str, body: str, target: str) -> str:
    return (
        f".version {VERSION}\n.target {target}\n.address_size 64\n\n"
        ".visible .entry probe(.param .u64 p0)\n{\n"
        f"{case_decls}\n{body}\n  ret;\n}}\n"
    )


def run_case(decls: str, body: str, target: str, tmp: Path) -> tuple[str, str]:
    src = tmp / f"{target}.ptx"
    src.write_text(build(decls, body, target))
    proc = subprocess.run(
        [PTXAS, f"-arch={target}", str(src), "-o", "/dev/null"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return "yes", ""
    err = (proc.stderr or "").strip()
    if NOT_SUPPORTED.search(err):
        return "no", err.splitlines()[0][:160]
    return "BROKEN", err.splitlines()[0][:160] if err else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--family")
    args = ap.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cid, family, decls, body in CASES:
            if args.family and family != args.family:
                continue
            row = {"id": cid, "family": family}
            for target in TARGETS:
                verdict, msg = run_case(decls, body, target, tmp)
                row[target] = verdict
                if verdict == "BROKEN":
                    row.setdefault("errors", {})[target] = msg
            rows.append(row)

    width = max(len(r["id"]) for r in rows)
    print(f"{'instruction':<{width}}  {'sm_90a':>8} {'sm_120':>8} {'sm_120a':>8}")
    print("-" * (width + 28))
    last = None
    for r in rows:
        if r["family"] != last:
            print(f"[{r['family']}]")
            last = r["family"]
        print(f"{r['id']:<{width}}  {r['sm_90a']:>8} {r['sm_120']:>8} {r['sm_120a']:>8}")
        for tgt, msg in (r.get("errors") or {}).items():
            print(f"    BROKEN {tgt}: {msg}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    broken = sum(1 for r in rows if "errors" in r)
    print(f"\n{len(rows)} cases, {broken} BROKEN (fix these before reading the table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
