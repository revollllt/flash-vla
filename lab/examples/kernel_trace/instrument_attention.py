"""Create a diagnostic copy of the existing attention source at original boundaries."""
from pathlib import Path
import shutil
import uuid
import json

KERNEL='src/flash_vla/hardware/nvidia/h100/gemma_backbone/backends/cuda/kernels/enc_attn.cu'


def instrument(source,destination):
    source,destination=Path(source),Path(destination)
    shutil.copytree(source,destination)
    path=destination/KERNEL
    code=path.read_text()
    edits=[
        ('namespace enc_attn {','#include "attention_trace.cuh"\n\nnamespace enc_attn {'),
        ('  for (int g = 0; g < key_blocks; ++g) {','  TRACE_BEGIN(producer_life, lane == 0, 7);\n  for (int g = 0; g < key_blocks; ++g) {'),
        ('      while (!tile::mbarrier_try_wait_parity(bars + BarEmpty + s, ph)) {}',
         '      TRACE_BEGIN(empty_wait, lane == 0, 0);\n      while (!tile::mbarrier_try_wait_parity(bars + BarEmpty + s, ph)) {}\n      TRACE_END(empty_wait, lane == 0, g, 0);'),
        ('      reinterpret_cast<FullBar*>(bars + BarFull + s)->arrive_and_expect_tx(KV_FRAME_B);',
         '      TRACE_BEGIN(issue, true, 1);\n      reinterpret_cast<FullBar*>(bars + BarFull + s)->arrive_and_expect_tx(KV_FRAME_B);'),
        ('                        bars + BarFull + s);','                        bars + BarFull + s);\n      TRACE_END(issue, true, g, 1);'),
        ('}\n\n// --------------------------------------------------------------- math',
         '  TRACE_END(producer_life, lane == 0, 0, 7);\n}\n\n// --------------------------------------------------------------- math'),
        ('  MmaS mma_s;','  TRACE_BEGIN(math_life, tid == 0, 7);\n  MmaS mma_s;'),
        ('    while (!tile::mbarrier_try_wait_parity(bars + BAR_FULL_K + ks, (g / KDEPTH) & 1)) {}',
         '    TRACE_BEGIN(k_wait, tid == 0, 2);\n    while (!tile::mbarrier_try_wait_parity(bars + BAR_FULL_K + ks, (g / KDEPTH) & 1)) {}\n    TRACE_END(k_wait, tid == 0, g, 2);'),
        ('    tile::gemm<true, 0, true, true>(mma_s, tSrQ, tSrK, acc_s);',
         '    TRACE_BEGIN(score, tid == 0, 3);\n    tile::gemm<true, 0, true, true>(mma_s, tSrQ, tSrK, acc_s);\n    TRACE_END(score, tid == 0, g, 3);'),
        ('    while (!tile::mbarrier_try_wait_parity(bars + BAR_FULL_V + vs, (g / VDEPTH) & 1)) {}',
         '    TRACE_BEGIN(v_wait, tid == 0, 4);\n    while (!tile::mbarrier_try_wait_parity(bars + BAR_FULL_V + vs, (g / VDEPTH) & 1)) {}\n    TRACE_END(v_wait, tid == 0, g, 4);'),
        ('    tile::gemm<false, -1, true, true>(mma_o, tOrP, tOrV, acc_o);',
         '    TRACE_BEGIN(pv_issue, tid == 0, 5);\n    tile::gemm<false, -1, true, true>(mma_o, tOrP, tOrV, acc_o);\n    TRACE_END(pv_issue, tid == 0, g, 5);'),
        ('  warpgroup_wait<0>();','  TRACE_BEGIN(drain, tid == 0, 6);\n  warpgroup_wait<0>();\n  TRACE_END(drain, tid == 0, p.key_blocks, 6);'),
        ('}\n\n__global__ void __launch_bounds__(THREADS, 1)',
         '  TRACE_END(math_life, tid == 0, 0, 7);\n}\n\n__global__ void __launch_bounds__(THREADS, 1)'),
    ]
    for old,new in edits:
        if code.count(old)!=1:
            raise ValueError(f'instrumentation anchor changed: {old[:80]}')
        code=code.replace(old,new)
    path.write_text(code)
    manifest=json.loads((destination/'source.json').read_text())
    manifest.update(id=uuid.uuid4().hex,root=str(destination.resolve()),
                    variant='diagnostic attention ranges at existing boundaries',parent=manifest['id'])
    (destination/'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return destination
