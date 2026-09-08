"""Bounded QKV tile-width probe with rotating weights and fixed paired blocks."""
import argparse
import importlib.util
import sys
from pathlib import Path
import torch
from benchmarks.latency import _env
from eval.acceptance import tolerances
from eval.metrics import error_metrics
from flash_vla.hardware.nvidia.h100.pi05.pipeline import rope_table
from flash_vla.hardware.nvidia.h100.spec import H100Spec
from flash_vla.tuning import cold_n_inner
from .component_probe import sample
from . import store

SOURCE='src/flash_vla/hardware/nvidia/h100/pi05/backends/tilelang/kernels/base.py'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels.qkv_probe_source',args.inputs/SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    m,k,n=968,2048,2560
    count=cold_n_inner(k*n*2,H100Spec.L2_CACHE_SIZE_BYTES)
    a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    weights=[torch.randn(k,n,device='cuda',dtype=torch.bfloat16)*0.05 for _ in range(count)]
    rope=rope_table(m,0,256,'cuda')
    variants=[];outputs=[];graphs=[]
    result=dict(env=_env(),source=store.read(args.inputs/'source.json'),
                hypothesis='wider N tile reduces repeated A transactions',
                alternative='larger accumulator and epilogue reduce throughput',
                protocol=dict(shape=[m,n,k],weight_rotation=count,weight_bytes=count*k*n*2,
                              warmup=5,reps=100,orders=[[0,1,1,0],[1,0,0,1]]*3),
                scope='Pi0.5 fused projection only; Pi0 has a different implementation',
                qualification=False,artifacts=[],samples=[])
    for bn in (64,128):
        config=dict(BLOCK_M=128,BLOCK_N=bn,BLOCK_K=64,NUM_STAGES=3,THREADS=128)
        kernel=module.tl_matmul_rope_scatter.compile(M=m,N=n,K=k,HEAD_DIM=256,NUM_HEADS=8,**config)
        from tvm import runtime
        library=(args.out/f'qkv_bn{bn}.so').resolve()
        kernel.adapter.executable.export_library(str(library))
        kernel.adapter.executable=runtime.load_module(str(library))
        kernel.torch_function=kernel.adapter._convert_torch_func()
        kernel.export_sources(kernel_path=str(args.out/f'qkv_bn{bn}.cu'))
        out=[torch.empty(m,width,device='cuda',dtype=torch.bfloat16) for width in (2048,256,256)]
        kernel(a,weights[0],rope,*out);torch.cuda.synchronize()
        result['artifacts'].append(dict(config=config,library=str(library),
                                        warp_specialization=False,tiles=((m+127)//128)*(n//bn)))
        outputs.append([x.clone() for x in out]);variants.append(kernel)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                for w in weights:kernel(a,w,rope,*out)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for w in weights:kernel(a,w,rope,*out)
        graphs.append(graph)
    metrics=[error_metrics(x,y) for x,y in zip(*outputs)]
    result['checks']=metrics
    tol=tolerances('bf16')['shallow']
    if any(v['rel_rms']>tol['rel_rms_max'] or v['cosine_similarity']<tol['cosine_min'] for v in metrics):
        store.write(args.out/'result.json',result)
        raise RuntimeError('QKV variant failed existing shallow tolerance')
    result['bitwise']=all(torch.equal(x,y) for x,y in zip(*outputs))
    for block,order in enumerate(result['protocol']['orders']):
        for variant in order:
            for _ in range(5):graphs[variant].replay()
            torch.cuda.synchronize()
            result['samples'].append(dict(block=block,variant=variant,
                                           samples_ms=[v/count for v in sample(graphs[variant],100)]))
    store.write(args.out/'result.json',result)


if __name__=='__main__': main()
