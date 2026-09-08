"""Actual two-version loading and A/A-prime diagnostics at both deployed shapes."""
import argparse
from pathlib import Path
import time
import torch

from eval.acceptance import tolerances
from eval.metrics import error_metrics
from benchmarks.latency import _env
from . import component, store


def sample(graph, reps):
    values=[]
    for _ in range(reps):
        begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        begin.record(); graph.replay(); end.record(); end.synchronize()
        values.append(begin.elapsed_time(end))
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cutlass',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    modules=[]; manifests=[]
    for label,flags in (('A',()),('Aprime',()),('B',('-DENC_KDEPTH=3','-DENC_VDEPTH=3'))):
        module,manifest=component.load(args.inputs,args.out/label,cutlass=args.cutlass,flags=flags)
        modules.append(module); manifests.append(manifest)
    component.require_distinct(manifests)
    from flash_vla.hardware.nvidia.h100.gemma_backbone.backends.cuda.enc_attn_reference import attention_reference
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0)
    result=dict(environment=_env(),artifacts=manifests,kind='diagnostic_probe',
                production_qualification=False,clock_policy='unlocked',
                protocol=dict(reps=100,orders=[[0,1,2],[2,1,0]]*4,warmup=5,
                              cache='same Q/K/V buffers, L2-resident; not full production weight rotation'),
                shapes=[])
    store.write(args.out/'result.json',result)
    tol=tolerances('bf16')['shallow']
    for target,rows,keys in (('pi05',7744,968),('pi0',6144,768)):
        q=torch.randn(rows,256,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(keys,256,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
        mask=torch.zeros(keys,device='cuda',dtype=torch.bfloat16)
        outputs=[torch.empty_like(q) for _ in modules]; oracle=torch.empty_like(q)
        attention_reference(q,k,v,1/16,mask,oracle)
        graphs=[]; checks=[]
        for module,output in zip(modules,outputs):
            side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5): module.attention(q,k,v,1/16,mask,output)
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): module.attention(q,k,v,1/16,mask,output)
            graph.replay(); torch.cuda.synchronize()
            metrics=error_metrics(oracle,output)
            checks.append(metrics)
            if metrics['rel_rms']>tol['rel_rms_max'] or metrics['cosine_similarity']<tol['cosine_min']:
                raise RuntimeError(f'{target} component parity failed: {metrics}')
            graphs.append(graph)
        if not torch.equal(outputs[0],outputs[1]): raise RuntimeError('A/A-prime is not bitwise identical')
        shape=dict(target=target,q_shape=list(q.shape),kv_shape=list(k.shape),checks=checks,
                   aa_prime_bitwise=True,legs=[])
        result['shapes'].append(shape)
        for block,order in enumerate(result['protocol']['orders']):
            for index in order:
                for _ in range(5): graphs[index].replay()
                torch.cuda.synchronize()
                shape['legs'].append(dict(block=block,artifact=manifests[index]['id'],variant=index,
                                          samples_ms=sample(graphs[index],100)))
        store.write(args.out/'result.json',result)
    print(args.out/'result.json',flush=True)


if __name__=='__main__': main()
