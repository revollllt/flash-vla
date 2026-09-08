"""Trace existing TMA/WGMMA attention on a real captured-region fixture."""
import argparse
import ctypes
from pathlib import Path
import struct
import torch

from benchmarks.kernel_trace.export_perfetto import convert
from benchmarks.kernel_trace.query import validate_tokens
from benchmarks.latency import _env
from . import component, fixture, store
from .component_probe import sample


def raw_trace(data,manifest,replay,detail,keys):
    names=('producer_empty_wait','tma_issue','k_ready_wait','score_gemm_scope',
           'v_ready_wait','pv_issue','wgmma_drain_wait','role_lifetime')
    semantics=('wait_scope','issue_scope','wait_scope','software_scope',
               'wait_scope','issue_scope','wait_scope','software_scope')
    raw=dict(schema_version=1,synthetic=False,clock_domain='gpu-local',timer_unit='ns',
             dropped_records=0,timer_resolution_ns=None,coverage='all CTAs role leaders' if detail==1 else 'CTA 0 role leaders',
             capture=dict(artifact=manifest,detail=detail,replay_id=replay),stage_dictionary=dict(enumerate(names)),
             ranges=[],markers=[])
    for index,record in enumerate(struct.iter_unpack('<QQIIIIII',data.cpu().numpy().tobytes())):
        begin,end,sm0,sm1,cta,warp,stage,valid=record
        if not valid: continue
        iteration=(index%256)//8 if stage!=7 else 0
        role='math' if warp==0 else 'producer_k' if warp==4 else 'producer_v'
        event=dict(device_id='cuda:0',launch_id=1,replay_id=replay,cta=[cta,0,0],warp_id=warp,
                   role=role,stage=names[stage],semantic=semantics[stage],token=iteration,
                   task_id=cta,iteration=iteration,sm_begin=sm0,sm_end=sm1,start_ns=begin,end_ns=end)
        raw['ranges'].append(event)
        if stage==1 or stage in (2,4):
            operation='tma_k' if warp==4 or stage==2 else 'tma_v'
            depth=4 if operation=='tma_k' else 2
            raw['markers'].append(dict(device_id='cuda:0',launch_id=1,replay_id=replay,cta=[cta,0,0],
                    operation=operation,token=iteration%depth,generation=iteration//depth,role=role,
                    kind='issue' if stage==1 else 'completion_observed',timestamp_ns=begin if stage==1 else end))
    # This kernel commits one score and one P.V group per iteration. The
    # score gemm's existing wait<0> also retires the previous P.V group.
    # Issue timestamps bound the call, not the exact commit instruction.
    ranges={(r['stage'],r['iteration']):r for r in raw['ranges'] if r['role']=='math'}
    for (stage,iteration),event in ranges.items():
        if stage not in ('score_gemm_scope','pv_issue'):
            continue
        completion=event if stage=='score_gemm_scope' else ranges.get(
            ('score_gemm_scope',iteration+1),ranges.get(('wgmma_drain_wait', (keys+63)//64)))
        if completion is None:
            raise ValueError('missing existing wait that retires WGMMA group')
        common=dict(device_id='cuda:0',launch_id=1,replay_id=replay,cta=event['cta'],
                    operation='wgmma_score' if stage=='score_gemm_scope' else 'wgmma_pv',
                    token=0,generation=iteration,role='math',
                    group_sequence=2*iteration+int(stage=='pv_issue'),
                    observation='before issue call through existing wait; not commit-to-completion active time')
        raw['markers'].extend([dict(common,kind='issue',timestamp_ns=event['start_ns']),
                               dict(common,kind='completion_observed',timestamp_ns=completion['end_ns'])])
    key_blocks=(keys+63)//64
    expected=3*(7744//64) if detail==1 else 3+2*key_blocks+max(0,key_blocks-4)+max(0,key_blocks-2)+4*key_blocks+1
    if len(raw['ranges'])!=expected:
        raise RuntimeError(f'expected {expected} records, got {len(raw["ranges"])}')
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--traced-inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--cutlass',type=Path,required=True)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    off,off_manifest=component.load(args.inputs,args.out/'off',cutlass=args.cutlass)
    traced,trace_manifest=component.load(args.traced_inputs,args.out/'traced',cutlass=args.cutlass,
                  flags=(f'-I{args.traced_inputs}/src',f'-I{args.traced_inputs}/lab/examples/kernel_trace'))
    configure=traced.library().enc_attn_trace_config
    configure.argtypes=[ctypes.c_void_p,ctypes.c_int]; configure.restype=ctypes.c_int
    from benchmarks.targets import build
    engine=build('h100/pi05','shipped',steps=1,layers=2)
    inputs=engine.sample_inputs(0); engine.forward(**inputs); torch.cuda.synchronize()
    saved=fixture.capture(engine,'llm_backbone','llm_backbone_attention')
    torch.save(saved,args.out/'attention.fixture.pt')
    before=fixture.restore(saved['before'],'cuda')
    # Use the component ABI directly on the actual first backbone attention inputs.
    op=engine.graph.vocabulary['llm_backbone_attention']
    params=dict(zip(op.params,before))
    store.write(args.out/'fixture-metadata.json',dict(params=list(op.params),identity=engine.identity.as_dict(),
                      values=saved['before']['values'],usage=saved['before']['usage']))
    q,k,v=params['q'],params['k'],params['v']
    mask=params['mask']; scale=params['scale']
    if q.shape[0]!=7744 or k.shape[0]!=968:
        raise ValueError(f'unsupported tracing fixture geometry: Q={q.shape}, K={k.shape}')
    outputs=[torch.empty_like(q),torch.empty_like(q)]
    off.attention(q,k,v,scale,mask,outputs[0]); torch.cuda.synchronize()
    nbytes=(q.shape[0]//64)*12*256*40
    buffers=[torch.zeros(nbytes,device='cuda',dtype=torch.uint8) for _ in range(2)]
    graphs=[]
    for module,output in zip((off,traced),outputs):
        side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5): module.attention(q,k,v,scale,mask,output)
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): module.attention(q,k,v,scale,mask,output)
        graphs.append(graph)
    result=dict(env=_env(),off=off_manifest,traced=trace_manifest,diagnostic_only=True,
                source='actual first backbone attention invocation, isolated replay',captures=[],samples=[])
    first=None
    for replay,(detail,buffer) in enumerate(zip((1,2),buffers)):
        rc=configure(ctypes.c_void_p(buffer.data_ptr()),detail)
        if rc: raise RuntimeError(f'trace configure failed: {rc}')
        graphs[1].replay(); torch.cuda.synchronize()
        if not torch.equal(outputs[0],outputs[1]): raise RuntimeError('trace-on/off bitwise mismatch')
        raw=raw_trace(buffer,trace_manifest,replay,detail,k.shape[0])
        trace,summary=convert(raw); validation=validate_tokens(raw)
        for suffix,value in (('raw',raw),('perfetto',trace),('summary',summary),('validation',validation)):
            store.write(args.out/f'replay{replay}.{suffix}.json',value)
        result['captures'].append(dict(replay=replay,detail=detail,ranges=len(raw['ranges']),tokens=validation))
        if replay==0: first=buffers[0].clone()
    if not torch.equal(first,buffers[0]): raise RuntimeError('later replay overwrote prior trace buffer')
    result['parity']='bitwise_pass'; result['replay_storage']='independent'
    for block,order in enumerate(((0,1,2),(2,1,0),(1,0,2),(2,0,1),(0,2,1),(1,2,0))):
        for mode in order:
            if mode:
                rc=configure(ctypes.c_void_p(buffers[1].data_ptr()),mode)
                if rc: raise RuntimeError(f'trace configure failed: {rc}')
            graph=graphs[0] if mode==0 else graphs[1]
            for _ in range(5): graph.replay()
            torch.cuda.synchronize()
            result['samples'].append(dict(block=block,mode=mode,samples_ms=sample(graph,100)))
    store.write(args.out/'result.json',result)


if __name__=='__main__': main()
