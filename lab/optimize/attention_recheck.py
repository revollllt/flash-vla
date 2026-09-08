"""Independent replay/parity confirmation using the previously recorded binaries."""
import argparse
import ctypes
import importlib.util
from pathlib import Path
import torch
from . import component, fixture, store
from .attention_trace_probe import raw_trace
from .component_probe import sample
from benchmarks.kernel_trace.query import validate_tokens
from benchmarks.latency import _env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timing',action='store_true')
    args=parser.parse_args()
    saved=torch.load(args.run/'attention.fixture.pt',weights_only=False)
    meta=store.read(args.run/'fixture-metadata.json')
    values=fixture.restore(saved['before'],'cuda')
    expected=fixture.restore(saved['after'],'cuda')
    params=dict(zip(meta['params'],values))
    outputs=[]; modules=[]; graphs=[]; manifests=[]
    for name in ('off','traced'):
        manifest=store.read(args.run/name/'loaded.json')
        spec=importlib.util.spec_from_file_location('confirmed_'+name,Path(manifest['source_root'])/component.MODULE)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.build=lambda verbose=False, path=manifest['loaded_library']: Path(path)
        module.library()
        manifests.append(dict(library=module.library()._name,geometry=module.geometry()))
        output=torch.empty_like(params['q']); outputs.append(output); modules.append(module)
        def call(module=module,output=output):
            module.attention(params['q'],params['k'],params['v'],params['scale'],params['mask'],output)
        call(); torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): call()
        graphs.append(graph)
    buffers=[torch.zeros((params['q'].shape[0]//64)*12*256*40,device='cuda',dtype=torch.uint8) for _ in range(2)]
    configure=modules[1].library().enc_attn_trace_config
    configure.argtypes=[ctypes.c_void_p,ctypes.c_int]; configure.restype=ctypes.c_int
    captures=[]
    for i,detail in enumerate((1,2)):
        rc=configure(ctypes.c_void_p(buffers[i].data_ptr()),detail)
        if rc: raise RuntimeError(f'configure failed: {rc}')
        graphs[1].replay(); torch.cuda.synchronize()
        if not torch.equal(outputs[0],outputs[1]): raise RuntimeError('trace parity failed')
        if not torch.equal(outputs[0],expected[meta['params'].index('out')]):
            raise RuntimeError('restored production fixture output differs')
        raw=raw_trace(buffers[i],manifests[1],i,detail,params['k'].shape[0])
        captures.append(dict(ranges=len(raw['ranges']),tokens=validate_tokens(raw)))
        if i==0: first=buffers[0].clone()
    if not torch.equal(first,buffers[0]): raise RuntimeError('replay storage overwritten')
    result=dict(env=_env(),parity='bitwise_pass',fixture='production output reproduced',
                replay_storage='independent',captures=captures,loaded=manifests,samples=[])
    if args.timing:
        for block in range(6):
            for _ in range(10): graphs[0].replay()
            torch.cuda.synchronize()
            result['samples'].append(dict(block=block,samples_ms=sample(graphs[0],100)))
    store.write(args.out,result)


if __name__=='__main__': main()
