"""Correctness-only tensor snapshots, including overlapping views; never a timer."""
from collections import defaultdict
import torch


def snapshot(values):
    groups=defaultdict(list)
    for index,value in enumerate(values):
        if isinstance(value,torch.Tensor):
            groups[(value.device,value.untyped_storage().data_ptr())].append((index,value))
    tensors=[]; items=list(values)
    for group in groups.values():
        alignment=max(t.element_size() for _,t in group)
        begin=min(t.storage_offset()*t.element_size() for _,t in group)
        begin=begin//alignment*alignment
        end=max((t.storage_offset()+sum((n-1)*s for n,s in zip(t.shape,t.stride()))+1)*t.element_size()
                if t.numel() else t.storage_offset()*t.element_size() for _,t in group)
        original=group[0][1]
        raw=torch.empty(0,dtype=torch.uint8,device=original.device).set_(
            original.untyped_storage(),begin,(end-begin,),(1,))
        tensors.append(raw.clone().cpu())
        for index,tensor in group:
            items[index]=dict(storage=len(tensors)-1,dtype=str(tensor.dtype).removeprefix('torch.'),
                              shape=list(tensor.shape),stride=list(tensor.stride()),
                              offset_bytes=tensor.storage_offset()*tensor.element_size()-begin,
                              original_offset_bytes=tensor.storage_offset()*tensor.element_size())
    return dict(storages=tensors,values=items,usage='correctness_only; copies excluded from timing')


def restore(record,device='cpu'):
    storages=[value.to(device) for value in record['storages']]
    values=[]
    for value in record['values']:
        if isinstance(value,dict) and 'storage' in value:
            dtype=getattr(torch,value['dtype'])
            tensor=torch.empty(0,dtype=dtype,device=device)
            tensor.set_(storages[value['storage']].untyped_storage(),
                        value['offset_bytes']//tensor.element_size(),value['shape'],value['stride'])
            values.append(tensor)
        else:
            values.append(value)
    return values


def capture(engine, segment, site):
    """Snapshot the first invocation in a real eager stage; replay timing uses original buffers."""
    result={}
    def wrap(name,fn):
        def call(*args,**kwargs):
            if name!=site or result:
                return fn(*args,**kwargs)
            if kwargs:
                raise ValueError('fixture adapter currently supports positional op arguments')
            result['before']=snapshot(args)
            output=fn(*args,**kwargs)
            torch.cuda.synchronize()
            result['after']=snapshot(args)
            result.update(site=site,segment=segment,identity=engine.identity.as_dict(),
                          cache_context='eager production-shaped stage; not a performance fixture')
            return output
        return call
    with engine.instrument(wrap):
        engine.run_eager(segment)
    if not result:
        raise ValueError(f'no invocation of {site} in {segment}')
    return result
