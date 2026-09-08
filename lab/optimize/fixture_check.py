"""Explicit GPU alias/mutation check for the correctness fixture adapter."""
import argparse
import torch
from benchmarks.latency import _env
from . import fixture,store


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    base=torch.arange(16,dtype=torch.float32,device='cuda')
    record=fixture.snapshot([base[2:10],base[4:12:2]])
    base.zero_()
    a,b=fixture.restore(record,'cuda')
    if not torch.equal(a,torch.arange(2,10,dtype=torch.float32,device='cuda')):
        raise RuntimeError('snapshot values changed after production storage mutation')
    if b.stride()!=(2,):raise RuntimeError('strided alias changed')
    a[2]=99
    if b[0].item()!=99:raise RuntimeError('overlapping alias was not preserved')
    store.write(args.out,dict(env=_env(),values='pass',stride='pass',overlapping_alias='pass'))


if __name__=='__main__':main()
