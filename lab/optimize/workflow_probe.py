"""Fixed CPU comparison of old/new gate control flow; no simulated GPU timings."""
import argparse
import importlib.util
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from eval import gate
from . import store


def run(module, status):
    checks=[dict(check='shallow',mode='gate',status=status)]
    official=[dict(check='official',mode='gate',status='unavailable')]
    with patch.object(module,'_registry_version',return_value='comparison'), \
         patch.object(module,'_run_in_engine_checks',return_value=checks), \
         patch.object(module,'_run_baseline_checks',return_value=official), \
         patch.object(module,'_finish',side_effect=lambda record,out:record), \
         patch.object(module,'_latency_verdict',return_value=dict(run_valid=True,passed=True)), \
         patch.object(module,'_deployment_verdict',return_value=dict(status='passed')), \
         patch.object(module.latency,'run',return_value={}) as latency, \
         patch.object(module.floor_model,'run',return_value={}) as floor:
        start=perf_counter()
        result=module.run('h100/pi05',baseline=True)
        elapsed=perf_counter()-start
    return dict(verdict=result['verdict'],latency_calls=latency.call_count,
                floor_calls=floor.call_count,cpu_dispatch_seconds=elapsed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('original_gate',args.original)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    for block in range(6):
        for name,module in ((('old',old),('new',gate)) if block%2==0 else (('new',gate),('old',old))):
            for status in ('failed','passed'):
                rows.append(dict(block=block,workflow=name,correctness=status,**run(module,status)))
    store.write(args.out,dict(kind='CPU control-flow comparison',samples=rows,
                             original=str(args.original),gpu_time_measured=False,
                             interpretation='counts avoided calls after known failures; does not estimate Agent productivity or kernel speedup',
                             limitations=['fixed mocked stage results','same author and task carryover','small sample','no GPU time imputed']))


if __name__=='__main__':main()
