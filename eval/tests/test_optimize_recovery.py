import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lab.optimize import scheduler, store, runner
from eval.tests.test_optimize import spec
from eval.tests import test_optimize


class RecoveryTests(unittest.TestCase):
    setUp = test_optimize.RunnerTests.setUp
    def test_worker_loss_accounted_once(self):
        record=runner.start(self.root,spec(),self.out)
        record['stages']['check']=dict(status='running',owner={'job':'123'},resource='gpu')
        store.write(self.out/'evidence.json',record)
        with patch.object(scheduler,'job_state',return_value='COMPLETED'):
            with self.assertRaises(RuntimeError): runner.reconcile(self.out)
            result=runner.reconcile(self.out,recovered_seconds=17)
        self.assertEqual(result['cost']['gpu_seconds'],17)
        self.assertEqual(store.read(self.root/'index/probe.json')['cost']['gpu_seconds'],17)
        self.assertEqual(runner.reconcile(self.out)['cost']['gpu_seconds'],17)

    def test_duplicate_missing_artifact_and_reopen(self):
        record=runner.start(self.root,spec(),self.out)
        item=store.index(self.root/'index',record)
        self.assertEqual(store.duplicate_status(item,spec(),record['source']),'duplicate_review')
        s=spec(); s['conditions']={'layout':'different'}
        self.assertEqual(store.duplicate_status(item,s,record['source']),'reopen_changed_conditions')
        s=spec(); s['replication_of']='probe'
        self.assertEqual(store.duplicate_status(item,s,record['source']),'deliberate_replication')
        (self.out/'inputs/src/kernel.cu').unlink()
        self.assertEqual(store.duplicate_status(item,spec(),record['source']),'identity_unknown_missing_artifacts')
        self.assertEqual(len(store.related(self.root/'index',spec())),1)

    def test_invalid_does_not_exhaust_non_improving_budget(self):
        s=spec(); s.update(task_id='task',kind='performance')
        record=runner.start(self.root,s,self.out)
        record.update(validity='incomplete',conclusion='no_benefit')
        store.index(self.root/'index',record)
        policy={'candidates':2,'non_improving':1,'jobs':12}
        self.assertEqual(store.task_budget(self.root/'index',s,policy)['non_improving'],0)
        record['validity']='valid'; store.index(self.root/'index',record)
        with self.assertRaises(RuntimeError): store.task_budget(self.root/'index',s,policy)

    def test_external_contention_is_recorded_failure(self):
        s=spec(); s['stages']['check']['resource']='gpu'
        runner.start(self.root,s,self.out)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}), patch.object(scheduler,'allocation',side_effect=RuntimeError('busy')):
            with self.assertRaises(RuntimeError): runner.run(self.out)
        result=store.read(self.out/'evidence.json')
        self.assertEqual(result['cost']['jobs'],['123'])
        self.assertGreater(result['cost']['gpu_seconds'],0)
        self.assertEqual(result['status'],'stopped')

    def test_gpu_uuid_selected_before_process_check(self):
        responses=['RUNNING','GPU-abc, H100, driver','999']
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_JOB_GPUS':'7'}), patch.object(scheduler,'command',side_effect=responses) as command:
            with self.assertRaises(RuntimeError): scheduler.allocation()
        self.assertEqual(command.call_args.args[0][2],'GPU-abc')

    def test_candidate_cap_and_confirmation_reserve(self):
        s=spec();s['kind']='performance'
        record=runner.start(self.root,s,self.out)
        with self.assertRaises(RuntimeError):
            store.task_budget(self.root/'index',s,dict(candidates=1,non_improving=3,jobs=12))
        record['budget']['jobs']=1
        record['spec']['stages']['qualify']={}
        record['spec']['stages']['check']['resource']='gpu'
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}):
            with self.assertRaisesRegex(RuntimeError,'confirmation reserve'):
                runner.run_stage(record,'check')
        self.assertEqual(record['cost']['jobs'],[])

    def test_qualification_rechecks_sources_after_gate(self):
        import json
        import sys
        s=spec();s['stages']['qualify']={}
        record=runner.start(self.root,s,self.out)
        record['spec']['conditions']={'seed':42}
        record['spec']['qualification_sources']=dict(
            incumbent=record['source'],candidate=record['source'],
            incumbent_checkout=str(self.root),candidate_checkout=str(self.root),
            affected_targets=[record['target']])
        command=runner._command(record,'qualify')
        self.assertIn('eval.gate',command)
        self.assertEqual(command[command.index('--seed')+1],'42')
        output=self.out/'qualify';output.mkdir()
        payload=json.dumps(dict(verdict='pass'))
        code=(f'from pathlib import Path; Path({str(output / "report.json")!r}).write_text({payload!r}); '
              f'Path({str(self.root / "src/kernel.cu")!r}).write_text("changed")')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}), \
             patch.object(scheduler,'allocation',return_value={'job':'123'}), \
             patch.object(runner,'_command',return_value=[sys.executable,'-c',code]):
            runner.run_stage(record,'qualify')
        self.assertEqual(record['promotion'],'blocked')
        self.assertFalse(record['promotion_review']['applicable'])

    def test_simultaneous_starts_share_candidate_budget(self):
        from concurrent.futures import ThreadPoolExecutor
        def start(index):
            s=spec();s.update(id=f'trial{index}',task_id='one-task',kind='performance',budget={'candidates':1})
            try:
                runner.start(self.root,s,self.root/s['id'])
                return 'started'
            except RuntimeError as error:
                if str(error) != 'task candidate budget exhausted':
                    raise
                return 'budget_stop'
        with ThreadPoolExecutor(max_workers=2) as pool:
            result=list(pool.map(start,(0,1)))
        self.assertEqual(sorted(result),['budget_stop','started'])
