import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lab.optimize import scheduler, store, runner
from tests.legacy.test_optimize import spec
from tests.legacy import test_optimize


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
        self.assertIn('lab.optimize.gate',command)
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


    def test_qualification_preserves_asset_options_through_gate_cli(self):
        import json
        import subprocess
        import sys

        s = spec()
        s['stages']['qualify'] = {}
        options = dict(
            checkpoint="/assets/checkpoint A/model.safetensors",
            checkpoint_id="trained-a", checkpoint_digest="publisher-revision-a",
            openpi_config="pi05_droid", tokenizer_path="/assets/tokenizer model",
            prompt="pick = cup; $(not-a-command)\nthen place it",
        )
        s['conditions'] = dict(seed=42)
        s['options'] = options
        from types import SimpleNamespace
        from lab.optimize import preflight
        from flash_vla.models.pi05 import openpi as openpi05
        with patch.object(openpi05, 'resolve_config',
                          return_value=SimpleNamespace(action_horizon=15, max_token_len=200)), \
             patch.object(preflight, 'declare', wraps=preflight.declare) as declare:
            prepared = preflight.inspect(s)
        self.assertEqual(prepared['workload']['shape']['chunk'], 15)
        self.assertEqual([call.kwargs['seed'] for call in declare.call_args_list], [42, 42])
        record = runner.start(self.root, s, self.out)
        record['spec']['qualification_sources'] = dict(
            incumbent=record['source'], candidate=record['source'],
            incumbent_checkout=str(self.root), candidate_checkout=str(self.root),
            affected_targets=[record['target']])
        store.write(self.out / 'evidence.json', record)
        command = runner._command(store.read(self.out / 'evidence.json'), 'qualify')
        # Exercise the real gate CLI parser in a fresh interpreter, replacing
        # only the GPU work with a recorder.
        script = (
            "import json, sys; from lab.optimize import gate; "
            "gate.run = lambda *args, **kwargs: dict(verdict='pass', args=args, kwargs=kwargs); "
            "gate.summary = json.dumps; raise SystemExit(gate.main(sys.argv[1:]))"
        )
        result = subprocess.run(
            [sys.executable, '-c', script, *command[3:]],
            cwd=Path(__file__).resolve().parents[2],
            check=True, text=True, capture_output=True)
        received = json.loads(result.stdout)
        self.assertEqual(received['args'][0], s['target'])
        self.assertEqual(received['kwargs']['seed'], 42)
        self.assertTrue(received['kwargs']['baseline'])
        self.assertEqual({key: received['kwargs'].get(key) for key in options}, options)
        item = store.read(self.root / 'index' / 'probe.json')
        self.assertEqual(item['conditions'], s['conditions'])
        self.assertEqual(item.get('options'), options)
        changed = dict(s, options=dict(options, checkpoint_id='trained-b'))
        self.assertEqual(store.duplicate_status(item, changed, record['source']),
                         'reopen_changed_conditions')

    def test_resume_does_not_drop_obsolete_asset_selection(self):
        record = runner.start(self.root, spec(), self.out)
        record['spec']['conditions'] = {'options': {'checkpoint': '/assets/trained-a'}}
        store.write(self.out / 'evidence.json', record)
        saved = (self.out / 'evidence.json').read_text()
        with patch.object(runner, 'run_stage') as launch:
            with self.assertRaisesRegex(ValueError, 'move conditions.options to spec.options'):
                runner.run(self.out)
        launch.assert_not_called()
        self.assertEqual((self.out / 'evidence.json').read_text(), saved)

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
