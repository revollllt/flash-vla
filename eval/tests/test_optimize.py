import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lab.optimize import context, runner, scheduler, store
from lab.optimize.schema import validate


def spec():
    return dict(version=1, id='probe', target='hardware/nvidia/h100/pi05', kind='diagnostic_probe',
                hypothesis=dict(mechanism='input wait', alternative='scheduling',
                                falsifier='unchanged wait', cheapest_probe='one trace'),
                scope=['src'], inputs=['src/kernel.cu'], changes=['src/kernel.cu'],
                stages={'check': dict(argv=['{python}', '-c', 'print("parity")'], resource='cpu', timeout_s=10),
                        'measure': dict(argv=['{python}', '-c', 'print("observed")'], resource='cpu', timeout_s=10)})


class SchemaTests(unittest.TestCase):
    def test_reject_missing_version_scope_and_budget(self):
        for mutate in (lambda s: s.pop('hypothesis'), lambda s: s.update(version=2),
                       lambda s: s.update(changes=['eval/acceptance.py']),
                       lambda s: s.update(budget={'jobs': 100}),
                       lambda s: s.update(inputs=['../escape'])):
            s = spec(); mutate(s)
            with self.assertRaises(ValueError): validate(s)

    def test_no_correctness_no_measurement(self):
        s = spec(); del s['stages']['check']
        with self.assertRaises(ValueError): validate(s)

    def test_obsolete_asset_options_require_explicit_migration(self):
        s = spec()
        s['conditions'] = {'options': {'checkpoint': '/assets/trained-a'}}
        with self.assertRaisesRegex(ValueError, 'move conditions.options to spec.options'):
            validate(s)

    def test_readonly_policy(self):
        policy, _ = validate(spec())
        policy['latency']['promotion_bar_ms'] = 0
        self.assertEqual(validate(spec())[0]['latency']['promotion_bar_ms'], 0.10)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                        'commit', '--allow-empty', '-qm', 'test'], check=True)
        (self.root/'src').mkdir()
        (self.root/'src/kernel.cu').write_text('original')
        self.out = self.root/'run'

    def test_success_resume_and_idempotent_index(self):
        runner.start(self.root, spec(), self.out)
        result = runner.run(self.out)
        elapsed = result['cost']['cpu_seconds']
        again = runner.run(self.out)
        self.assertEqual(again['cost']['cpu_seconds'], elapsed)
        self.assertEqual(again['status'], 'completed')
        self.assertEqual(again['correctness'], 'pass')
        self.assertEqual(again['conclusion'], 'inconclusive')
        self.assertEqual(len(list((self.root/'index').glob('*.json'))), 1)

    def test_failure_stops_before_measure(self):
        s=spec(); s['stages']['check']['argv']=['{python}', '-c', 'raise SystemExit(3)']
        runner.start(self.root,s,self.out)
        result=runner.run(self.out)
        self.assertEqual(result['status'],'stopped')
        self.assertNotIn('measure',result['stages'])
        with self.assertRaises(RuntimeError): runner.run(self.out)

    def test_timeout_accounted_and_reconciled(self):
        s=spec(); s['stages']['check'].update(argv=['{python}', '-c', 'import time; time.sleep(10)'], timeout_s=1)
        runner.start(self.root,s,self.out)
        with self.assertRaises(subprocess.TimeoutExpired): runner.run(self.out)
        before=store.read(self.out/'evidence.json')['cost']['cpu_seconds']
        self.assertGreaterEqual(before,1)
        result=runner.reconcile(self.out)
        self.assertEqual(result['cost']['cpu_seconds'],before)
        self.assertEqual(len(result['attempts']),1)

    def test_only_build_inputs_affect_comparison(self):
        a=store.capture(self.root,['src/kernel.cu'],self.root/'a')
        (self.root/'notes.md').write_text('new note')
        b=store.capture(self.root,['src/kernel.cu'],self.root/'b')
        self.assertEqual(store.changed_inputs(a,b),[])
        (self.root/'src/kernel.cu').write_text('changed')
        c=store.capture(self.root,['src/kernel.cu'],self.root/'c')
        self.assertEqual(store.changed_inputs(b,c),['src/kernel.cu'])

    def test_live_owner_cannot_be_reconciled(self):
        result=runner.start(self.root,spec(),self.out)
        result['stages']['check']=dict(status='interrupted',owner={'job':'123'})
        store.write(self.out/'evidence.json',result)
        with patch.object(scheduler,'job_state',return_value='RUNNING'):
            with self.assertRaises(RuntimeError): runner.reconcile(self.out)

    def test_wrong_reference_conditions_and_missing_metrics(self):
        (self.root/'reference.md').write_text('observations')
        values=context.collect(self.root,{'architecture':'sm90','layout':'row'},
                               [{'source':'reference.md','conditions':{'architecture':'sm100','layout':'column'}}])
        self.assertEqual(values[0]['applicability'],'inapplicable')
        self.assertIsNone(values[0]['observation'])

    def test_conflicts_and_terminal_states(self):
        self.assertTrue(scheduler.conflicts({'changes':['src/kernel.cu']},{'inputs':['src/kernel.cu']})['conflicting'])
        for state in ('RUNNING','COMPLETING','UNKNOWN'):
            self.assertFalse(scheduler.released(state))
