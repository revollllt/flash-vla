import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eval import gate
from flash_vla.runtime.identity import Identity


def declared(model_revision='fixture/seed-0'):
    return SimpleNamespace(identity=Identity(
        target='hardware/nvidia/h100/pi05', hardware='h100-sxm5-80gb', model='pi05',
        model_revision=model_revision, shape={'chunk': 50, 'steps': 10},
        plan={'site': 'backend'}, precision='bf16', engine_revision='engine'))


class EarlyExitTests(unittest.TestCase):
    def run_with_checks(self, checks, baseline_checks):
        with patch.object(gate, '_run_in_engine_checks', return_value=checks), \
             patch.object(gate, '_run_baseline_checks', return_value=baseline_checks), \
             patch.object(gate, 'declare', return_value=declared()), \
             patch.object(gate, '_registry_version', return_value='test'), \
             patch.object(gate, '_finish', side_effect=lambda record, out: record), \
             patch.object(gate.latency, 'run') as measure, \
             patch.object(gate.floor_model, 'run') as floor:
            result = gate.run('h100/pi05')
            measure.assert_not_called()
            floor.assert_not_called()
        return result

    def test_correctness_failure_stops_before_timing(self):
        result = self.run_with_checks([dict(check='shallow', mode='gate', status='failed')], [])
        self.assertEqual(result['verdict'], 'fail')
        self.assertEqual(result['identity']['schema_version'], 2)
        self.assertIn('model_revision', result['identity'])
        self.assertIn('engine_revision', result['identity'])
        self.assertEqual(result['performance_incumbent']['plan'], 'shipped')
        self.assertEqual(result['numerical_oracle']['plan'], 'reference')
        self.assertEqual(result['correctness_coverage']['candidate_to_official'], 'not_run')

    def test_missing_official_evidence_blocks_before_timing(self):
        result = self.run_with_checks([dict(check='shallow', mode='gate', status='passed')],
                                     [dict(check='official', mode='gate', status='unavailable')])
        self.assertEqual(result['verdict'], 'blocked')

    def test_report_checks_not_run_after_required_failure(self):
        report = dict(replay_identical=True, finite=True, within_tolerance=False,
                      min_cosine=0, max_rel_rms=1, tolerance={})
        checks = [dict(check='shallow', mode='gate', oracle='in_engine_reference', threshold='shallow'),
                  dict(check='deep', mode='report', oracle='in_engine_reference')]
        with patch.object(gate.in_engine, 'run', return_value=report) as run:
            gate._run_in_engine_checks('target', 'candidate', checks, 0)
        self.assertEqual(run.call_count, 1)

    def test_missing_adapter_is_unavailable(self):
        checks = [dict(check='official', mode='gate', oracle='official_baseline')]
        result = gate._run_baseline_checks((), checks, True, sys.executable,
                                           declared().identity, 0)
        self.assertEqual(result[0]['status'], 'unavailable')

    def test_baseline_identity_mismatch_is_preserved_and_rejected(self):
        checks = [dict(check='official', mode='gate', oracle='official_baseline')]
        baseline = declared(model_revision='fixture/seed-1').identity.as_dict()
        stdout = json.dumps({'identity': baseline}) + '\n' + json.dumps({'identity': baseline})
        proc = subprocess.CompletedProcess([], 0, stdout=stdout, stderr='')
        with patch.object(gate.subprocess, 'run', return_value=proc) as invoke:
            result = gate._run_baseline_checks(('adapter',), checks, True, sys.executable,
                                               declared().identity, 0)
        self.assertEqual(result[0]['status'], 'mismatched')
        self.assertEqual(result[0]['scripts'][0]['identities'], [baseline, baseline])
        self.assertEqual(invoke.call_args.args[0][-2:], ['--seed', '0'])
