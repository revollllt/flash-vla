import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from lab.optimize import campaign, runner, scheduler, store


IDENTITY = {
    'schema_version': 2,
    'target': 'hardware/nvidia/h100/pi05',
    'hardware': 'h100-sxm5-80gb',
    'model': 'pi05',
    'model_revision': 'flash-vla/pi05-fixture/v1',
    'shape_profile': 'batch1-chunk50-steps10-views3',
    'shape': {'batch': 1, 'views': 3, 'chunk': 50, 'steps': 10},
    'precision': 'bf16',
    'plan': {'model': 'shipped'},
    'engine_revision': 'a' * 40,
}


def experiment(candidate='candidate', identity=None, stages=None):
    return dict(
        version=1, id=candidate, target=IDENTITY['target'], kind='performance',
        identity=identity or IDENTITY, protocol='latency-v2', fixture='pi05-perf-v1',
        hypothesis=dict(mechanism=f'{candidate} mechanism', alternative='memory traffic',
                        falsifier='probe is unchanged', cheapest_probe='one CPU probe',
                        affected_call_sites=['model'], max_recoverable_ms=0.2,
                        promotion_bar_ms=0.1),
        change={'summary': f'implement {candidate}'},
        scope=['src'], inputs=['src/kernel.cu'], changes=['src/kernel.cu'],
        stages=stages or {
            'check': dict(argv=[sys.executable, '-c', 'print("pass")'],
                          resource='cpu', timeout_s=10),
            'measure': dict(argv=[sys.executable, '-c', 'print("measured")'],
                            resource='cpu', timeout_s=10),
        })


PASS = {'status': 'pass'}
VALID = {'validity': 'valid'}
QUALIFIED = {'status': 'pass', 'gate_verdict': 'pass'}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'checkout'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.org', 'commit', '--allow-empty',
                        '-qm', 'baseline'], check=True)
        (self.root / 'src').mkdir()
        (self.root / 'src/kernel.cu').write_text('original')
        self.directory = self.base / 'campaign'
        campaign.create(self.directory, {'identity': IDENTITY}, 'e2e_chunk_latency_ms',
                        'latency-v2', 'pi05-perf-v1')

    def start(self, candidate):
        return campaign.start(self.root, experiment(candidate), self.directory)

    def finalize(self, iteration, verdict, correctness=PASS, measurement=VALID,
                 qualification=QUALIFIED):
        return campaign.finalize(self.directory, iteration, verdict, correctness,
                                 measurement, qualification)

    def test_baseline_and_deleted_state_are_rebuilt(self):
        baseline = store.read(self.directory / 'runs/iter-000-baseline/evidence.json')
        self.assertEqual(baseline['iteration'], 0)
        self.assertEqual(baseline['verdict'], 'accepted')
        (self.directory / 'state.json').unlink()
        state = campaign.rebuild(self.directory)
        self.assertEqual(state['baseline'], 'iter-000')
        self.assertEqual(state['current_incumbent'], 'iter-000')
        self.assertTrue((self.directory / 'state.json').is_file())
        with self.assertRaises(FileExistsError):
            campaign.create(self.directory, {'identity': IDENTITY}, 'changed',
                            'latency-v2', 'pi05-perf-v1')

    def test_lineage_keeps_rejections_and_moves_incumbent_atomically(self):
        first = self.start('first')
        self.assertEqual(first['parent_incumbent'], 'iter-000')
        self.finalize(1, 'no_benefit')
        rejected = self.directory / 'runs/iter-001-first/evidence.json'
        self.assertTrue(rejected.is_file())
        second = self.start('second')
        self.assertEqual(second['iteration'], 2)
        self.assertEqual(second['parent_incumbent'], 'iter-000')
        with self.assertRaises(ValueError):
            self.finalize(2, 'accepted', qualification={'status': 'failed', 'gate_verdict': 'fail'})
        self.assertEqual(campaign.rebuild(self.directory)['current_incumbent'], 'iter-000')
        self.finalize(2, 'accepted')
        third = self.start('third')
        self.assertEqual(third['parent_incumbent'], 'iter-002')
        state = campaign.rebuild(self.directory)
        self.assertEqual(state['current_incumbent'], 'iter-002')
        self.assertEqual(state['failed_hypotheses'][0]['verdict'], 'no_benefit')

    def test_invalid_is_not_no_benefit_and_cost_survives_new_process(self):
        baseline_path = self.directory / 'runs/iter-000-baseline/evidence.json'
        baseline = store.read(baseline_path)
        baseline['cost']['jobs'] = [9000]
        store.write(baseline_path, baseline)
        record = self.start('invalid-read')
        record['cost'] = {'cpu_seconds': 3.0, 'gpu_seconds': 17.0, 'jobs': ['9001']}
        store.write(Path(record['directory']) / 'evidence.json', record)
        self.finalize(1, 'invalid', measurement={'validity': 'invalid'},
                      qualification={'status': 'not_run'})
        state = campaign.rebuild(self.directory)
        self.assertEqual(state['budget']['used']['non_improving'], 0)
        self.assertEqual(state['cost']['gpu_seconds'], 17.0)
        self.assertEqual(state['cost']['jobs'], ['9000', '9001'])
        output = subprocess.check_output(
            [sys.executable, '-m', 'lab.optimize', 'campaign-status', str(self.directory)],
            text=True)
        fresh = json.loads(output)
        for key in ('target', 'baseline', 'current_incumbent', 'current_stage',
                    'current_measurement_segment', 'failed_hypotheses', 'budget',
                    'experiments', 'highest_value_unresolved_hypotheses', 'next_action'):
            self.assertIn(key, fresh)
        self.assertEqual(fresh['cost']['gpu_seconds'], 17.0)

    def test_ranked_unresolved_hypotheses_survive_state_rebuild(self):
        queue = {'version': 1, 'campaign_id': self.directory.name,
                 'unresolved': [{'id': 'next', 'mechanism': 'next mechanism'}]}
        store.write(self.directory / 'hypotheses.json', queue)
        (self.directory / 'state.json').unlink()
        state = campaign.rebuild(self.directory)
        self.assertEqual(state['highest_value_unresolved_hypotheses'], queue['unresolved'])

    def test_non_improving_budget_survives_state_rebuilds(self):
        for iteration in range(1, 4):
            self.start(f'flat-{iteration}')
            self.finalize(iteration, 'no_benefit')
            (self.directory / 'state.json').unlink()
            campaign.rebuild(self.directory)
        self.assertEqual(campaign.rebuild(self.directory)['budget']['remaining']['non_improving'], 0)
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            self.start('over-budget')

    def test_target_protocol_and_fixture_must_match(self):
        for mutate in (
            lambda value: value['identity'].update(model_revision='flash-vla/pi05-fixture/v2'),
            lambda value: value['identity']['shape'].update(chunk=32),
            lambda value: value['identity'].update(precision='fp16'),
            lambda value: value.update(protocol='latency-v3'),
            lambda value: value.update(fixture='other-fixture'),
        ):
            value = experiment('mismatch', identity=dict(IDENTITY, shape=dict(IDENTITY['shape'])))
            mutate(value)
            with self.subTest(value=value):
                with self.assertRaises((ValueError, KeyError)):
                    campaign.start(self.root, value, self.directory)

    def test_checkout_and_branch_are_not_needed_to_read_or_continue_ledger(self):
        self.start('detached')
        self.finalize(1, 'blocked', qualification={'status': 'blocked'})
        subprocess.run(['git', '-C', str(self.root), 'branch', 'discarded'], check=True)
        subprocess.run(['git', '-C', str(self.root), 'branch', '-D', 'discarded'],
                       check=True, stdout=subprocess.DEVNULL)
        shutil.rmtree(self.root)
        state = campaign.rebuild(self.directory)
        self.assertEqual(state['iterations'], 2)
        self.assertEqual(state['current_incumbent'], 'iter-000')
        self.assertEqual(state['current_stage'], 'candidate_selection')
        self.assertEqual(state['current_measurement_segment'], 0)

    def test_interrupted_segments_require_reconcile_before_one_resume(self):
        for segment in ('probe', 'measure', 'qualify'):
            with self.subTest(segment=segment):
                directory = self.base / f'campaign-{segment}'
                campaign.create(directory, {'identity': IDENTITY}, 'e2e_chunk_latency_ms',
                                'latency-v2', 'pi05-perf-v1')
                stages = {segment: ({} if segment == 'qualify' else
                                     dict(argv=[sys.executable, '-c', 'print("done")'],
                                          resource='gpu', timeout_s=10))}
                if segment in ('measure', 'qualify'):
                    stages = {'check': dict(argv=[sys.executable, '-c', 'print("pass")'],
                                            resource='cpu', timeout_s=10), **stages}
                spec = experiment(segment, stages=stages)
                record = campaign.start(self.root, spec, directory)
                if segment in ('measure', 'qualify'):
                    record['stages']['check'] = dict(status='completed', resource='cpu',
                                                     elapsed_s=1.0)
                    record['correctness'] = 'pass'
                record['stages'][segment] = dict(status='interrupted', owner=None,
                                                  resource='gpu', elapsed_s=9.0)
                record['cost']['gpu_seconds'] = 9.0
                record['status'] = 'stopped'
                store.write(Path(record['directory']) / 'evidence.json', record)
                state = campaign.rebuild(directory)
                self.assertEqual(state['current_stage'], segment)
                self.assertEqual(state['current_measurement_segment'], 0)
                with patch.object(runner, 'run') as resume_run:
                    with self.assertRaisesRegex(RuntimeError, 'explicit reconcile'):
                        campaign.resume(directory)
                    resume_run.assert_not_called()
                    campaign.resume(directory, recovered_seconds=0)
                    resume_run.assert_called_once()
                recovered = store.read(Path(record['directory']) / 'evidence.json')
                self.assertEqual(recovered['cost']['gpu_seconds'], 9.0)
                self.assertEqual(len(recovered['attempts']), 1)

    def test_cancelled_slurm_attempt_needs_an_explicit_invalid_verdict(self):
        record = self.start('slurm-cancelled')
        run = Path(record['directory'])
        record['stages']['check'] = dict(status='completed', resource='cpu', elapsed_s=1.0)
        record['stages']['measure'] = dict(
            status='running', resource='gpu', owner={'job': '123'}, started=1.0,
        )
        record['correctness'] = 'pass'
        record['status'] = 'stopped'
        record['cost']['jobs'] = ['123']
        store.write(run / 'evidence.json', record)

        with patch.object(scheduler, 'job_state', return_value='CANCELLED'):
            recovered = runner.reconcile(run, recovered_seconds=9.0)
        self.assertEqual(recovered['cost']['gpu_seconds'], 9.0)
        self.assertIsNone(recovered['verdict'])
        self.assertNotEqual(recovered['validity'], 'valid')

        state = self.finalize(
            1, 'invalid', measurement={'validity': 'invalid'},
            qualification={'status': 'not_run'},
        )
        self.assertEqual(state['current_incumbent'], 'iter-000')
        self.assertEqual(state['failed_hypotheses'][0]['verdict'], 'invalid')

    def test_simultaneous_start_allocates_only_one_monotonic_iteration(self):
        from concurrent.futures import ThreadPoolExecutor

        def attempt(candidate):
            try:
                return campaign.start(self.root, experiment(candidate), self.directory)['iteration']
            except RuntimeError as error:
                return str(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, ('one', 'two')))
        self.assertEqual(sum(value == 1 for value in results), 1)
        self.assertEqual(sum('active candidate' in str(value) for value in results), 1)


if __name__ == '__main__':
    unittest.main()
