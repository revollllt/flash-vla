import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from lab.optimize import campaign, migrate, store, trace
from eval.tests.test_campaign import experiment
from eval.tests.test_optimization_trace import SEGMENT1


def legacy_report(target, revision):
    model = target
    prompt = 0 if target == 'pi0' else 200
    identity = dict(target=f'hardware/nvidia/h100/{target}', hardware='h100-sxm5-80gb',
                    model=model, shape_profile=f'chunk50-prompt_len{prompt}',
                    shape={'chunk': 50, 'steps': 10, 'prompt_len': prompt},
                    precision='bf16', plan={'model': 'shipped'}, revision=revision)
    return dict(identity=identity,
                env=dict(gpu='NVIDIA H100 80GB HBM3', driver='570.86.10',
                         torch_cuda='13.0', torch='2.13.0+cu130', tilelang='0.1.11',
                         clocks='unlocked', node='ACD1-3', job='600566'),
                config={'seed': 0}, deltas={'reference_leg': 0},
                legs=[dict(identity=identity,
                           metrics={'chunk_latency': {'min': 14.5}},
                           attribution={'started_unix': 100.0})])


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.org', 'commit', '--allow-empty',
                        '-qm', 'legacy'], check=True)
        self.revision = subprocess.check_output(
            ['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True).strip()
        (self.root / 'src').mkdir()
        (self.root / 'src/kernel.cu').write_text('original')
        for target in ('pi0', 'pi05'):
            store.write(self.root / f'artifacts/campaign/600566/latency_h100_{target}.json',
                        legacy_report(target, self.revision[:7]))
        lane = self.root / 'artifacts/ktasks/siglip'
        lane.mkdir(parents=True)
        rows = [dict(id='B2', status='kept', value=1.0),
                dict(id='B1', status='REJECTED', value=2.0),
                dict(id='B0', status='measured', value=3.0)]
        (lane / 'candidates.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')

    def test_both_campaigns_preserve_raw_values_and_mark_legacy_history(self):
        for target in ('pi0', 'pi05'):
            directory = Path(self.temp.name) / target
            original = store.read(
                self.root / f'artifacts/campaign/600566/latency_h100_{target}.json')
            migrate.pi_campaign(self.root, directory)
            baseline = store.read(directory / 'runs/iter-000-baseline/evidence.json')
            source = baseline['measurement']['evidence']
            self.assertEqual(source['raw_measurement'], original)
            self.assertEqual(source['evidence_level'], 'legacy_import')
            if target == 'pi0':
                self.assertEqual(source['legacy_history']['accepted'][0]['record']['value'], 1.0)
            else:
                self.assertEqual(source['legacy_history']['accepted'], [])
            self.assertEqual(source['legacy_history']['rejected'][0]['record']['value'], 2.0)
            self.assertIn(3.0, [item['record']['value']
                                for item in source['legacy_history']['unclassified']])
            state = campaign.rebuild(directory)
            self.assertEqual(state['current_incumbent'], 'iter-000')
            self.assertTrue(state['reanchor_required'])
            self.assertIn('re-anchor', state['next_action'])
            self.assertIn('legacy_rejected',
                          [item['verdict'] for item in state['failed_hypotheses']])
            self.assertEqual(store.read(
                self.root / f'artifacts/campaign/600566/latency_h100_{target}.json'), original)

    def test_reanchor_is_not_a_candidate_or_code_promotion(self):
        directory = Path(self.temp.name) / 'pi05'
        migrate.pi_campaign(self.root, directory)
        identity = store.read(directory / 'campaign.json')['target']
        identity = dict(schema_version=2, **identity, plan={'model': 'shipped'},
                        engine_revision=self.revision)
        spec = experiment('must-wait', identity=identity)
        spec['fixture'] = 'flash-vla-random-inputs-v1/seed-0'
        with self.assertRaisesRegex(RuntimeError, 'requires an incumbent re-anchor'):
            campaign.start(self.root, spec, directory)
        evidence = dict(identity=identity,
                        objective={'name': 'e2e_chunk_latency_ms', 'unit': 'ms', 'value': 14.2},
                        measurement_segment=SEGMENT1,
                        measurement_context={'hostname': 'ACD1-4', 'slurm_job_id': '606100',
                                             'timestamp': 200.0})
        state = campaign.reanchor(directory, evidence)
        self.assertFalse(state['reanchor_required'])
        self.assertEqual(state['current_incumbent'], 'iter-001')
        self.assertEqual(state['budget']['used']['candidates'], 0)
        value = trace.normalize(directory)
        self.assertTrue(value['iterations'][1]['reanchor'])
        self.assertEqual(value['iterations'][1]['promotion'], 'reanchor')
        self.assertEqual(value['iterations'][1]['current_incumbent_latency_ms'], 14.2)
        candidate = campaign.start(self.root, spec, directory)
        self.assertEqual(candidate['iteration'], 2)
        self.assertEqual(candidate['parent_incumbent'], 'iter-001')


if __name__ == '__main__':
    unittest.main()
