import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from lab.optimize import campaign, render, store, trace
from eval.legacy_tests.test_campaign import IDENTITY, PASS, QUALIFIED, experiment


SEGMENT0 = {
    'id': 0, 'gpu_sku': 'H100-SXM5-80GB', 'driver': '580.82.07',
    'cuda_runtime': '13.0', 'pytorch': '2.13.0', 'tilelang': '0.1.11',
    'clock_policy': 'unlocked', 'power_policy': 'default',
    'benchmark_protocol': 'latency-v2', 'capture_regime': 'cuda-graph',
}
SEGMENT1 = dict(SEGMENT0, id=1, driver='581.1')


def measured(segment, candidate_ms=None, parent_ms=None, validity='valid', anchor=None):
    value = dict(validity=validity, identity=IDENTITY, measurement_segment=segment,
                 measurement_context={'hostname': 'node-1', 'slurm_job_id': '123',
                                      'timestamp': 10.0})
    if candidate_ms is not None:
        value['candidate_ms'] = candidate_ms
    if parent_ms is not None:
        value['parent_incumbent_ms'] = parent_ms
    if anchor is not None:
        value['segment_anchor_ms'] = anchor
    return value


class OptimizationTraceTests(unittest.TestCase):
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
        baseline = dict(identity=IDENTITY,
                        objective={'name': 'e2e_chunk_latency_ms', 'unit': 'ms', 'value': 16.0},
                        measurement_segment=SEGMENT0,
                        measurement_context={'hostname': 'node-0', 'slurm_job_id': '100',
                                             'timestamp': 1.0})
        campaign.create(self.directory, baseline, 'e2e_chunk_latency_ms',
                        'latency-v2', 'pi05-perf-v1')

    def add(self, iteration, verdict, measurement, correctness=PASS,
            qualification=QUALIFIED):
        record = campaign.start(self.root, experiment(f'candidate-{iteration}'), self.directory)
        measurement['identity'] = dict(measurement['identity'],
                                       engine_revision=record['change']['engine_revision'])
        campaign.finalize(self.directory, iteration, verdict, correctness, measurement,
                          qualification, diagnostics={'nsys': f'iter-{iteration}.nsys-rep'})

    def populate(self):
        self.add(1, 'no_benefit', measured(SEGMENT0, 15.95, 16.0))
        self.add(2, 'accepted', measured(SEGMENT0, 15.8, 16.0))
        self.add(3, 'correctness_failed', measured(SEGMENT0, validity='not_run'),
                 correctness={'status': 'failed'}, qualification={'status': 'not_run'})
        self.add(4, 'invalid', measured(SEGMENT0, 15.7, 15.8, validity='invalid'),
                 qualification={'status': 'not_run'})
        self.add(5, 'blocked', measured(SEGMENT1, validity='valid', anchor=15.6),
                 qualification={'status': 'blocked'})
        self.add(6, 'accepted', measured(SEGMENT1, 15.4, 15.6))

    def test_normalized_curve_and_complete_trace_schema(self):
        self.populate()
        state = campaign.rebuild(self.directory)
        self.assertEqual(state['baseline_latency_ms'], 16.0)
        self.assertEqual(state['current_incumbent_latency_ms'], 15.4)
        self.assertAlmostEqual(state['improvement_vs_baseline_pct'], -3.75)
        self.assertEqual(state['experiments'], {
            'accepted': 2, 'no_benefit': 1, 'correctness_failed': 1,
            'invalid': 1, 'blocked': 1,
        })
        value = trace.normalize(self.directory)
        entries = value['iterations']
        self.assertEqual([item['current_incumbent_latency_ms'] for item in entries],
                         [16.0, 16.0, 15.8, 15.8, 15.8, 15.6, 15.4])
        self.assertEqual([item['measurement_segment']['id'] for item in entries],
                         [0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(entries[1]['candidate_latency_ms'], 15.95)
        self.assertEqual(entries[3]['verdict'], 'correctness_failed')
        self.assertEqual(entries[4]['verdict'], 'invalid')
        self.assertEqual(entries[5]['promotion'], 'not_promoted')
        self.assertAlmostEqual(entries[6]['delta_vs_parent_pct'], (15.4 / 15.6 - 1) * 100)
        self.assertAlmostEqual(entries[6]['delta_vs_baseline_pct'], (15.4 / 15.6 - 1) * 100)
        required = {'iteration', 'timestamp', 'target_key', 'measurement_segment',
                    'candidate_id', 'parent_incumbent', 'engine_revision', 'hypothesis',
                    'change_summary', 'correctness', 'measurement_validity',
                    'candidate_latency_ms', 'parent_incumbent_latency_ms',
                    'current_incumbent_latency_ms', 'campaign_baseline_latency_ms',
                    'delta_vs_parent_pct', 'delta_vs_baseline_pct', 'qualification',
                    'verdict', 'promotion', 'diagnostic_artifacts', 'experiment_cost'}
        self.assertTrue(required.issubset(entries[6]))
        metadata, points = render.from_trace(value)
        self.assertEqual(metadata.model_revision, IDENTITY['model_revision'])
        self.assertEqual(metadata.objective, 'e2e_chunk_latency_ms')
        self.assertEqual(points[1].incumbent_ms, 16.0)
        self.assertEqual(points[4].incumbent_ms, 15.8)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib optional dependency absent')
    def test_plot_preserves_same_run_control_distinct_from_historical_incumbent(self):
        self.add(1, 'accepted', measured(SEGMENT0, 15.0, 15.2))
        metadata, points = render.from_trace(trace.normalize(self.directory))
        self.assertEqual(points[0].incumbent_ms, 16.0)
        self.assertEqual(points[1].parent_ms, 15.2)
        output = self.base / 'paired.svg'
        render.render_optimization_progress(metadata=metadata, points=points, output_svg=output)
        svg = output.read_text()
        self.assertIn('same-run parent control', svg)
        self.assertIn('15.200 ms', svg)
        self.assertIn('15.000 ms', svg)

    def test_materialization_is_semantically_deterministic_from_ledger(self):
        self.populate()
        first = trace.materialize(self.directory)
        first_markdown = (self.directory / 'progress.md').read_text()
        for name in ('state.json', 'optimization_trace.json', 'progress.md', 'progress.svg',
                     'progress.png', 'progress.html'):
            (self.directory / name).unlink(missing_ok=True)
        second = trace.materialize(self.directory)
        self.assertEqual(first, second)
        self.assertEqual(first_markdown, (self.directory / 'progress.md').read_text())
        self.assertEqual(render.from_trace(first), render.from_trace(second))
        self.assertEqual([item['measurement_segment']['id'] for item in first['iterations']],
                         [item['measurement_segment']['id'] for item in second['iterations']])

    def test_target_injection_is_a_hard_error(self):
        record = campaign.start(self.root, experiment('injected'), self.directory)
        path = Path(record['directory']) / 'evidence.json'
        measurement = measured(SEGMENT0, validity='invalid')
        measurement['identity'] = dict(measurement['identity'],
                                       engine_revision=record['change']['engine_revision'])
        campaign.finalize(self.directory, 1, 'invalid', PASS, measurement,
                          {'status': 'not_run'})
        record = store.read(path)
        for key, value in (('hardware', 'other-gpu'), ('model_revision', 'other-revision'),
                           ('shape', {'chunk': 32}), ('precision', 'fp16')):
            changed = store.read(path)
            changed['measurement']['identity'][key] = value
            store.write(path, changed)
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'measurement has a different TargetKey'):
                    trace.normalize(self.directory)
            store.write(path, record)

    def test_blocked_attempt_without_measurement_identity_remains_visible(self):
        revision = subprocess.check_output(
            ['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True).strip()
        identity = dict(IDENTITY, engine_revision=revision)
        campaign.start(self.root, experiment('blocked-before-measurement', identity=identity),
                       self.directory)
        campaign.finalize(self.directory, 1, 'blocked', PASS,
                          {'validity': 'incomplete', 'reason': 'gate infrastructure'},
                          {'status': 'blocked'})
        entry = trace.normalize(self.directory)['iterations'][1]
        self.assertEqual(entry['verdict'], 'blocked')
        self.assertEqual(entry['measurement_segment']['id'], 0)
        self.assertIsNone(entry['candidate_latency_ms'])
        self.assertEqual(entry['engine_revision'], revision)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib optional dependency absent')
    def test_canonical_svg_and_optional_outputs_consume_plot_points(self):
        self.populate()
        command = [sys.executable, '-m', 'lab.optimize', 'campaign-render',
                   str(self.directory), '--png', '--html']
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        first = json.loads((self.directory / 'optimization_trace.json').read_text())
        first_plot = render.from_trace(first)
        svg = (self.directory / 'progress.svg').read_text()
        self.assertTrue(svg.startswith('<?xml'))
        for text in (IDENTITY['model_revision'], 'objective=e2e_chunk_latency_ms',
                     'iter 2: accepted', 'vs parent', 'vs baseline',
                     'segment 1 re-anchor', 'iter 3: correctness_failed',
                     'iter 4: invalid', 'iter 5: blocked', 'no valid latency'):
            self.assertIn(text, svg)
        self.assertGreater((self.directory / 'progress.png').stat().st_size, 1000)
        html = (self.directory / 'progress.html').read_text()
        self.assertIn(IDENTITY['model_revision'], html)
        self.assertIn('candidate-6', html)
        for name in ('state.json', 'optimization_trace.json', 'progress.md', 'progress.svg',
                     'progress.png', 'progress.html'):
            (self.directory / name).unlink()
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        second = json.loads((self.directory / 'optimization_trace.json').read_text())
        self.assertEqual(first, second)
        self.assertEqual(first_plot, render.from_trace(second))
        for name in ('progress.md', 'progress.svg', 'progress.png', 'progress.html'):
            self.assertTrue((self.directory / name).is_file())


if __name__ == '__main__':
    unittest.main()
