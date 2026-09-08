"""Persistent campaign ledger built from immutable metadata and run evidence."""
import fcntl
from pathlib import Path
import time

from . import runner, store
from eval import acceptance
from .schema import STAGES, validate as validate_experiment


VERDICTS = ('accepted', 'no_benefit', 'correctness_failed', 'invalid', 'blocked')
TERMINAL = set(VERDICTS)


def _target(identity):
    if identity.get('schema_version') != 2 or not identity.get('model_revision'):
        raise ValueError('campaign evidence needs Identity v2 with a resolved model revision')
    return {key: identity[key] for key in
            ('target', 'hardware', 'model', 'model_revision', 'shape', 'precision')}


def _same_target(target, identity):
    return target == _target(identity)


def _records(directory):
    paths = sorted((Path(directory) / 'runs').glob('iter-*/evidence.json'))
    return [(path, store.read(path)) for path in paths]


def create(directory, baseline_evidence, objective, protocol, fixture):
    directory = Path(directory).resolve()
    identity = baseline_evidence['identity']
    target = _target(identity)
    if directory.exists():
        raise FileExistsError(directory)
    metadata = dict(version=1, id=directory.name, created=time.time(),
                    target=target, objective=objective,
                    protocol=protocol, fixture=fixture,
                    budget=acceptance.for_target(target['target'])['budget'])
    if metadata['budget'] is None:
        raise ValueError(f'no acceptance budget for {target["target"]}')
    baseline = dict(iteration=0, candidate_id='baseline', parent_incumbent=None,
                    hypothesis=None,
                    change=dict(summary='campaign baseline', scope=[],
                                engine_revision=identity.get('engine_revision')),
                    correctness={'status': 'imported', 'evidence': None},
                    measurement={'validity': 'imported', 'evidence': baseline_evidence},
                    qualification={'status': 'imported', 'gate_verdict': None},
                    diagnostics={}, cost=baseline_evidence.get(
                        'cost', {'cpu_seconds': 0.0, 'gpu_seconds': 0.0, 'jobs': []}),
                    verdict='accepted')
    store.write(directory / 'campaign.json', metadata)
    store.write(directory / 'runs/iter-000-baseline/evidence.json', baseline)
    rebuild(directory)
    return metadata


def _validate_record(metadata, record, expected_iteration, incumbent):
    if record.get('iteration') != expected_iteration:
        raise ValueError(f'non-monotonic iteration: expected {expected_iteration}')
    if expected_iteration == 0:
        if record.get('parent_incumbent') is not None or record.get('verdict') != 'accepted':
            raise ValueError('iter-000 must be the accepted baseline')
        return
    if record.get('parent_incumbent') != incumbent:
        raise ValueError(f'iter-{expected_iteration:03d} parent is not the allocation incumbent')
    if record.get('verdict') not in TERMINAL | {None}:
        raise ValueError(f'unknown verdict {record.get("verdict")!r}')
    if not _same_target(metadata['target'], record['campaign_identity']):
        raise ValueError(f'iter-{expected_iteration:03d} has a different TargetKey')
    if record['spec'].get('protocol') != metadata['protocol']:
        raise ValueError(f'iter-{expected_iteration:03d} has a different protocol')
    if record['spec'].get('fixture') != metadata['fixture']:
        raise ValueError(f'iter-{expected_iteration:03d} has a different fixture')


def rebuild(directory):
    """Validate authoritative files and atomically recreate derived state.json."""
    directory = Path(directory).resolve()
    metadata = store.read(directory / 'campaign.json')
    if metadata.get('version') != 1:
        raise ValueError(f'unsupported campaign version {metadata.get("version")!r}')
    records = _records(directory)
    if not records:
        raise ValueError('campaign has no iter-000 baseline')
    incumbent = 'iter-000'
    failed = []
    active = None
    cost = {'cpu_seconds': 0.0, 'gpu_seconds': 0.0, 'jobs': []}
    candidates = 0
    non_improving = 0
    for expected, (path, record) in enumerate(records):
        _validate_record(metadata, record, expected, incumbent)
        run_cost = record.get('cost', {})
        cost['cpu_seconds'] += run_cost.get('cpu_seconds', 0.0)
        cost['gpu_seconds'] += run_cost.get('gpu_seconds', 0.0)
        cost['jobs'].extend(run_cost.get('jobs', []))
        verdict = record.get('verdict')
        label = f'iter-{expected:03d}'
        if expected:
            candidates += record.get('kind') == 'performance'
            if verdict == 'accepted':
                incumbent = label
                non_improving = 0
            elif verdict == 'no_benefit':
                non_improving += 1
            if verdict in TERMINAL - {'accepted'}:
                failed.append(dict(iteration=expected, candidate_id=record['candidate_id'],
                                   mechanism=record['hypothesis']['mechanism'], verdict=verdict))
            if verdict is None:
                if active is not None:
                    raise ValueError('campaign has multiple active candidates')
                active = (path.parent, record)
    cost['jobs'] = sorted(set(cost['jobs']))
    budget = metadata['budget']
    used = dict(candidates=candidates, non_improving=non_improving, jobs=len(cost['jobs']))
    remaining = {key: max(0, budget[key] - used[key]) for key in budget}
    segment, next_action = 'candidate_selection', 'start the next candidate from the incumbent'
    campaign_status = 'BASELINED' if len(records) == 1 else 'ACTIVE'
    if active:
        run_dir, record = active
        unfinished = next((name for name, stage in record['stages'].items()
                           if stage['status'] in ('running', 'interrupted', 'failed')), None)
        if unfinished:
            segment = unfinished
            next_action = f'reconcile {run_dir}'
            campaign_status = 'PAUSED'
        else:
            segment = next((name for name in STAGES if name in record['spec']['stages']
                            if record['stages'].get(name, {}).get('status') != 'completed'),
                           'verdict')
            next_action = (f'run {segment} for {run_dir}' if segment != 'verdict'
                           else f'record a verdict for {run_dir}')
    elif any(value == 0 for value in remaining.values()):
        next_action = 'campaign budget exhausted'
        campaign_status = 'COMPLETED'
    state = dict(version=1, campaign_id=metadata['id'], status=campaign_status,
                 target=metadata['target'], objective=metadata['objective'],
                 protocol=metadata['protocol'], fixture=metadata['fixture'],
                 baseline='iter-000', current_incumbent=incumbent,
                 current_measurement_segment=segment, failed_hypotheses=failed,
                 budget=dict(limit=budget, used=used, remaining=remaining), cost=cost,
                 next_action=next_action, iterations=len(records))
    store.write(directory / 'state.json', state)
    return state


def validate(directory):
    return rebuild(directory)


def start(root, spec, directory):
    """Atomically allocate a candidate against the last accepted incumbent."""
    directory = Path(directory).resolve()
    validate_experiment(spec)
    if spec['kind'] != 'performance':
        raise ValueError('campaign candidates must be performance experiments')
    required = ('affected_call_sites', 'max_recoverable_ms', 'promotion_bar_ms')
    missing = [key for key in required if key not in spec['hypothesis']]
    if missing:
        raise ValueError(f'missing campaign hypothesis fields: {missing}')
    if not spec.get('change', {}).get('summary'):
        raise ValueError('campaign candidate needs change.summary')
    declared = spec['identity']
    _target(declared)
    with (directory / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = store.read(directory / 'campaign.json')
        if not _same_target(metadata['target'], declared):
            raise ValueError('candidate TargetKey does not match campaign')
        if spec.get('protocol') != metadata['protocol'] or spec.get('fixture') != metadata['fixture']:
            raise ValueError('candidate protocol or fixture does not match campaign')
        state = rebuild(directory)
        if state['current_measurement_segment'] != 'candidate_selection':
            raise RuntimeError('campaign already has an active candidate')
        if any(value == 0 for value in state['budget']['remaining'].values()):
            raise RuntimeError('campaign budget exhausted')
        iteration = state['iterations']
        label = f'iter-{iteration:03d}'
        run_dir = directory / 'runs' / f'{label}-{spec["id"]}'
        metadata_fields = dict(iteration=iteration, candidate_id=spec['id'],
                               parent_incumbent=state['current_incumbent'],
                               campaign_id=metadata['id'], campaign_identity=spec['identity'],
                               hypothesis=spec['hypothesis'],
                               change=dict(summary=spec['change']['summary'], scope=spec['scope']),
                               measurement={}, qualification={}, diagnostics={}, verdict=None)
        record = runner.start(root, spec, run_dir, metadata=metadata_fields)
        record['change']['engine_revision'] = record['source']['revision']
        store.write(run_dir / 'evidence.json', record)
        rebuild(directory)
        return record


def finalize(directory, iteration, verdict, correctness, measurement, qualification,
             diagnostics=None):
    if verdict not in VERDICTS:
        raise ValueError(f'unknown verdict {verdict!r}')
    directory = Path(directory).resolve()
    with (directory / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = rebuild(directory)
        matches = list((directory / 'runs').glob(f'iter-{iteration:03d}-*/evidence.json'))
        if len(matches) != 1:
            raise ValueError(f'expected one iter-{iteration:03d} evidence record')
        record = store.read(matches[0])
        if record.get('verdict') is not None:
            raise ValueError('iteration verdict is immutable')
        if record['parent_incumbent'] != state['current_incumbent']:
            raise ValueError('candidate is no longer based on the incumbent')
        correctness_pass = correctness.get('status') == 'pass'
        measurement_valid = measurement.get('validity') == 'valid'
        qualification_pass = (qualification.get('status') == 'pass'
                              and qualification.get('gate_verdict') == 'pass')
        if verdict == 'accepted' and not (correctness_pass and measurement_valid
                                          and qualification_pass):
            raise ValueError('accepted requires correctness, valid measurement, qualification and gate pass')
        if verdict == 'no_benefit' and not (correctness_pass and measurement_valid):
            raise ValueError('no_benefit requires correctness and a valid measurement')
        if verdict == 'correctness_failed' and correctness.get('status') != 'failed':
            raise ValueError('correctness_failed requires a failed correctness result')
        if verdict == 'invalid' and measurement_valid:
            raise ValueError('invalid cannot carry a valid measurement')
        record.update(correctness=correctness, measurement=measurement,
                      qualification=qualification, diagnostics=diagnostics or {},
                      verdict=verdict, status='terminal', validity=measurement.get('validity'),
                      conclusion=('improved' if verdict == 'accepted' else verdict))
        store.write(matches[0], record)
        return rebuild(directory)


def resume(directory, until='measure', recovered_seconds=None):
    directory = Path(directory).resolve()
    state = rebuild(directory)
    if state['current_measurement_segment'] == 'candidate_selection':
        return state
    matches = list((directory / 'runs').glob('iter-*/evidence.json'))
    active = [path.parent for path in matches if store.read(path).get('verdict') is None]
    if len(active) != 1:
        raise ValueError('expected one active campaign iteration')
    record = store.read(active[0] / 'evidence.json')
    unfinished = any(stage['status'] in ('running', 'interrupted', 'failed')
                     for stage in record['stages'].values())
    if unfinished:
        if recovered_seconds is None:
            raise RuntimeError('interrupted stage requires explicit reconcile before resume')
        runner.reconcile(active[0], recovered_seconds)
    runner.run(active[0], until)
    return rebuild(directory)
