"""Persistent campaign ledger built from immutable metadata and run evidence."""
import fcntl
from pathlib import Path
import time

from . import runner, store
from eval import acceptance
from .schema import STAGES, validate as validate_experiment


VERDICTS = ('accepted', 'no_benefit', 'correctness_failed', 'invalid', 'blocked')
TERMINAL = set(VERDICTS)
MEASUREMENT_ENVIRONMENT_FIELDS = (
    'gpu_sku', 'driver', 'cuda_runtime', 'pytorch', 'tilelang',
    'clock_policy', 'power_policy', 'benchmark_protocol', 'capture_regime',
)


def target_key(identity):
    version = identity.get('schema_version')
    if version == 3:
        if not identity.get('model_revision') or not identity.get('inference_signature'):
            raise ValueError('Identity v3 needs architecture revision and inference signature')
        return {key: identity[key] for key in
                ('target', 'hardware', 'model', 'model_revision', 'inference_signature', 'shape')}
    if version == 2 and identity.get('model_revision'):
        # Existing ledgers retain their v2 semantics until explicitly migrated.
        return {key: identity[key] for key in
                ('target', 'hardware', 'model', 'model_revision', 'shape', 'precision')}
    raise ValueError('campaign evidence needs a resolved Identity v2 or v3')


def campaign_key(identity, *, objective, protocol):
    if identity.get('schema_version') != 3:
        raise ValueError('Campaign discovery needs explicitly migrated Identity v3')
    return dict(target=target_key(identity), execution_variant=identity['execution_variant'],
                objective=objective, benchmark_protocol=protocol)


def same_target(target, identity):
    return target == target_key(identity)


def same_workload(metadata, identity):
    return (same_target(metadata['target'], identity)
            and metadata.get('execution_variant') == identity.get('execution_variant'))


def _records(directory):
    paths = sorted((Path(directory) / 'runs').glob('iter-*/evidence.json'))
    return [(path, store.read(path)) for path in paths]


def create(directory, baseline_evidence, objective, protocol, fixture):
    directory = Path(directory).resolve()
    identity = baseline_evidence['identity']
    target = target_key(identity)
    if directory.exists():
        raise FileExistsError(directory)
    metadata = dict(version=1, id=directory.name, created=time.time(),
                    target=target, objective=objective,
                    protocol=protocol, fixture=fixture,
                    budget=acceptance.for_target(target['target'])['budget'])
    if identity.get('schema_version') == 3:
        metadata['execution_variant'] = identity['execution_variant']
    if metadata['budget'] is None:
        raise ValueError(f'no acceptance budget for {target["target"]}')
    baseline = dict(iteration=0, candidate_id='baseline', parent_incumbent=None,
                    timestamp=metadata['created'],
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
        if not same_workload(metadata, record['measurement']['evidence']['identity']):
            raise ValueError('iter-000 measurement has a different TargetKey')
        return
    if record.get('parent_incumbent') != incumbent:
        raise ValueError(f'iter-{expected_iteration:03d} parent is not the allocation incumbent')
    if record.get('verdict') not in TERMINAL | {None}:
        raise ValueError(f'unknown verdict {record.get("verdict")!r}')
    if not same_workload(metadata, record['campaign_identity']):
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
    baseline_objective = records[0][1]['measurement']['evidence'].get('objective', {})
    baseline_latency_ms = (float(baseline_objective['value'])
                           if baseline_objective.get('unit') == 'ms' else None)
    legacy_history = records[0][1]['measurement']['evidence'].get('legacy_history', {})
    failed = [dict(iteration=None, candidate_id=item['record'].get('id'),
                   mechanism=item['record'].get('thesis'), verdict='legacy_rejected',
                   evidence_level=item['evidence_level'], source=item['source'])
              for item in legacy_history.get('rejected', [])]
    active = None
    cost = {'cpu_seconds': 0.0, 'gpu_seconds': 0.0, 'jobs': []}
    candidates = 0
    non_improving = 0
    imported_reanchor_required = records[0][1]['measurement']['evidence'].get(
        'continuation', {}).get('reanchor_required', False)
    current_measurement_segment = records[0][1]['measurement']['evidence'].get(
        'measurement_segment', {'id': 0})['id']
    has_reanchor = False
    for expected, (path, record) in enumerate(records):
        _validate_record(metadata, record, expected, incumbent)
        run_cost = record.get('cost', {})
        cost['cpu_seconds'] += run_cost.get('cpu_seconds', 0.0)
        cost['gpu_seconds'] += run_cost.get('gpu_seconds', 0.0)
        cost['jobs'].extend(run_cost.get('jobs', []))
        verdict = record.get('verdict')
        measured_segment = record.get('measurement', {}).get('measurement_segment')
        if measured_segment is not None:
            current_measurement_segment = measured_segment['id']
        label = f'iter-{expected:03d}'
        if expected:
            has_reanchor |= record.get('kind') == 'reanchor'
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
    incumbent_iteration = int(incumbent.split('-', 1)[1])
    incumbent_measurement = records[incumbent_iteration][1]['measurement']
    current_incumbent_latency_ms = (
        baseline_latency_ms if incumbent_iteration == 0
        else incumbent_measurement.get('candidate_ms')
    )
    improvement_vs_baseline_pct = (
        (current_incumbent_latency_ms / baseline_latency_ms - 1.0) * 100.0
        if baseline_latency_ms is not None and current_incumbent_latency_ms is not None
        else None
    )
    experiments = {verdict: 0 for verdict in VERDICTS}
    for _, record in records[1:]:
        if record.get('verdict') in experiments:
            experiments[record['verdict']] += 1
    hypotheses_path = directory / 'hypotheses.json'
    highest_value_unresolved_hypotheses = (
        store.read(hypotheses_path).get('unresolved', []) if hypotheses_path.is_file() else []
    )
    cost['jobs'] = sorted({str(job) for job in cost['jobs']})
    budget = metadata['budget']
    used = dict(candidates=candidates, non_improving=non_improving, jobs=len(cost['jobs']))
    remaining = {key: max(0, budget[key] - used[key]) for key in budget}
    reanchor_required = imported_reanchor_required and not has_reanchor
    stage, next_action = 'candidate_selection', 'start the next candidate from the incumbent'
    campaign_status = 'BASELINED' if len(records) == 1 else 'ACTIVE'
    if active:
        run_dir, record = active
        unfinished = next((name for name, stage in record['stages'].items()
                           if stage['status'] in ('running', 'interrupted', 'failed')), None)
        if unfinished:
            stage = unfinished
            next_action = f'reconcile {run_dir}'
            campaign_status = 'PAUSED'
        else:
            stage = next((name for name in STAGES if name in record['spec']['stages']
                          if record['stages'].get(name, {}).get('status') != 'completed'),
                         'verdict')
            next_action = (f'run {stage} for {run_dir}' if stage != 'verdict'
                           else f'record a verdict for {run_dir}')
    elif any(value == 0 for value in remaining.values()):
        next_action = 'campaign budget exhausted'
        campaign_status = 'COMPLETED'
    if reanchor_required and active is None:
        next_action = 'record a fresh incumbent re-anchor before starting a candidate'
        campaign_status = 'PAUSED'
    state = dict(version=1, campaign_id=metadata['id'], status=campaign_status,
                 target=metadata['target'], objective=metadata['objective'],
                 protocol=metadata['protocol'], fixture=metadata['fixture'],
                 baseline='iter-000', current_incumbent=incumbent,
                 baseline_latency_ms=baseline_latency_ms,
                 current_incumbent_latency_ms=current_incumbent_latency_ms,
                 improvement_vs_baseline_pct=improvement_vs_baseline_pct,
                 current_stage=stage,
                 current_measurement_segment=current_measurement_segment,
                 experiments=experiments,
                 highest_value_unresolved_hypotheses=highest_value_unresolved_hypotheses,
                 failed_hypotheses=failed,
                 legacy_import={key: len(legacy_history.get(key, [])) for key in
                                ('accepted', 'rejected', 'unclassified', 'experiment_evidence')},
                 budget=dict(limit=budget, used=used, remaining=remaining), cost=cost,
                 reanchor_required=reanchor_required,
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
    target_key(declared)
    if not declared.get('engine_revision'):
        raise ValueError('campaign candidate needs a resolved engine revision')
    with (directory / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = store.read(directory / 'campaign.json')
        if not same_workload(metadata, declared):
            raise ValueError('candidate TargetKey does not match campaign')
        if spec.get('protocol') != metadata['protocol'] or spec.get('fixture') != metadata['fixture']:
            raise ValueError('candidate protocol or fixture does not match campaign')
        state = rebuild(directory)
        if state['reanchor_required']:
            raise RuntimeError('campaign requires an incumbent re-anchor before a new candidate')
        if state['current_stage'] != 'candidate_selection':
            raise RuntimeError('campaign already has an active candidate')
        if any(value == 0 for value in state['budget']['remaining'].values()):
            raise RuntimeError('campaign budget exhausted')
        iteration = state['iterations']
        label = f'iter-{iteration:03d}'
        run_dir = directory / 'runs' / f'{label}-{spec["id"]}'
        metadata_fields = dict(iteration=iteration, candidate_id=spec['id'],
                               timestamp=time.time(),
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


def reanchor(directory, evidence):
    """Record a fresh incumbent measurement without claiming a code improvement."""
    directory = Path(directory).resolve()
    with (directory / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = store.read(directory / 'campaign.json')
        state = rebuild(directory)
        if state['current_stage'] != 'candidate_selection':
            raise RuntimeError('campaign already has an active candidate')
        identity = evidence['identity']
        if not same_workload(metadata, identity):
            raise ValueError('re-anchor TargetKey does not match campaign')
        if not identity.get('engine_revision'):
            raise ValueError('re-anchor needs a resolved engine revision')
        objective = evidence['objective']
        if objective.get('name') != metadata['objective'] or objective.get('unit') != 'ms':
            raise ValueError('re-anchor objective does not match campaign')
        incoming_segment = dict(evidence['measurement_segment'])
        missing = [key for key in MEASUREMENT_ENVIRONMENT_FIELDS if key not in incoming_segment]
        if missing:
            raise ValueError(f're-anchor measurement segment is missing: {missing}')
        if incoming_segment.get('benchmark_protocol') != metadata['protocol']:
            raise ValueError('re-anchor protocol does not match campaign')
        records = [record for _, record in _records(directory)]
        incumbent_iteration = int(state['current_incumbent'].split('-', 1)[1])
        previous = records[incumbent_iteration]['measurement']
        previous_segment = (previous['evidence']['measurement_segment']
                            if incumbent_iteration == 0 else previous['measurement_segment'])
        previous_value = (float(previous['evidence']['objective']['value'])
                          if incumbent_iteration == 0 else float(previous['candidate_ms']))
        value = float(objective['value'])
        drift_limit_ms = acceptance.for_target(metadata['target']['target'])[
            'latency']['control_spread_max_ms']
        drift_ms = abs(value - previous_value)
        drifted = drift_ms > drift_limit_ms
        expected_segment_id = previous_segment['id'] + int(drifted)
        if incoming_segment.get('id', expected_segment_id) != expected_segment_id:
            raise ValueError('re-anchor segment id does not match measured drift')
        segment = incoming_segment if drifted else dict(previous_segment)
        segment['id'] = expected_segment_id
        iteration = state['iterations']
        measurement = dict(identity=identity, validity='valid', measurement_segment=segment,
                           measurement_context=evidence['measurement_context'],
                           candidate_ms=value, parent_incumbent_ms=value,
                           segment_anchor_ms=value, reanchor=True,
                           reanchor_drift_ms=drift_ms,
                           reanchor_drift_limit_ms=drift_limit_ms,
                           evidence=evidence)
        record = dict(iteration=iteration, timestamp=evidence['measurement_context']['timestamp'],
                      candidate_id='incumbent-reanchor', parent_incumbent=state['current_incumbent'],
                      campaign_id=metadata['id'], campaign_identity=identity,
                      id=f'reanchor-{iteration:03d}', target=identity['target'], kind='reanchor',
                      spec=dict(protocol=metadata['protocol'], fixture=metadata['fixture'], stages={}),
                      hypothesis=dict(mechanism='incumbent re-anchor', alternative='environment drift',
                                      falsifier='fresh incumbent agrees within the acceptance policy',
                                      cheapest_probe='one incumbent A/A measurement',
                                      affected_call_sites=[], max_recoverable_ms=0.0,
                                      promotion_bar_ms=0.0),
                      change=dict(summary='fresh incumbent re-anchor', scope=[],
                                  engine_revision=identity['engine_revision']),
                      correctness=evidence.get('correctness', {'status': 'not_run'}),
                      measurement=measurement,
                      qualification={'status': 'not_applicable', 'gate_verdict': None},
                      diagnostics=evidence.get('diagnostics', {}),
                      cost=evidence.get('cost', {'cpu_seconds': 0.0, 'gpu_seconds': 0.0,
                                                 'jobs': []}),
                      stages={}, status='terminal', validity='valid', conclusion='reanchor',
                      promotion='not_requested', verdict='accepted')
        path = directory / 'runs' / f'iter-{iteration:03d}-incumbent-reanchor/evidence.json'
        store.write(path, record)
        return rebuild(directory)


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
    if state['current_stage'] == 'candidate_selection':
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
