"""Normalize Campaign Ledger facts into the one optimization trace schema."""
from pathlib import Path

from . import campaign, store


ENVIRONMENT_FIELDS = campaign.MEASUREMENT_ENVIRONMENT_FIELDS
CONTEXT_FIELDS = ('hostname', 'slurm_job_id', 'timestamp')


def _shape_profile(shape):
    return '-'.join(f'{key}{value}' for key, value in sorted(shape.items()))


def _segment(value):
    segment = value['measurement_segment']
    missing = [key for key in ('id', *ENVIRONMENT_FIELDS) if key not in segment]
    if missing:
        raise ValueError(f'measurement segment is missing: {missing}')
    if type(segment['id']) is not int or segment['id'] < 0:
        raise ValueError('measurement segment id must be a non-negative integer')
    return dict(segment)


def _pct(value, reference):
    return (value / reference - 1.0) * 100.0


def _context(value):
    context = value['measurement_context']
    missing = [key for key in CONTEXT_FIELDS if key not in context]
    if missing:
        raise ValueError(f'measurement context is missing: {missing}')
    return dict(context)


def normalize(directory):
    """Read only ledger facts; all renderers consume the returned value."""
    directory = Path(directory).resolve()
    campaign.validate(directory)
    metadata = store.read(directory / 'campaign.json')
    records = [store.read(path) for path in
               sorted((directory / 'runs').glob('iter-*/evidence.json'))]
    baseline_source = records[0]['measurement']['evidence']
    objective = baseline_source['objective']
    if objective.get('name') != metadata['objective'] or objective.get('unit') != 'ms':
        raise ValueError('baseline objective must match the campaign and use milliseconds')
    campaign_baseline = float(objective['value'])
    previous_segment = _segment(baseline_source)
    segment_anchor = campaign_baseline
    incumbent = campaign_baseline
    target = metadata['target']
    if not campaign.same_workload(metadata, baseline_source['identity']):
        raise ValueError('baseline measurement has a different TargetKey')
    if (not baseline_source['identity'].get('engine_revision')
            or baseline_source['identity']['engine_revision']
            != records[0]['change'].get('engine_revision')):
        raise ValueError('baseline measurement has a different engine revision')
    if previous_segment['benchmark_protocol'] != metadata['protocol']:
        raise ValueError('baseline measurement protocol differs from the campaign')
    trace_metadata = dict(hardware=target['hardware'], model=target['model'],
                          model_revision=target['model_revision'],
                          shape_profile=_shape_profile(target['shape']),
                          shape=dict(target['shape']),
                          precision=(metadata['execution_variant']['quantization']['mode']
                                     if 'execution_variant' in metadata else target['precision']),
                          objective=metadata['objective'], protocol=metadata['protocol'],
                          fixture=metadata['fixture'])
    if 'execution_variant' in metadata:
        trace_metadata['execution_variant'] = metadata['execution_variant']
        trace_metadata['inference_signature'] = target['inference_signature']
    entries = []
    for record in records:
        iteration = record['iteration']
        measurement = record['measurement']
        verdict = record.get('verdict')
        if iteration == 0:
            segment = previous_segment
            context = _context(baseline_source)
            candidate_ms = campaign_baseline
            parent_ms = None
            validity = measurement['validity']
        elif record['verdict'] is None and not measurement:
            segment = previous_segment
            context = {}
            candidate_ms = None
            parent_ms = None
            validity = 'incomplete'
        elif verdict in ('invalid', 'blocked', 'correctness_failed') and 'identity' not in measurement:
            identity = record['campaign_identity']
            if (not campaign.same_workload(metadata, identity)
                    or identity.get('engine_revision') != record['change'].get('engine_revision')):
                raise ValueError(f'iter-{iteration:03d} campaign identity does not match its change')
            segment = previous_segment
            context = dict(measurement.get('measurement_context', {}))
            candidate_ms = measurement.get('candidate_ms')
            candidate_ms = None if candidate_ms is None else float(candidate_ms)
            parent_ms = measurement.get('parent_incumbent_ms')
            parent_ms = None if parent_ms is None else float(parent_ms)
            validity = measurement.get('validity', 'incomplete')
        else:
            if 'identity' not in measurement:
                raise ValueError(f'iter-{iteration:03d} measurement has no Identity')
            identity = measurement['identity']
            if not campaign.same_workload(metadata, identity):
                raise ValueError(f'iter-{iteration:03d} measurement has a different TargetKey')
            if (not identity.get('engine_revision')
                    or identity['engine_revision'] != record['change'].get('engine_revision')):
                raise ValueError(f'iter-{iteration:03d} measurement has a different engine revision')
            segment = _segment(measurement)
            if segment['benchmark_protocol'] != metadata['protocol']:
                raise ValueError(f'iter-{iteration:03d} measurement protocol differs from the campaign')
            context = _context(measurement)
            if segment['id'] == previous_segment['id']:
                if segment != previous_segment:
                    raise ValueError('environment changed without a new measurement segment')
            elif segment['id'] == previous_segment['id'] + 1:
                if measurement.get('validity') != 'valid':
                    raise ValueError('a measurement segment needs a valid incumbent re-anchor')
                segment_anchor = float(measurement['segment_anchor_ms'])
                incumbent = segment_anchor
            else:
                raise ValueError('measurement segment ids must be monotonic and contiguous')
            previous_segment = segment
            candidate_ms = measurement.get('candidate_ms')
            candidate_ms = None if candidate_ms is None else float(candidate_ms)
            parent_ms = measurement.get('parent_incumbent_ms')
            parent_ms = None if parent_ms is None else float(parent_ms)
            validity = measurement['validity']
        if verdict in ('accepted', 'no_benefit') and iteration and (
                candidate_ms is None or parent_ms is None):
            raise ValueError(f'iter-{iteration:03d} performance verdict needs candidate and parent latency')
        if verdict == 'accepted' and iteration:
            incumbent = candidate_ms
        delta_parent = (None if candidate_ms is None or parent_ms is None
                        else _pct(candidate_ms, parent_ms))
        delta_baseline = (0.0 if iteration == 0 else None if candidate_ms is None
                          else _pct(candidate_ms, segment_anchor))
        entry = dict(
            iteration=iteration, timestamp=context.get('timestamp', record.get('timestamp')),
            target_key=target,
            measurement_segment=segment, measurement_context=context,
            candidate_id=record['candidate_id'], parent_incumbent=record['parent_incumbent'],
            engine_revision=record['change'].get('engine_revision'),
            hypothesis=record['hypothesis'], change_summary=record['change']['summary'],
            correctness=record['correctness'], measurement_validity=validity,
            candidate_latency_ms=candidate_ms, parent_incumbent_latency_ms=parent_ms,
            current_incumbent_latency_ms=incumbent,
            campaign_baseline_latency_ms=campaign_baseline,
            segment_baseline_latency_ms=segment_anchor,
            delta_vs_parent_pct=delta_parent, delta_vs_baseline_pct=delta_baseline,
            qualification=record['qualification'], verdict=verdict,
            reanchor=measurement.get('reanchor', False),
            promotion=('reanchor' if measurement.get('reanchor') else
                       'promoted' if verdict == 'accepted' else 'not_promoted'),
            diagnostic_artifacts=record['diagnostics'], experiment_cost=record['cost'])
        entries.append(entry)
    return dict(schema_version=1, campaign_id=metadata['id'], metadata=trace_metadata,
                iterations=entries)


def progress_markdown(value):
    """Format normalized values without reinterpreting evidence."""
    metadata = value['metadata']
    lines = [f'# Optimization progress: {value["campaign_id"]}', '',
             f'{metadata["hardware"]} | {metadata["model"]} @ {metadata["model_revision"]}',
             f'{metadata["shape_profile"]} | {metadata["precision"]}',
             f'objective={metadata["objective"]} | protocol={metadata["protocol"]}', '',
             '| Iteration | Segment | Candidate | Verdict | Candidate ms | Incumbent ms | '
             'Delta parent | Delta baseline | Summary |',
             '|---:|---:|---|---|---:|---:|---:|---:|---|']
    for item in value['iterations']:
        number = lambda field: '—' if item[field] is None else f'{item[field]:.3f}'
        percent = lambda field: '—' if item[field] is None else f'{item[field]:+.2f}%'
        summary = item['change_summary'].replace('|', '\\|')
        lines.append(f'| {item["iteration"]} | {item["measurement_segment"]["id"]} | '
                     f'{item["candidate_id"]} | {item["verdict"] or "active"} | '
                     f'{number("candidate_latency_ms")} | {number("current_incumbent_latency_ms")} | '
                     f'{percent("delta_vs_parent_pct")} | {percent("delta_vs_baseline_pct")} | '
                     f'{summary} |')
    return '\n'.join(lines) + '\n'


def materialize(directory):
    directory = Path(directory).resolve()
    value = normalize(directory)
    store.write(directory / 'optimization_trace.json', value)
    (directory / 'progress.md').write_text(progress_markdown(value))
    return value
