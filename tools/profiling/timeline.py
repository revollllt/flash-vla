"""Single-capture software interval statistics, independent of Torch or CUDA."""
from collections import defaultdict


def intervals(events):
    """Durations, union and envelope of events from one capture, in microseconds."""
    spans = sorted((e['start_us'], e['end_us']) for e in events)
    merged = []
    for start, end in spans:
        if end < start:
            raise ValueError('event ends before it starts')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return {
        'sum_kernel_duration_us': sum(end - start for start, end in spans),
        'interval_union_us': sum(end - start for start, end in merged),
        'region_makespan_us': spans and max(end for _, end in spans) - spans[0][0] or 0.0,
    }


def occurrences(events):
    """Summarize individual call-site invocations, never the first-to-last layer."""
    grouped = defaultdict(list)
    for event in events:
        if event.get('invocation_id') is not None and event.get('call_site') is not None:
            grouped[(event['call_site'], event['invocation_id'])].append(event)
    return [dict(call_site=site, invocation_id=inv, **intervals(items))
            for (site, inv), items in sorted(grouped.items(), key=lambda kv: kv[0][1])]


def region_occurrences(events, members):
    """Recognize repeated complete atomic groups in mapped invocation order.

    Without a unique repeated group pattern, report unsupported rather than
    inventing correspondence between layers or across streams.
    """
    selected = [e for e in events if e.get('call_site') in members]
    if not selected or any(e.get('invocation_id') is None for e in selected):
        return {'status': 'unsupported', 'reason': 'missing invocation mapping', 'occurrences': []}
    if len({e.get('stream') for e in selected}) != 1:
        return {'status': 'unsupported', 'reason': 'multi-stream occurrence mapping', 'occurrences': []}
    by_invocation = defaultdict(list)
    for event in selected:
        by_invocation[event['invocation_id']].append(event)
    invocations = [items for _, items in sorted(by_invocation.items())]
    width = len(members)
    groups = []
    for offset in range(0, len(invocations), width):
        block = invocations[offset:offset + width]
        if len(block) != width or {item[0]['call_site'] for item in block} != set(members):
            return {'status': 'unsupported', 'reason': 'ambiguous repeated group', 'occurrences': []}
        groups.append(dict(occurrence=len(groups), **intervals([e for item in block for e in item])))
    return {'status': 'diagnostic_only', 'occurrences': groups,
            'sum_occurrence_makespan_us': sum(g['region_makespan_us'] for g in groups)}
