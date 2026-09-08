"""Filter software observations and validate cross-role operation tokens."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def validate_tokens(raw):
    """Pair issue/completion observations by operation and generation, not warp."""
    grouped=defaultdict(dict)
    for marker in raw.get('markers', []):
        key=tuple(marker[field] if field!='cta' else tuple(marker[field])
                  for field in ('device_id','launch_id','replay_id','cta','operation','token','generation'))
        kind=marker['kind']
        if kind not in ('issue','completion_observed') or kind in grouped[key]:
            raise ValueError('unknown or duplicate operation marker')
        if type(marker['timestamp_ns']) is not int:
            raise ValueError('integer nanosecond marker timestamp required')
        grouped[key][kind]=marker
    pairs=[]; incomplete=[]
    for key, endpoints in grouped.items():
        if set(endpoints)!={'issue','completion_observed'}:
            incomplete.append(dict(identity=key,present=list(endpoints)))
            continue
        start=endpoints['issue']; end=endpoints['completion_observed']
        if end['timestamp_ns']<start['timestamp_ns']:
            raise ValueError('completion observed before issue')
        pairs.append(dict(identity=key,issue=start,completion_observed=end,
                          observed_window_ns=end['timestamp_ns']-start['timestamp_ns'],
                          interpretation='observation window, not asynchronous engine active time'))
    return dict(pairs=pairs,incomplete=incomplete,complete=not incomplete and not raw['dropped_records'])


def query(raw, **filters):
    if raw['clock_domain']!='gpu-local' or raw['timer_unit']!='ns':
        raise ValueError('unknown clock: no cross-domain alignment')
    ranges=[r for r in raw['ranges'] if all(r.get(key)==value for key,value in filters.items())]
    waits=[r for r in ranges if r['semantic']=='wait_scope']
    tails=sorted(ranges,key=lambda r:r['end_ns']-r['start_ns'],reverse=True)[:10]
    return dict(filters=filters,range_count=len(ranges),coverage=raw.get('coverage','unknown'),
                dropped_records=raw['dropped_records'],clock_domain=raw['clock_domain'],
                waits=waits,tails=tails,tokens=validate_tokens(raw),
                observation='selected software scopes only',hypotheses=[],
                next_discriminating_probe=None,hardware_active='not_measured')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    for key in ('sm_begin','launch_id','replay_id','task_id','iteration'):
        parser.add_argument('--'+key.replace('_','-'),type=int)
    for key in ('stage','role','call_site'):
        parser.add_argument('--'+key.replace('_','-'))
    parser.add_argument('--out',type=Path)
    args=vars(parser.parse_args())
    raw=json.loads(args.pop('input').read_text()); out=args.pop('out')
    result=query(raw,**{key:value for key,value in args.items() if value is not None})
    text=json.dumps(result,indent=2)
    if out: out.write_text(text+'\n')
    print(text)


if __name__=='__main__': main()
