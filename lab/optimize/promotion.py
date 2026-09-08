"""Evidence applicability checks; the existing eval.gate remains the only verdict."""
from pathlib import Path
from .store import changed_inputs


def applicability(evidence, incumbent, candidate, acceptance, required_targets):
    reasons=[]
    for role, current in (('incumbent', incumbent), ('candidate', candidate)):
        previous = evidence.get(role + '_source')
        if previous is None or any(not (Path(previous['root']) / p).is_file() for p in previous['inputs']):
            reasons.append(f'{role} source evidence missing; remeasure affected candidate')
        elif changed_inputs(previous, current):
            reasons.append(f'{role} changed; repeat affected correctness and measurement')
    if evidence.get('acceptance')!=acceptance:
        reasons.append('acceptance changed; existing evidence is not applicable')
    missing=set(required_targets)-set(evidence.get('qualified_targets',[]))
    if missing:
        reasons.append(f'missing affected targets: {sorted(missing)}')
    if evidence.get('gate_verdict')!='pass':
        reasons.append('existing gate has not passed')
    return dict(applicable=not reasons,reasons=reasons,action='review' if not reasons else 'remeasure',
                rollback_source=incumbent['root'],automatic_promotion=False)
