"""Small evidence context with explicit applicability, never inferred hardware facts."""
from pathlib import Path
from .schema import relative_path


def collect(root, conditions, references):
    entries = []
    for reference in references:
        source = Path(root) / relative_path(reference['source'])
        declared = reference.get('conditions', {})
        mismatches, unknown = [], []
        for key in ('architecture', 'precision', 'layout', 'synchronization', 'kernel'):
            if key not in conditions or key not in declared:
                unknown.append(key)
            elif declared[key] != conditions[key]:
                mismatches.append(key)
        entries.append(dict(source=str(source), conditions=declared,
                            applicability='inapplicable' if mismatches else 'unknown' if unknown else 'applicable',
                            mismatches=mismatches, unknown=unknown,
                            observation=reference.get('observation'),
                            hypotheses=reference.get('hypotheses', []),
                            next_discriminating_probe=reference.get('next_discriminating_probe'),
                            excerpt=source.read_text()[:3000]))
    return entries
