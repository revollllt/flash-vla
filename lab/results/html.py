"""Static context-local results dashboard; no network or application runtime."""
from html import escape
import json

FILTERS = (("hardware", "Hardware"), ("model", "Model"), ("revision", "Model revision"),
           ("shape", "Shape"), ("variant", "Execution variant"),
           ("context", "Checkpoint"), ("objective", "Objective"))


def _text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, dict) else str(value)


def render(campaigns):
    rows, choices = [], {key: set() for key, _ in FILTERS}
    for summary, contexts, directory in campaigns:
        key = summary["campaign_key"]
        target = key["target"]
        for context_id, context in sorted(contexts.items()):
            values = dict(hardware=target["hardware"], model=target["model"],
                          revision=target["model_revision"], shape=_text(target["shape"]),
                          variant=_text(key["execution_variant"]),
                          context=context["weights"]["checkpoint_id"], objective=key["objective"])
            for name in choices:
                choices[name].add(values[name])
            attributes = " ".join(f'data-{name}="{escape(value, quote=True)}"' for name, value in values.items())
            anchor, best = context["segment_anchor_ms"], context["best_validated_ms"]
            representative = ' <span class="badge">Representative</span>' if context_id == summary["representative_context"] else ""
            context_url = f"{directory}/contexts/{context_id}/summary.json"
            variant = key["execution_variant"]
            variant_label = (variant["quantization"]["mode"].upper()
                             + " · cache " + variant["cache"]["mode"])
            cells = [
                escape(values["hardware"]),
                f'<a href="{escape(directory, quote=True)}/README.md">{escape(values["revision"])}</a><small>{escape(values["model"])}</small>',
                f'<details><summary>View shape</summary><code>{escape(values["shape"])}</code></details>',
                f'<details><summary>{escape(variant_label)}</summary><code>{escape(values["variant"])}</code></details>',
                f'<a href="{escape(context_url, quote=True)}">{escape(values["context"])}</a>{representative}<small>Fixture: {escape(context["fixture"]["id"])} · Segment {context["latest_segment"]}</small>',
                escape(values["objective"]) + f'<small>{escape(key["benchmark_protocol"])}</small>',
                f"{anchor:.3f}", f"{best:.3f}", f"{anchor / best:.3f}×",
                str(context["last_validated_iteration"]),
                f'<a href="{escape(directory, quote=True)}/progress.svg">Progress plot</a><small>Lineage: {escape(summary["lineage_id"])}</small>',
            ]
            rows.append(f"<tr {attributes}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    filters = []
    for name, label in FILTERS:
        options = "".join(f'<option value="{escape(value, quote=True)}">{escape(value)}</option>'
                          for value in sorted(choices[name]))
        filters.append(f'<label>{label}<select data-filter="{name}"><option value="">All</option>{options}</select></label>')
    return PAGE_PREFIX + "".join(filters) + PAGE_TABLE + "".join(rows) + PAGE_END


PAGE_PREFIX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flash-VLA published results</title>
<style>
:root{color-scheme:light;font:15px/1.5 system-ui,sans-serif;color:#172536;background:#f3f6f8}
*{box-sizing:border-box}body{margin:0;padding:40px 3vw}main{max-width:1600px;margin:auto}
header{margin-bottom:28px}.eyebrow{color:#376d72;letter-spacing:.12em;font-size:12px;font-weight:700}
h1{font-size:clamp(26px,3vw,38px);letter-spacing:-.035em;margin:8px 0}
p{max-width:850px;color:#516174}a{color:#116c79;text-underline-offset:3px}
nav{display:flex;gap:20px}form{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;padding:20px;background:white;border:1px solid #dbe3e9;border-radius:12px}
label{display:flex;flex-direction:column;gap:7px;font-size:12px;font-weight:600;color:#516174;min-width:0}
select,button{font:inherit;border:1px solid #c6d3dc;border-radius:6px;padding:9px;background:white;color:#172536;min-width:0;width:100%}
button{align-self:end;cursor:pointer;font-size:14px;background:#eaf3f4}select:focus,button:focus{outline:2px solid #167d8a;outline-offset:2px}
#count{margin:18px 0 10px;font-size:13px}.table-scroll{overflow-x:auto;background:white;border:1px solid #dbe3e9;border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:1120px;text-align:left}th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;background:#edf2f5;color:#516174;white-space:nowrap}
th,td{padding:14px 12px;border-bottom:1px solid #e6ecef;vertical-align:top}td{font-size:13px;max-width:250px;overflow-wrap:anywhere}
td:nth-child(n+7):nth-child(-n+10){font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}tbody tr:hover{background:#f6fafb}small{display:block;font-size:11px;color:#617386;margin-top:5px}
code{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;white-space:nowrap}
.badge{display:inline-block;background:#e4f2ee;color:#286455;border-radius:4px;padding:2px 5px;font-size:10px;margin:5px 0}
#empty{padding:20px;margin:0}footer{font-size:12px;color:#617386;margin-top:18px}
@media(max-width:600px){body{padding:22px 14px}form{padding:14px;grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body><main>
<header><div class="eyebrow">FLASH-VLA · PUBLISHED RESULTS</div>
<h1>Inference performance, with context.</h1>
<p>Each row shows the latest validated segment for one checkpoint and fixture.
Speedup is relative to that segment's own anchor. Values from different contexts are not directly comparable.</p>
<nav><a href="README.md">Results overview</a><a href="index.json">Campaign index JSON</a></nav></header>
<form id="filters" aria-label="Filter published results">
"""
PAGE_TABLE = """<button type="reset">Reset filters</button></form>
<p id="count" role="status" aria-live="polite"></p>
<div class="table-scroll"><table aria-label="Published checkpoint contexts">
<thead><tr><th>Hardware</th><th>Model revision</th><th>Shape</th><th>Variant</th>
<th>Checkpoint / fixture</th><th>Objective</th><th>Anchor ms</th><th>Best ms</th>
<th>Speedup</th><th>Best iter</th><th>Evidence</th></tr></thead><tbody>
"""
PAGE_END = """</tbody></table><p id="empty" hidden>No published contexts match these filters.</p></div>
<footer>Generated from validated published summaries. Raw experiment logs and profiler captures remain outside this dashboard.</footer>
<script>
const rows = [...document.querySelectorAll('tbody tr')];
const filters = [...document.querySelectorAll('[data-filter]')];
function update() {
  let visible = 0;
  for (const row of rows) {
    row.hidden = !filters.every(filter => !filter.value || row.dataset[filter.dataset.filter] === filter.value);
    if (!row.hidden) visible++;
  }
  document.querySelector('#count').textContent = visible + ' of ' + rows.length + ' checkpoint contexts';
  document.querySelector('#empty').hidden = visible !== 0;
}
document.querySelector('#filters').addEventListener('change', update);
document.querySelector('#filters').addEventListener('reset', () => setTimeout(update, 0));
update();
</script>
</main></body></html>
"""
