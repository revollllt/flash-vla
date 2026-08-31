# Kernel Task Contract: <task-name>

Mode: <auto | human>    Date: <YYYY-MM-DD>    Target: <e.g. hardware/nvidia/h100/pi05>

Objective: <one sentence — the op, where it sits in the pipeline, why now>

## Tensors

| name | shape (named -> fixed) | dtype | role | mutation / aliasing |
|---|---|---|---|---|
| x | (M_PAD, D) -> (64, 1024) | bf16 | in | — |
| out | <...> | bf16 | inout | written in place, graph-captured |

Fixed shapes bound from: <model spec / ABI header path>.
Dims that may vary, and their guard: <e.g. prompt_len, guarded by plan route>.

## Fusion region

Folded into the kernel: <op list>. The boundary is the tensor table above.
Intermediates never materialized: <names, with shapes — they still appear as
named locals in the reference>.

## Correctness

Reference: <path to T2 (+ T3 when required)>.
Validation command: <exact command>.
Gates: <which parity checks gate promotion; which are report-only>.

## Baselines — measured BEFORE candidate 1, exact production shape

| implementation | command | min ms | job |
|---|---|---|---|
| current production route | | | |
| torch / SDPA | | | |
| <library kernel, if one applies> | | | |

## Floor and promotion criteria

Floor: <arithmetic over measured tags, e.g. bytes / [ld.bw.dev.dram] +
[launch.lat.dev.ramp]> = <value>, with tags and job ids.
Promotion: <e.g. min_ms <= X at the production shape AND all gates pass>.

## Allowed approaches

<backends permitted; constraints: graph-capturable, no allocation on the
replay path, scratch via ScratchPool, PDL control points, plan-route
atomicity rules that apply, ...>

## Budget and stop

<candidate cap / non-improvement cap / job budget; defaults per
references/loop.md>
