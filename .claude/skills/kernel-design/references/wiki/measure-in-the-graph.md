---
id: measure-in-the-graph
type: pattern
arch: sm90
tags: [benchmark, cuda-graph, l2, fusion, methodology]
confidence: measured
---

# An isolated cold timer overstates a fusion's prize when the producer just wrote the buffer

## Context

A per-kernel benchmark with rotating cold inputs says the route you want to
replace costs X, your fused kernel costs less, and the fusion looks like a
win. In the captured graph the same replacement is neutral or negative.

## Move

Before pricing a fusion, ask what the *predecessor in the graph* leaves in
cache. A cold-input timer charges the incumbent a DRAM read that the real
pipeline never pays: its operand was just written by the kernel before it and
is L2-resident. Measure the incumbent in graph context (replay the captured
graph and read the position out of the trace) and use that number as the
baseline. Expect the incumbent to be materially cheaper there -- a library
attention kernel measured ~11 us cold and ~8.5 us in place, which was the
entire margin the fusion was built to capture.

The same applies in reverse to the candidate: a fused kernel that removes a
node does not necessarily remove that node's *gap*. Compare whole-stage wall
time per layer, not the sum of kernel durations.

## Why it works

The cold-rotating-buffer regime exists to make weight streams honest, and it
does. Activations are the opposite case: they are produced in place, one
kernel earlier, and are the operand a fusion is usually trying to keep in
registers or shared memory. Charging them at DRAM prices credits the fusion
with bytes nobody was paying.

## Caveats

Keep the cold timer for weight-dominated kernels and for ranking configs of
one implementation, where the bias is common to every row. Switch to
in-graph measurement the moment the decision is "fuse or not".
