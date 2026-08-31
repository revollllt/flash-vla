# Attention-block tile analysis

Hypothesis-screening scripts for the Pi0.5 attention block. They price a
geometry before it is built (see `../attention_block_plan.md`, working rule 1)
using the measured constants in
`.claude/skills/hardware-unit-test/sm90/constants.yaml`; nothing here is a
measurement.

- `wave.py`    — CTA count, wave fraction and floor per tile choice for qkv,
                 o_proj and attention; the source of the proposal's floor table.
- `layout.py`  — row-major vs M-major activation: does the transpose pay once
                 split-K is free to rise?
- `budget.py`  — TMA descriptor legality, per-CTA smem pool union, accumulator
                 register union for one vs two math warpgroups.
- `check.cpp`  — host-side static checks over `sm90_attn_task_desc.cuh`
                 (task counts vs slots, trip counts, smem, workspace bytes,
                 reduction traffic). Build: `g++ -std=c++17 check.cpp -o check`.

Run the Python scripts with any interpreter; they have no dependencies.

## What the measurements corrected (2026-08-27, jobs 555047-555193)

- `wave.py` prices the copy column as `txns_per_warp x 248 ns` but not the
  latency exposed at every task start nor the serialisation of a 1-CTA/SM
  task chain: at BK=64 the qkv mainloop measured 4.7 us against a 2.0 us
  copy column. Bigger frames (BK=256 via a 3-D box, `budget.py` step 1 marks
  the 2-D BK=128 row as illegal but a {64, rows, 4} box is legal) cut it to
  3.3 us.
- `budget.py` does not model the S = Q K^T shared-memory traffic: at
  BKK=32 the m64n32k16 wgmma re-reads the Q tile per 1 KB of K and runs at
  ~75 cycles, 3x [wgmma.issue.wg.ss]; 64 keys per stage is the minimum.
- Neither script prices the split joins: a global-memory partial hop costs
  1.7-2.8 us on the critical path (fence + counter RTT + staging), three of
  which sit in series in this block.
