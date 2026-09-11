# Worked Query Examples

How to translate a situation the candidate loop meets into a navigation path
and a synthesis. Commands run from the skill directory with the repository
venv (`.venv/bin/python`).

---

## Example 1: "The report says gmma stalls dominate and the mainloop is 3x its floor"

**Navigation path**:
1. `scripts/query.py --symptom stall-gmma` → `pattern-serialized-wgmma`
2. Read its Diagnosis Checklist: read the full ptxas log, confirm RS form
3. Candidate technique: `technique-wgmma-rs-fragment-parity` (the fix and its snippet)
4. Cite the floor by tag: `[wgmma.issue.wg.ss]`; do not restate a number

**Command**:
```bash
.venv/bin/python scripts/query.py --symptom stall-gmma --compact
.venv/bin/python scripts/get_page.py technique-wgmma-rs-fragment-parity
```

**Ledger line**: `thesis: apply technique-wgmma-rs-fragment-parity; expect ~2-3x on the mainloop per pattern-serialized-wgmma (measured)`.

---

## Example 2: "Should I fuse these three launches into one persistent kernel?"

**Navigation path**:
1. `pattern-fusion-latency-chain` — price boundaries removed x
   `[launch.lat.dev.ramp]` against hops added x `[atom.lat.dev.hop]`
2. `kernel-megakernel-forms` — the four forms, their measured STATUS rows,
   and the rule that decides them (the dependency wait under the weight stream)
3. If the composition wins, `technique-producer-fusion-pdl` keeps the
   launches and buys the overlap instead

**Command**:
```bash
.venv/bin/python scripts/query.py --symptom fusion-regression --compact
.venv/bin/python scripts/get_page.py kernel-megakernel-forms --follow-sources
```

**Synthesis**: quote template 40's loss (115.25 us vs 90.75 us for three launches) and template 42's win only as the STATUS rows they are, with gpu, dtype, shape, metric, value, source.

---

## Example 3: "A library kernel is faster than ours at the same shape and the tile sweep is flat"

**Navigation path**:
1. `pattern-epilogue-bound-short-k` — the symptom is the epilogue, not the mainloop
2. `technique-bulk-store-publish` — stage the tile, one wide store
3. `technique-epilogue-fusion` — check whether cuBLASLt already fuses what you need

**Command**:
```bash
.venv/bin/python scripts/query.py --symptom tile-sweep-flat --compact
```

---

## Example 4: "My fused kernel wins in the benchmark and loses in the graph"

**Navigation path**:
1. `pattern-isolated-timer-overstates-fusion` — the cold timer charged the incumbent a DRAM read it never pays
2. `technique-same-process-aba` — measure in the graph regime; whole-stage wall time per layer
3. `doc-benchmark-kernel` for the harness

**Command**:
```bash
.venv/bin/python scripts/query.py --symptom isolated-vs-in-graph-mismatch --compact
```

---

## Example 5: "Where should the PDL trigger go?"

**Navigation path**:
1. `technique-pdl-placement` — the wait is derived, the trigger is swept; make it a compile-time knob
2. `hw-pdl-gdc` — what each instruction promises
3. `pattern-pdl-primary-is-a-resource` before folding any small primary away

**Command**:
```bash
.venv/bin/python scripts/get_page.py technique-pdl-placement --body-only
.venv/bin/python scripts/grep_wiki.py "griddepcontrol" --only wiki
```

---

## Example 6: "Every lever on this phase measures null"

**Navigation path**:
1. `pattern-stacked-floors` — two floors close together; compare the phase to the machine, cold and warm, in one job
2. `pattern-one-sided-gradient` — do not read a downward ablation as an upward gain
3. `pattern-cold-burst-ceiling` if the phase is a cold weight stream

**Command**:
```bash
.venv/bin/python scripts/query.py --symptom null-ablation --compact
```

---

## Example 7: "How does FlashAttention-3 overlap softmax with the GEMM, and should this decode kernel do it?"

**Navigation path**:
1. `kernel-flash-attention-3` — pingpong is latency hiding, not throughput (`[wgmma.ratio.sm.wg2]`)
2. `technique-release-on-retirement` — the cheaper alternative when the epilogue does not gate the copy column
3. `kernel-flashmla` — the decode-shaped structure with a separate combine

**Command**:
```bash
.venv/bin/python scripts/get_page.py kernel-flash-attention-3 --follow-sources
```

---

## Example 8: "What is the fastest way to load a row-major BK=256 tile?"

**Navigation path**:
1. `technique-tma-3d-box-row-major` — one 3-D box, K-major landing image
2. `hw-tma` — descriptor constraints; `[tma.bytes.txn.max]`, `[tma.bytes.txn.dtype]`

**Command**:
```bash
.venv/bin/python scripts/query.py "row-major deep-K tile TMA box" --architecture sm90 --compact
```

---

## Example 9: "What changes when this kernel is ported to Blackwell?"

**Navigation path**:
1. `migration-wgmma-to-tcgen05`, `migration-register-to-tmem`
2. `hw-tcgen05-mma`, `hw-tmem` — the appendix pages; check `architectures:` before transferring any sm90 rule

**Command**:
```bash
.venv/bin/python scripts/query.py --architecture sm100 --type migration --compact
```

---

## Example 10: "Which pages are this repository's own measurements?"

**Command**:
```bash
.venv/bin/python scripts/query.py --confidence measured --architecture sm90 --compact
```

Each carries `evidence_basis` naming a `hardware-unit-test` document, a
reference-grade template's `STATUS` block, or an Agent Note (`note-*`).

---

## Synthesis Pattern

```
1. Symptom framing (cite the pattern page; name the ncu stall reason or rule)
2. Mechanism (cite the hardware page; constants by tag, never by number)
3. The move (cite the technique page; include its snippet, named by file)
4. Price and evidence (performance_claims with all six fields; STATUS rows
   are this repository's, upstream rows are upstream's)
5. References (source ids: doc-*, blog-*, pr-*, note-*)
```

## Anti-Patterns

- Don't recommend a move without citing `sources:`.
- Don't quote a machine number; cite its tag.
- Don't quote a kernel number without the six-field record and whose it is.
- Don't conflate `sm90` and `sm100` pages; check `architectures:`.
- Don't treat `source-reported` upstream numbers as this machine's.
- Don't put a job id, a revision or "our kernel" on a page; the Agent Note
  the page cites holds them.
