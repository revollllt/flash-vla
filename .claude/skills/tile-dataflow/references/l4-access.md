# L4 accesses — the two maps, and the file that holds them

L4's job is to say what each access is **and how wide**, and every number it
reports — bits per thread, vector width, transactions, bank-conflict ways — is a
function of exactly two maps:

```
buffer   coord -> element offset        the layout, plus any swizzle
tv       (tid, vid) -> coord            who touches what
```

Write those two down and the rest is arithmetic over 32 lanes.
`scripts/tv_check.py` does the arithmetic; this file is the input format.

**Why a file and not a paragraph.** The prose form of an L4 table states
*conclusions* — `128 B swizzle atom, aligned -> 0-way bank conflict` — and a
conclusion cannot be rechecked, cannot be regenerated when L2 changes a tile
extent, and cannot be wrong out loud. Both worked examples marked most of their
conflict counts `[I]`, and `example-deepgemm.md` had to open
`open_questions.bank_ways` because nobody could settle one by hand. They are all
`[D]` now, computed, and the DeepGEMM figure was right.

## Contents

- [The file](#the-file)
- [buffer](#buffer)
- [tv](#tv)
- [What comes back](#what-comes-back)
- [The bank model](#the-bank-model)
- [Accesses with no per-thread address](#accesses-with-no-per-thread-address)
- [expect](#expect)

## The file

One YAML file per spec, named in the spec's `l4_accesses` field, holding one
entry per distinct touch in a steady-state stage plus the epilogue:

```yaml
accesses:
  - id: smem_d store          # the row label in the spec's L4 table
    space: smem               # smem | gmem
    op: st.shared.b64         # the instruction, for the table; not interpreted
    dtype: f32
    threads: 128              # threads participating in ONE instruction
    buffer: {kind: tile, rows: 64, cols: 128, swizzle: none}
    tv: {atom: wgmma_acc, n: 128, vec: 2, project: both}
    expect: {wavefronts: 8, ideal: 2}
```

```
python3 scripts/tv_check.py accesses.yaml              # the human report
python3 scripts/tv_check.py accesses.yaml --markdown   # the L4 table, to paste
```

## buffer

Two forms. Use `tile` whenever the buffer is a row-major smem tile, which is
almost always:

```yaml
buffer: {kind: tile, rows: 64, cols: 128, swizzle: 128}   # swizzle: none | 32 | 64 | 128
```

`swizzle` is the byte width of the CuTe atom, the way everyone names these. The
three `Swizzle<B,M,S>` parameters are then forced, not chosen — `M` from the
dtype so that 16 B stays contiguous, `B` from the mode, `S` always 3 — so the
same "128 B swizzle" is `Swizzle<3,3,3>` for bf16 and `Swizzle<3,4,3>` for fp8
and you never write either down.

**A tile wider than its atom is a tiling of atoms, not one big swizzle.** Apply
the XOR to the flat offset of a 256 B-row tile and it keys off the wrong bits
and reports conflicts the real layout does not have. `kind: tile` handles this;
it is the reason to prefer it over a hand-written layout.

Use `layout` for anything else — a padded staging buffer, a 1-D scale vector, a
gmem tensor:

```yaml
buffer: {kind: layout, shape: [64, 512], stride: [520, 1]}   # elements, not bytes
buffer: {kind: layout, shape: [128], stride: [1]}
```

Nested shapes work, matching CuTe: `shape: [[8, 8], 2]` with
`stride: [[1, 16], 128]`.

## tv

Either a named atom or an expression.

**`wgmma_acc`** — the fp32 accumulator fragment of `wgmma.mma_async.m64nNk*`.
One warp group holds a 64 x N tile; warp `w` owns rows `[16w, 16w+16)`, lane `l`
owns rows `16w + l//4` and `+8`, and columns `2*(l%4) + 8j + {0,1}`. It ships as
an atom because it is the most error-prone map in a Hopper kernel and it decides
three separate things: the epilogue's store pattern, the softmax reduction's
span, and which scale entry a thread needs.

```yaml
tv: {atom: wgmma_acc, n: 128, vec: 2, project: both}   # a 2-D accumulator-shaped buffer
tv: {atom: wgmma_acc, n: 128, vec: 1, project: row}    # a per-row vector: an A-scale, a running max
tv: {atom: wgmma_acc, n: 128, vec: 2, project: col}    # a per-column vector: a B-scale, a bias
```

`project: row` and `project: col` are what make a scale load checkable: they are
the same fragment map with one axis dropped, so the broadcast factor the spec
claims (`4 lanes share a row`, `8 lanes share a column pair`) is computed rather
than asserted.

**`linear`** — the plain coalesced pattern: thread `t` takes the run
`[t*vec, t*vec+vec)`, the whole CTA strides forward each instruction.

```yaml
tv: {atom: linear, cols: 128, rows: 64, vals: 8, vec: 4}
```

**`expr`** — the escape hatch, a snippet with `tid`, `vid` and `elem` bound that
assigns `coord`. Restricted namespace: no imports, no builtins beyond
`range/min/max/abs/int`.

```yaml
tv:
  expr: "coord = (tid, elem)"     # 32 lanes, one 16 B row segment each
  vals: 1
  vec: 8
```

`elem` is the element index inside one instruction and **the expression must use
it**, or the checker reports the access as non-contiguous — correctly, since a
map that returns the same coordinate for all `vec` elements has not described a
vector access.

## What comes back

```
=== smem_d store ===
  buffer   tile 64x128, swizzle none, row stride 512 B
  tv       wgmma_acc[m64n128], 128 threads x 32 inst x 2 elem
  access   st.shared.b64 64 b/thread (8 B)
  banks    wavefronts 8   ideal 2   -> 4x serialisation   FAIL
           worst inst 0, warp 0: 64 distinct words over 8 banks, broadcast 1x
  total    32 inst x 4 warps -> 1024 bank cycles (ideal 256)
```

`--markdown` emits the same thing as an L4 table row. Paste that into the spec:
the point is that the table is *generated*, so it cannot drift from the layout
it describes.

Three failures are reported besides conflicts, and each is a real lowering bug:

- **NOT CONTIGUOUS** — the `vec` elements are not adjacent after swizzling, so
  the vector width is illegal. Usually a `vec` wider than the swizzle's 16 B
  base granule.
- **MISALIGNED** — the base address is not `vec * sizeof` aligned.
- **worst inst / warp** — warps of a warp group do not always behave alike; the
  reported figures are the worst one, not warp 0's.

## The bank model

Stated explicitly because "N-way conflict" is ambiguous for wide accesses. Shared
memory serves one distinct 4 B word per bank per cycle, broadcasting to every
lane wanting that same word. For one instruction from one warp:

```
wavefronts    = max over banks of (distinct words landing in that bank)
ideal         = ceil(distinct words in the whole request / 32)
serialisation = wavefronts / ideal
```

**`wavefronts` is the number usually written as "N-way", and N-way alone is not
a verdict.** A 64-bit access moves 2 words per lane, so 2 words per bank is the
optimum; FlashMLA's stride-520 staging buffer measures 2 and is exactly optimal,
and a naive "2-way conflict" label would condemn it. DeepGEMM's epilogue
measures 8 against an ideal of 2 — a real 4x, and what `sm90.hpp`'s rule about
keeping `BLOCK_N` off multiples of 32 buys back.

For global memory the granule is a 32 B sector instead of a bank:

```
sectors = distinct (byte_addr >> 5) over the warp
ideal   = ceil(distinct bytes touched / 32)
```

## Accesses with no per-thread address

On Hopper most of the data movement has no per-thread address at all, and a
spec that invents one has described a kernel it did not write. Give those the
`unit` field and no maps:

```yaml
  - id: smem_a / smem_b fill
    unit: tma            # one elected thread issues cp.async.bulk.tensor; the copy engine writes smem
  - id: smem A/B read by wgmma
    unit: wgmma-desc     # operands come through a matrix descriptor, not ld.shared
  - id: acc -> tmem
    unit: tcgen05
```

They still get a table row. The row is the point: it says *no per-thread access*
where a reader would otherwise expect a width, which is why the producer warp
group needs 24 registers and not 240.

## expect

Any computed key can be pinned:

```yaml
    expect: {wavefronts: 8, ideal: 2, distinct_words: 64}
```

Available: `wavefronts`, `ideal`, `distinct_words`, `banks_touched`, `requests`,
`sectors`, `bytes_touched`, `instructions`, `warps`, `total_cost`,
`total_ideal`.

An `expect` block does two jobs. It turns the access into a regression test, so
a later layout change that invalidates the spec's table fails the run instead of
ageing quietly. And it lets a **deliberately bad** access be pinned without
failing the run — a counterfactual that shows what a padding or a swizzle is
worth. An access with no `expect` must pass its bank check outright.
