# Agent Note: BK256 wins only with a fused internal activation producer

Status: upper bound established; standalone production path rejected

## Question

TileLang uses BK256/depth3 for the Pi0.5 GatedUp projection. The production
CuTe kernel used BK64/depth4 because its row-major activation can be copied by
one legal 8 KB SW128 TMA box. This experiment separated three effects:

1. reducing packed-weight stages from 16 to 4;
2. reducing activation TMA issues with an internal `[K, M_PAD]` layout; and
3. paying the cost to produce that layout and apply F/S scaling.

DownResidual remained BK64 throughout. Using one global BK256 constant is
invalid because it changes its TMA byte contract and dependency geometry.

## Row-major BK256

The row-major candidate used depth3. Each stage issued four legal 8 KB
activation boxes, one 32 KB packed gate/up weight box, and one 512-byte scale
copy. Four activation boxes shared one transaction-barrier arrival expecting
32 KB.

H100 job 551954:

| Path | BK64 production | Row-major BK256 |
| --- | ---: | ---: |
| GatedUp | 15.64 us | 17.01 us |
| fused | 25.25 us | 26.80 us |

Reducing weight stages alone does not pay for 198272 bytes of shared memory,
depth3, and the larger per-stage transaction/compute batch. Row-major BK256 is
rejected.

## Internal-layout kernel upper bound

The upper-bound input is a BF16-scaled, contiguous `[K, M_PAD]` tensor. M is
the 128-byte TMA row, so each BK256 stage uses one 32 KB activation TMA and one
32 KB packed-weight TMA. GatedUp uses MN-major A and MN-major B WGMMA
descriptors. One stage contains sixteen N64 WGMMA instructions; ptxas and H100
accepted one commit per stage with `wait_group<1>`.

H100 job 552091, with layout production outside the timed graph:

| Path | BK64 production | BK256 upper bound |
| --- | ---: | ---: |
| GatedUp | 15.64 us | 14.28 us |
| fused | 25.25 us | 23.80 us |
| TileLang composition | 22.84 us | 22.94 us |

The candidate used 105 registers, zero stack/local memory, and 196736 bytes of
dynamic shared memory. GatedUp parity passed with cosine 1.0.

Bring-up exposed one reusable CuTe rule. A composed
`Layout_MN_SW128_Atom` carrying `smem_ptr_flag` must not be called directly and
treated as a BF16 physical offset for generic shared-memory loads/stores. The
correct MN-major 128-byte TMA mapping is:

```cpp
physical = k * M_PAD + (((m >> 3) ^ (k & 7)) << 3) + (m & 7);
```

Using the direct composed-layout return value swapped the padded M chunk with
valid rows 48-49 during the in-place scale pass. Jobs 552049, 552053, and
552074 localized this issue; it was not a TMA or WGMMA failure.

## Layout-production price

Job 552134 added an optimized 32x32 shared-memory scale-transpose kernel to
the captured graph before the persistent FFN launch. It preserved cosine 1.0
but measured:

| Path | Pre-scaled upper bound | Separate producer + BK256 |
| --- | ---: | ---: |
| GatedUp | 14.28 us | 17.28 us |
| fused | 23.80 us | 26.93 us |

The separate producer costs about 3.13 us and consumes more than the 1.45 us
kernel-side fused gain. A second launch is also incompatible with the required
single persistent megakernel.

## Decision

Do not merge either BK256 experimental kernel into production. Preserve the
BK256 internal-layout result as a target profile for the next persistent
planner/dataflow phase.

The next implementation must generate the scaled `[K, M_PAD]` activation
inside the single persistent launch and reuse it across multiple output tiles.
The static CTA plan must guarantee producer residency and dependency progress;
a runtime work queue or a standalone transpose launch is not allowed. If that
integration cannot keep its added critical-path cost below about 1.45 us, the
production kernel remains BK64/depth4.
