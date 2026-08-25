# Worked example — one stage at all four levels

A synthetic scaled FP8 GEMM, written out at L1–L4 so the notation and the
level split have a concrete instance. SKILL.md's notation and "L4 is computed"
sections point here; read it when writing a nest for the first time, or when a
level boundary feels ambiguous. For real reverse-engineered kernels see
`example-deepgemm.md` (producer/consumer) and `example-flashmla.md` (cooperating
math groups).

## Write the form the hardware computes

```
Oᵀ[dv, B_H] = Vᵀ[dv, kv] @ Pᵀ[kv, B_H]      and      O[B_H, dv] = P[B_H, kv] @ V[kv, dv]
```

are the same math and two different kernels. They assign different tensors to
the MMA's A and B operands, imply different smem layouts, and need different
`Major::` flags. Which one you write *is* the decision — leaving it implicit is
the bug. So transposes are never elided for tidiness: `ᵀ` on an operand is a
claim about how that tensor sits in shared memory.

## Shape

A scaled FP8 GEMM, complete at all four levels:

```
L1 ------------------------------------------------------------ iteration space
  gemm(A[M,K] e4m3 k-major, B[N,K] e4m3 k-major,
       SFA[M, K/128] f32, SFB[N, K/128] f32) -> D[M,N] f32

  for m0 in range(0, M, 128):                 # ceil(M/128) tiles      parallel
    for n0 in range(0, N, 128):               # ceil(N/128) tiles      parallel
      for k0 in range(0, K, 128):             # K/128 steps            SERIAL, contraction
        D[m0:m0+128, n0:n0+128] += SFA[m0:m0+128, k0//128] * SFB[n0:n0+128, k0//128]
                                   * ( A[m0:m0+128, k0:k0+128] @ Bᵀ[k0:k0+128, n0:n0+128] )
                                       (128,128) @ (128,128) -> (128,128)

L2 ------------------------------------------------------- mapped to hardware
  grid 132 persistent CTAs: (m0,n0) distributed by the scheduler, k0 stays serial per CTA
  384 threads = 1 producer WG + 2 math WGs; math WG w owns rows [64w, 64w+64)

  for (m0, n0) in scheduler(cta_id):          # ~ceil(M/128)*ceil(N/128)/132 tiles per CTA
    D_acc[64, 128] = 0                        # f32 RF, 64 elems/thread, carried across k0

    for k0 in range(0, K, 128):               # mainloop: start 0, stop K, step 128, trip K/128
      s, phase = (k0//128) % 4, (k0//128)//4 & 1        # stage, depth 4

      producer  wait empty[s] @ phase^1
                A_s[s][128,128] <- A[m0:m0+128, k0:k0+128]    16384 B  TMA
                B_s[s][128,128] <- B[n0:n0+128, k0:k0+128]    16384 B  TMA
                sfa_s[s][128]   <- SFA[m0:m0+128, k0//128]      512 B
                sfb_s[s][128]   <- SFB[n0:n0+128, k0//128]      512 B
                full[s].arrive_and_expect_tx(33792)

      math WG w wait full[s] @ phase
                C_s[64,128] = A_s[s][64w:64w+64, 0:128] @ B_s[s]ᵀ[0:128, 0:128]
                                (64,128) @ (128,128) -> (64,128)   f32 RF, stage-local
                empty[s].arrive()                                   release before promoting
                D_acc += sfa_s[s][64w:64w+64] * sfb_s[s][0:128] * C_s     64 CUDA-core FMAs

    epilogue  D[m0+64w : m0+64w+64, n0:n0+128] <- D_acc

L3 ------------------------------------------------ schedule, one stage
  engine timeline for stage s (three columns = three engines, top to bottom = time)

    copy engine (TMA)         CUDA cores (LSU/ALU)      tensor cores (WGMMA)
    ------------------------- ------------------------- ----------------------
 t0 issue A_s[s'],B_s[s']     WG0 promote(s-1)          WG1 mma k-blk 0..3 of s-1
    elected thread, 1 inst    64 f32 FMA/thread
    s' = most recently released buffer, NOT s+1: in a depth-d pipeline the
    producer floats 1..d-1 stages ahead and the distance drifts with the balance
 t1 in flight ~700 ns [I]     WG0 wait full[s] @ phase  WG1  "
 t2 full[s].arrive_and_       WG1 wait<0> returns;      WG0 mma k-blk 0..3 of s
    expect_tx(33792)          empty[s-1].arrive();
                              promote(s-1)
 t3 wait empty[s'+1] @ phase  WG0 wait<0> returns;      WG0  "
                              empty[s].arrive();
                              promote(s)

  Columns are per WARP GROUP on the math side, and that is load-bearing: within
  ONE warp group `wgmma.wait_group<0>` is a full barrier on the batch, so its
  promote sits strictly BETWEEN two batches and the tensor cores idle for the
  promote's whole duration. The overlap above is real only because the promote
  belongs to the *other* group. Nothing orders WG0 against WG1; the tensor cores
  serialising their two batches is the only thing that staggers them.

  ORDERING EDGES, and which are real
    the TMA issue sits ABOVE the promote, not below: it depends on empty[s']
      only, never on stage s's compute. Put it after any CUDA-core work and the
      copy engine idles for exactly that work's duration.
    empty[s].arrive() likewise sits ABOVE the promote -- the release is what the
      copy engine waits on. Safe only because the scales were pulled into
      registers before warpgroup_arrive.
    the ONE true serialisation is full[s] -- bytes must land before wgmma reads.

  BUBBLE CHECK   (cycles are [I]; the criterion is the RATIO, not the absolutes)
    copy engine idle    the t3 wait on empty[s'+1] only, and only when the copy
                        column runs UNDER the math column. Balanced here, so the
                        residual wait lands on the math side as full[s] instead.
    tensor cores idle   prologue (stages 0..depth-1). In steady state the two
                        groups' batches tile the stage back to back.
    CUDA cores idle     ~50%: one promote per group inside a two-batch stage.
                        Fine -- each group's promote is covered by the other's
                        batch, which is exactly what the column labels show.

L4 ------------------------------------- instructions and threads, one stage
      for ki in range(0, 128, 32):            # iter: start 0, stop BLOCK_K=128, step 32, trip 4
        wgmma.m64n128k32(
          A = A_s[s][64w:64w+64, ki:ki+32]    smem-desc, k-major
          B = B_s[s][ki:ki+32, 0:128]ᵀ        smem-desc, k-major
          C = C_s[64, 128]                    f32 RF, 64*128/128 = 64 elems/thread
          clear = (ki == 0) )                 # ScaleOut::Zero on iter 0, accumulate on 1..3

  PER-THREAD ACCESS, every gmem/smem touch in the stage. GENERATED, verbatim,
  by `tv_check.py <access-file> --markdown`; see references/l4-access.md.

  | touch         | width       | count             | banks / sectors                |
  |---------------|-------------|-------------------|--------------------------------|
  | A_s, B_s fill | n/a         | n/a               | NO per-thread access -- one elected thread issues the whole tile; the copy engine writes smem |
  | smem A/B read | n/a         | n/a               | NO per-thread access -- operands come through a matrix descriptor, not ld.shared |
  | sfa load      | 32 b/thread | 2 inst x 4 warps  | 1-wavefront, ideal 1 -> 1x     |
  | sfb load      | 64 b/thread | 16 inst x 4 warps | 1-wavefront, ideal 1 -> 1x     |
  | D store       | 64 b/thread | 32 inst x 4 warps | 8 sectors, ideal 8 -> 1x       |

    addressing  tile base computed once in the prologue, carried in a register;
                the stage index is the only per-iteration arithmetic (1 IADD)
```

## L4 is computed, not asserted

Every number in that table — width, vector, transactions, conflict ways — is a
function of exactly two maps: the buffer's layout (with its swizzle) and the
access's thread-value map. Write those two down and the rest is arithmetic over
32 lanes, which is why this level does not get written in prose.

The failure prose invites is specific. `128 B swizzle atom, aligned -> 0-way`
is a *conclusion*: it cannot be rechecked, cannot be regenerated when L2 changes
a tile extent, and cannot be wrong out loud. Both worked examples marked nearly
every conflict count `[I]`, and `example-deepgemm.md` had to open an
`open_questions.bank_ways` entry because nobody could settle one by hand. They
are all `[D]` now. So:

```
python3 scripts/tv_check.py <access-file> --markdown    # the table above
```

The composition is also not something to do in your head: **a swizzle is an XOR,
not an affine map**, and a tile wider than its atom is a *tiling* of atoms rather
than one big swizzle — apply the XOR to the flat offset of a 256 B-row tile and
it keys off the wrong bits and reports conflicts the real layout does not have.

Two consequences worth carrying into the spec:

- **"N-way conflict" is not a verdict on its own.** A 64-bit access moves two
  words per lane, so two words per bank is the *optimum*: FlashMLA's stride-520
  staging buffer measures 2 and is exactly optimal, while DeepGEMM's epilogue
  measures 8 against an ideal of 2 and is a real 4x. Report against the ideal.
- **The unit is not always a thread.** On Hopper most data movement has no
  per-thread address at all — TMA moves a tile from one elected lane, wgmma reads
  smem through a descriptor. Those rows say *no per-thread access*, and saying it
  is the point: a spec that invents a per-thread pattern for a TMA load has
  described a kernel it did not write. A per-thread affine model imported whole
  from a scalar-load ISA gets this exactly backwards.

The `=` on `C_s` beside the `+=` on `D_acc` is the whole design in two lines: the
tensor-core accumulator is stage-local because the scales change every K block,
so a second fp32 accumulator carries the mainloop. Every bound at L4 traces to
L2 and every name at L2 traces to L1 — a number that cannot be traced upward
means a field is missing.
