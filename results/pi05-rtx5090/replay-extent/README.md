# Pi0.5 / RTX 5090: the prompt's valid length at replay time

The prompt's valid length changes with every inference. The backbone is
captured once per 64-row bucket of the rows a workload's prompts reach, and
the host picks the bucket after tokenizing (`runtime/replay.py`). Attentions
that can run the exact length read it through `Scratch.on_replay`. Each
bucket's GEMMs run the tile that is fastest at its rows.

| Workload | Valid prefix rows | Buckets (rows) |
|---|---|---|
| `robodojo` | 811–869 | 832, 896, 968 (full) |
| `libero` | 518–534 | 576, 712 (full) |

## Latency

Each step is an ABBA against the commit before it: fresh processes, ordered
A, B, B, A, with 100 chunks each, on the fixture observations (`abba/`).
Deltas are in median chunk latency.

| Step | Commit | bf16 robodojo | bf16 libero | mxfp8 robodojo | mxfp8 libero |
|---|---|---:|---:|---:|---:|
| N: one graph per bucket, masked kernels removed | `6898390` | −0.28 ms | −0.40 ms | −0.33 ms | −0.49 ms |
| A: split-KV expert and head-72 vision attention | `3c8ad2a` | −0.36 ms | −1.17 ms | not run | not run |
| T: per-bucket GEMM tiles | `7735c9f` | −0.06 ms | −1.22 ms | −0.02 ms | −0.19 ms |

End to end, from the first A leg (`4addd53`) to the last B leg (`7735c9f`).
These legs ran in different sessions, so treat the rows as approximate:

| Recipe | robodojo | libero |
|---|---:|---:|
| bf16 | 30.28 → 29.46 ms (−2.7%) | 24.35 → 21.61 ms (−11.2%) |
| mxfp8-llm-ffn | 21.86 → 21.00 ms (−3.9%) | 18.05 → 16.23 ms (−10.1%) |

## Screens

- `v0/bucket-granularity.json`: 64-row against 128-row buckets for the BF16
  backbone GEMMs. At 64 rows, libero runs 576 rows instead of 640, about
  42 µs per layer less.
- `v0/attention-screen.json`, `a/attention-screen.json`: every attention
  against FlashInfer and our kernels (`lab/pi05/attention_screen.py`).
  - **Backbone:** FlashInfer at the valid rows runs 40.6–44.6 µs/layer against
    48.6–53.5 for compiled torch at the bucket's rows on robodojo, and is
    level on libero.
    - End to end, FlashInfer measured −0.06 ms on robodojo and −0.01 ms on
      libero (`abba/a1-vs-compiled-bucket-bf16`), so it is not routed.
    - A Triton flash kernel at head width 256 loses to compiled torch.
  - **Expert:** split-KV runs 10.3 µs against 12.3 on robodojo and 5.5 against
    9.7 on libero. FlashInfer paged is 13.6 / 13.0 µs.
  - **Vision:** our kernel runs 12.9 µs/layer against 16.9 on robodojo and 7.8
    against 16.5 on libero. FlashInfer has no head width 72.
- `a/attention-tile-sweep.json`: tiles of both kernels at each workload. No
  tile spills.
- `v0/mxfp8-gemm-screen.json`: the MXFP8 down GEMM tile at every bucket's rows.
- `t/bucket-gemm-screen.json`: cuBLAS and every stream-K family tile for each
  backbone GEMM at every bucket's rows, checked against torch. Entries that
  beat the deployed candidate by more than 5% were adopted; a rerun after
  adoption finds none left.
