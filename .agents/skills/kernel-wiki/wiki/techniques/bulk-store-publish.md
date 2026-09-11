---
id: technique-bulk-store-publish
title: "Publish a partial with one bulk store from a staged shared-memory image"
type: technique
architectures: [sm90]
tags: [cp-async-bulk, bulk-store, proxy-fence, task-loop, tma]
confidence: measured
reproducibility: snippet
prerequisites: [hw-tma, hw-mbarrier]
related: [technique-release-on-retirement, pattern-layout-production-cost, pattern-epilogue-bound-short-k, technique-epilogue-fusion]
sources: [doc-kernel-design-templates, doc-ptx-isa-sm90, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
  - evidence_type: upstream-code
    source_id: doc-kernel-design-templates
---

# Publish a partial with one bulk store from a staged shared-memory image

A task that writes a partial result to global scratch and then releases a
counter or barrier pays for every store the release fence has to wait on.
Written as thousands of small (4-16 B) generic stores, the publish can cost
more than the compute that produced it. Stage the partial as one contiguous
shared-memory image, a freed ring frame is the natural place, and publish it
with a single `cp.async.bulk` store, `wait_group 0`, and one proxy fence
before the release. One completion event replaces thousands: the fence cost
stops scaling with element count, and the bulk store moves the same bytes at
copy-engine efficiency.

## Source-backed fragment

This contiguous excerpt is from `04_epilogue_persistent.cu` in
`doc-kernel-design-templates`. It shows the two steps a staged store needs
before the issuing lane runs, then the store, commit and read-wait:

```cpp
    tmpl::fence_proxy_async_shared();
    __syncthreads();

    if (tid == 0) {
      tmpl::tma_store_2d(&out_map, stage[frame], task.col_block * kTileN,
                         task.row_block * kTileM);
      tmpl::tma_store_commit();
      // Keep one store in flight; this only claims the OTHER frame is free.
      tmpl::tma_store_wait<1>();
    }
```

## Design checks

- Both steps before the store are needed: the proxy fence orders this
  thread's generic-proxy shared writes before an async-proxy read, and the
  barrier makes every thread's writes visible before the single issuing lane
  runs. Neither does the other's job.
- `cp.async.bulk.wait_group.read` only says the shared frame is reusable; the
  global write may still be landing. A release that a consumer will read with
  TMA needs `fence.proxy.async.global` after the write-completion wait, not
  just release/acquire on the counter (see `pattern-layout-production-cost`).
- Needs a frame free at publish time; `technique-release-on-retirement` is
  why one is.
- Drain outstanding bulk groups before the kernel ends: the launch boundary
  alone does not order them for the next kernel.
