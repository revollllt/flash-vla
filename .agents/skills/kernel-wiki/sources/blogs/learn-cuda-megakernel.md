---
id: blog-learn-cuda-megakernel
title: "learn-cuda 12_megakernel: a flag-barrier megakernel"
author: Thien Tran (gau-nernst)
url: https://github.com/gau-nernst/learn-cuda
source_category: community-note
architectures: [sm90]
tags: [megakernel, flag-barrier, decode, fused-kernel, cache-hint]
retrieved_at: 2026-09-07
---

# learn-cuda 12_megakernel

The `12_megakernel` chapter of gau-nernst/learn-cuda builds the minimal
megakernel: one launch whose phases are split by a global-memory flag that
the last CTA resets, with redundant norms in place of a dependency hop and
weight loads carrying cache-policy hints. The repository had no license file
at the revision read (2026-09-07), so the wiki's template is distilled from
it rather than excerpted.

## Source-reported results

The chapter reports an MLP at about 1622 GB/s and a Triton attention v2 at
29.8 / 29.1 us on H200. These are that author's machine and shapes; the sm90
template's STATUS block measures the same forms on H100.
