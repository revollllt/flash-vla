---
id: bulk-store-publish
type: technique
arch: sm90
tags: [cp-async-bulk, release-fence, partials, task-loop]
confidence: measured
---

# Publish a partial with one bulk store from a staged smem image

## Context

A task writes a partial result to global scratch and then releases a
counter or barrier. Written as thousands of small (4–16 B) generic stores,
the release fence must wait for every one of them, and the publish can cost
more than the compute that produced it.

## Move

Stage the partial as one contiguous shared-memory image — a freed ring
frame is the natural place — and publish it with a single `cp.async.bulk`
store, `wait_group 0`, and one proxy fence before the release.

## Why it works

One completion event replaces thousands: the fence cost stops scaling with
element count, and the bulk store moves the same bytes at copy-engine
efficiency.

## Caveats

Needs a frame free at publish time — see
[release-on-retirement](release-on-retirement.md) for why one is. The
proxy fence before the release is mandatory whenever a consumer may read
the partial with TMA: release/acquire alone does not order generic stores
against a following TMA read (see
[layout-production-budget](layout-production-budget.md) caveats).
