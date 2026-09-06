"""The Gemma language backbone on H100, shared by every Target that runs it.

A device-level component package
(`.agents/notes/implemented/architecture/2026-09-07-device-component-packages.md`):
it holds the kernels, the reference mirrors and the backend factories of the
`llm_backbone` segment on this device, written once. Both H100 Targets run the
same 18-layer, 2048-wide transformer over the image-and-prompt prefix at the
same head geometry (8 query heads, one KV head, head_dim 256); only the prefix
row count differs, and it is a runtime argument everywhere here.

This package imports `models/`, `runtime/` and the shared SM90 tile library,
and never a Target. A Target registers a backend from here under a name of its
own and keeps every routing decision, its plans and its reference route.
"""
