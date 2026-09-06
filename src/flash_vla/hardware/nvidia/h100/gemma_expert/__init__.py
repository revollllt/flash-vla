"""The Gemma action expert on H100: kernels and backend factories, written once.

A device component package (`ARCHITECTURE.md`, "Dependency direction"): it
holds the kernels and the `make_wrappers` factories of one model component on
one device, imports `models/`, `runtime/` and the shared tile and TileLang
libraries, and imports no Target. Pi0.5 and Pi0 register its backends under
names of their own and keep every routing decision, their plans and their
route constraints.

The expert is 18 layers of width 1024 with a 4096 FFN, 8 query heads over 1 KV
head at head dim 256, on both Targets. What differs between them is the row
count, the prefix length and whether the norms are adaptive; the first two are
the CUDA build's compile-time profile and the third is carried by the
per-(step, layer) constant vectors the wrappers read.
"""
