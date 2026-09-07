# Evidence ledger

Access date: 2026-09-07. The authoritative sources behind this repository's
sm90 pages; KernelWiki's own ledger for the copied corpus is at
https://github.com/mit-han-lab/KernelWiki/blob/main/audit/evidence-ledger.md
(comparison base `52b8eb06`, evidence access 2026-08-18 UTC).

| ID | Authoritative source | Version/date and locator | Claims supported |
| --- | --- | --- | --- |
| E001 | NVIDIA PTX ISA, `https://docs.nvidia.com/cuda/parallel-thread-execution/` | rolling document accessed 2026-09-07; section titles named in `doc-ptx-isa-sm90` | wgmma batches and operand forms, bulk tensor copies and multicast, mbarrier accounting, proxy fences, cluster/DSMEM instructions, `griddepcontrol`, `setmaxnreg`, cache-hint spellings |
| E002 | CUDA Programming Guide, PDL | KernelWiki's `doc-cuda-programming-guide-pdl`, accessed 2026-08-18; §4.5 | the PDL protocol and the CC 9.0 floor |
| E003 | `hardware-unit-test` constants table | `.claude/skills/hardware-unit-test/`; each tag carries machine, toolchain, date and sweep | every bracketed machine constant on an sm90 page (42 tags at access) |
| E004 | `kernel-design` templates, reference grade | `40`, `42`, `43`, `44`, `45` STATUS blocks: H100 SXM5, CUDA 13.0/13.1, clocks not pinned | every `performance_claims` row on `kernel-megakernel-forms`; snippets on technique and hardware pages are contiguous excerpts named by file |
| E005 | Agent Notes | `.agents/notes/implemented/performance/2026-09-06-optimization-campaign-plan.md` and the notes each page names | the `measured` confidence of every migrated page: jobs, revisions and rejections live there |
| E006 | FlashAttention-3 paper, `https://arxiv.org/abs/2407.08608` | abstract, accessed 2026-09-07 | 740 TFLOPS FP16 (75%), about 1.2 PFLOPS FP8 on H100, as sweep maxima |
| E007 | DeepGEMM and FlashMLA READMEs at the `third_party/` submodule commits | `559d79fb` and `15f13e50`, the same commits KernelWiki's `blog-deepgemm` and `blog-flashmla` pin | the sm90 choreography sections on `kernel-deepgemm` and `kernel-flashmla` |
| E008 | HazyResearch Megakernels blog and repository; Mirage MPK paper `https://arxiv.org/abs/2512.22219`; gau-nernst/learn-cuda | accessed 2026-09-07; upstream numbers kept as upstream's beside the STATUS rows | the four megakernel forms and their design rules |
| E009 | `ncu-report` sm90 metric vocabulary | `.claude/skills/ncu-report/references/08-sm90-metric-names.md`, Nsight Compute 2025.4.1 | the `symptoms` vocabulary and the stall-reason names pattern pages open with |
| E010 | `benchmark-kernel` | `.claude/skills/benchmark-kernel/SKILL.md` | the measurement rules on `technique-same-process-aba` |

## Terminal gate receipts

- `scripts/validate.py`: 0 errors on the full corpus at 2026-09-07 (counts
  from `scripts/repo_status.py`).
- `scripts/scan_numbers.py --sm90`: every unit-bearing number on an sm90 page
  is dispositioned in `numeric-claims-ledger.md`.
- `tests/`: 8 contract tests and the PyYAML-fallback check pass.
