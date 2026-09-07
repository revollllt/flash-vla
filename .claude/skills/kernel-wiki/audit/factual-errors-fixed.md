# Factual errors fixed

Corrections made while migrating this repository's 25 wiki entries into
KernelWiki's schema on 2026-09-07. Families, not sentences.

| # | Correction family | Problem and corrected state | Evidence |
| ---: | --- | --- | --- |
| 1 | Unbacked `measured` | 21 entries declared `confidence: measured` with no source, note or tag on the page. Every migrated page now carries `sources` and, where `measured`, an `evidence_basis` naming the Agent Note, the constants table or the template STATUS block that established it; the validator enforces it. | E003, E004, E005 |
| 2 | Unroutable symptoms | The routing table was hand-written prose; the profiler's vocabulary and the wiki's were unrelated. Pattern pages now carry `symptoms` from a controlled list spelled after the sm90 stall reasons and rules, and the by-problem index is generated from them. | E009 |
| 3 | Moves without code | Nine technique entries had no snippet. Each now carries a contiguous excerpt of a template or of `sm90_common.cuh`, named by file, so a reader can see the instruction and the checker can see the file. | E004 |
| 4 | Upstream references without a pin | Four `ext-*` entries cited a repository URL. DeepGEMM and FlashMLA now cite the README at the `third_party/` submodule commits (the same commits KernelWiki pins); FA3, HazyResearch, MPK and learn-cuda have source pages with dates and locators; the megakernel page's numbers are `performance_claims` rows citing STATUS blocks. | E006, E007, E008 |
| 5 | Numbers in prose | The 28 unit-bearing numbers on sm90 pages are dispositioned in the numeric ledger; measured kernel numbers moved into `performance_claims` records with gpu, dtype, shape, metric, value and locator. | numeric ledger |
| 6 | Dangling cross-references | Copied KernelWiki pages referenced pages and bundles outside the first copy set. The full wiki and the closure of its sources and bundles were copied, and the validator now fails an unresolved `related`, `prerequisites`, `candidate_techniques` id or relative link. | validator |

No claim was found to be wrong on its merits during the migration; the
corrections are of evidence and structure. Header rules of the eighteen
structural templates were not re-verified by a run and are listed as open in
`unverified-claims.md`.
