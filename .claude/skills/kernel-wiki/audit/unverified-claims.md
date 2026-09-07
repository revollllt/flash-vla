# Unverified and disputed claims

## Open material claims

- **Structural template header rules.** The eighteen `structural` templates
  of `doc-kernel-design-templates` (`01`-`04`, `10`-`14`, `20`-`23`,
  `30`-`33`, `41`) compile and have their PTX asserted but have never been
  run; the design rules in their headers ("this halves A traffic", "why
  BLOCK_K is 128") are design statements. Wiki pages quote them only as
  mechanism, never as measurement. Closing this needs the shared host
  harness the template grades note deferred, then a STATUS block per
  archetype.
- **`technique-pdl-placement` stays `source-reported`.** The trigger sweep
  it prescribes has been run on some chains (the STATUS-carrying templates
  declare their positions) but not recorded as a constant; the CUTLASS
  quotation and the programming guide carry the page.
- **Upstream numbers on copied kernel pages** (`kernel-deepgemm`,
  `kernel-flashmla`, `kernel-sparse-mla`, `technique-pipeline-stages`) are
  KernelWiki's source-reported rows and are not re-measured here.

## Resolved

- The old wiki's "no content is copied from KernelWiki" statement no longer
  holds and was removed; provenance is recorded in `README.md` and the
  workflow note.
