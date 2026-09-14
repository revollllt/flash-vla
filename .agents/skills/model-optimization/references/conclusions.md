# Reviewing a conclusion

Six checks, answered before a round is reported. They test a *conclusion*, not a
process, and each is answerable from numbers already written down. The cases
behind them are in [corrections](../../../../docs/corrections/README.md).

- **A compared number comes from this round, or states its provenance.** A row
  in a summary table is not the current baseline.
- **A ratio names its denominator.** An impossible ratio is a bug in the
  denominator before it is a finding: a measurement above "peak" means the peak
  was computed wrong -- wrong precision tier, wrong clock, wrong unit.
- **An estimate names the memory tier it assumes**, against this machine's
  measured constants. A workspace that fits L2 cannot be priced at DRAM
  bandwidth.
- **A negative carries the toolbox it was measured with, and expires when that
  changes.** If a technique later measured to be worth a factor was not in it,
  the conclusion is deferred, not settled, until it is re-run.
- **"Unavailable" needs the boundary probed, not one error message.** Probe with
  [hardware-unit-test](../../hardware-unit-test/SKILL.md); an implementation
  such as `ptxas` outranks a second-hand document.
- **A blocker named precisely is the next target, not a property of the world.**
  "It holds once X changes" has to come with the cost of changing X, or it parks
  a proven technique inside your own conclusion.
