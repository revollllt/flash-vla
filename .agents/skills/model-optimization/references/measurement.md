# Measurement conditions

What must hold for two versions to be comparable at all. The workflow's step 2
runs the command; this is the contract it runs under.

- **Same everything but the change**: physical GPU, driver, runtime, checkpoint,
  input, shape, precision, and timing scope.
- **Own process per version**, first capture, one graph per stream.
- **5 warmup and 100 measured iterations**, compared on the median, raw samples
  kept.
- **Timing covers** input transfer, host work, replay and the final sync. It
  excludes load and capture.
- **Measure A and B separately.** Re-measure a version only on evidence of
  drift, not to refresh a number.
- **Reuse is allowed** for a measurement from this round taken on the same code
  under these conditions, and for existing references and tolerances.
- **Profiling and timing are separate processes.** Profile with the benchmark's
  inputs and execution config; measure deployed latency with no profiler
  attached.
- **GPU time is the reading.** A CPU segment label marks submission range, not
  device work.

Changed conditions are a different experiment: they start a separate record and
are never joined into one speedup curve.
