# Optional result outline

A short experiment entry is usually enough. For a larger investigation, use
this outline and include only sections supported by the capture:

```markdown
# Attention kernel investigation

Question: Is the producer limiting tensor-core activity?
Conditions: GPU, workload/shape, build, NCU version, selected launch and
replay/cache/clock settings. Link the capture command and native report.

Finding: State the observed metrics and what they imply. Separate the measured
fact from its likely explanation; list missing evidence that affects the choice.

Next experiment: One change that tests the explanation, with expected impact.
Validation: Link the separate correctness and uninstrumented timing result when
available. A counter-based estimate alone is not a speedup measurement.
```

Readability matters more than filling a template. Additional plots or exports
are useful only when they answer the question or make the finding easier to assess.
