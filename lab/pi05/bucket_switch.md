# Runtime backbone bucket switch check

One RTX5090 Pi0.5 engine is built from the short oracle configuration. It runs
short896 -> long903 -> short896 through set_task and forward, comparing each
forward with its corresponding existing official oracle. Fixture files load on
CPU, and images/noise round to BF16 on the device exactly as in
eval.pi05.parity.compare. No seeded fixture is regenerated.

This is full-depth parity, not a latency measurement. It reuses error_metrics,
tolerances, and to_pair_layout; reports every layer's valid prefix K/V,
actions, finite padded K/V, exact staged token IDs, and the runtime valid count.
Acceptance uses the existing parity formulas for layer0/deepest/step/actions.
The two fixture directories are inputs; their state/image/noise equivalence is
checked separately by the experiment owner.

The initial captured Program, forward Step tuple, each StreamGraph, and
each underlying CUDAGraph are retained and compared by object identity after
each forward. ModelRunner.forward stages inputs and calls host/replay;
Program.replay only calls the existing StreamGraph.replay. set_task changes
the tokenizer's task string. None of these paths calls capture, and this
harness neither replaces a graph nor invokes capture. No recapture monkeypatch
or new runtime hook is installed.

Run from the checkout containing the bucket candidate, with its usual CUDA
environment and PYTHONPATH, and CHECKPOINT set to the converted belt-cup weights:

    python -m lab.pi05.bucket_switch \
      --short-oracle artifacts/rtx5090-pi05/oracle-belt-cup-short896 \
      --long-oracle artifacts/rtx5090-pi05/oracle-belt-cup \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
      --plan shipped --output artifacts/rtx5090-pi05/bucket-switch.json

Prepared CPU-only: python3 -m py_compile lab/pi05/bucket_switch.py.
No Torch import, model load, CUDA compilation, or GPU execution was performed
during harness preparation. Numerical mismatches are written to JSON with exit
code 1; loading/build/runtime errors propagate with their original traceback.
