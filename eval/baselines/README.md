# Baselines

Official implementation adapters belong here. They should accept the canonical
model inputs and expose final outputs plus optional named stage tensors without
becoming a dependency of a production target.

## OpenPI PyTorch parity

Install OpenPI with its official PyTorch setup, then run the H100/Pi0 comparison
against an official PyTorch `model.safetensors` checkpoint:

```bash
python -m eval.correctness.pi0.openpi_parity \
  --checkpoint /path/to/model.safetensors
```

The command gives both implementations identical images, state, and diffusion
noise, then reports max/mean/RMS/P99 absolute error and cosine similarity.
