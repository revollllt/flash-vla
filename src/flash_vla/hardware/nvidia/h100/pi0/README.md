# H100 / Pi0 target

This directory is the Pi0 execution target for NVIDIA H100: the model
(`flash_vla.models.pi0`) composed with this device's backends, fusion
boundaries and plans, and the workload-specific kernels.

The measured default profile is H100 SXM5, BF16, three views, empty prompt,
action chunk 50, ten diffusion steps, and 18 action-expert layers. Other shapes
may compile, but must be retuned and revalidated before being advertised as an
optimized profile.
