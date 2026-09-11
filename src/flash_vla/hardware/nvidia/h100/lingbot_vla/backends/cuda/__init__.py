"""Hand-written CUDA kernels for this Target's fixed shapes."""
from . import pointwise

rope_project = pointwise.rope_project

__all__ = ["rope_project"]
