"""GR00T N1.7 routes on the RTX 5090: the plain-torch reference stages."""
from flash_vla.runtime.registry import Registry

from . import torch_ops

REGISTRY = Registry({"torch": torch_ops.BACKEND}, default="torch")
