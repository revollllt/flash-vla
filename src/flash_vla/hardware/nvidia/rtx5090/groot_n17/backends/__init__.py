from flash_vla.runtime.registry import Registry
from . import torch_ops

REGISTRY = Registry({"torch": torch_ops}, default="torch")
