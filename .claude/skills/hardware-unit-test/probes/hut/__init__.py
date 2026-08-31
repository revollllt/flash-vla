"""Shared harness for the hardware unit tests.

`abi.py` mirrors `include/hut/unit.hpp`; `harness.py` owns build, timing and
repetition; `regime.py` owns the cold/L2 distinction. A unit contributes only
its sweeps.
"""
