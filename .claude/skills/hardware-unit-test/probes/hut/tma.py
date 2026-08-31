"""TMA descriptor geometry and the walk, shared by every unit that drives TMA.

This lives in `hut/` and not in one unit because `overlap` and `pipeline_ws`
need it too. Before, `overlap.py` reached into `tma_ring.py` through a sys.path
insert -- one probe's driver importing another probe's driver.

Names are the driver API's: cuTensorMapEncodeTiled takes globalDim,
globalStrides, boxDim, elementStrides, swizzle. [references/vocabulary.md]
"""

from __future__ import annotations

import ctypes

# CUtensorMapSwizzle
SW_NONE, SW_32B, SW_64B, SW_128B = 0, 1, 2, 3
SW_NAME = {SW_NONE: "none", SW_32B: "32B", SW_64B: "64B", SW_128B: "128B"}

# CUtensorMapDataType, with the ELEMENT WIDTH the host believes each one has.
# TMA has no fp8 type: an fp8 tensor is encoded as UINT8 and the tensor core
# does the interpreting, so the copy engine's axis is element WIDTH, not numeric
# format. The 4-bit entries are Blackwell's; sm90 rejects them. [tma.bytes.txn.dtype]
DTYPES = [
    ("uint8   (fp8)", 0, 1),
    ("float16", 6, 2),
    ("bfloat16", 9, 2),
    ("float32", 7, 4),
    ("tfloat32", 11, 4),
    ("16u4_a8  (fp4)", 13, 1),
    ("16u4_a16 (fp4)", 14, 1),
]
DT_BF16 = 9


def _floor_pow2(n: int) -> int:
    return 1 << (max(1, n).bit_length() - 1)


class Geom:
    """One descriptor plus the coordinate walk that keeps every box in bounds.

    `global_dim_0` is the fastest-varying extent in ELEMENTS, so a packed row is
    `global_dim_0 * elem_bytes` and that is also the stride between consecutive
    box rows. `global_dim_0 == box_dim_0` therefore means the box rows are
    ADJACENT and the whole box is one contiguous run; anything larger leaves
    128 B strips at that stride. That distinction is what the geometry axis
    measures. [tma.bw.cta.geom]
    """

    def __init__(self, name, global_dim_0, box_dim_0=64, swizzle=SW_128B,
                 tensor_data_type=DT_BF16, elem_bytes=2, buf_bytes=None):
        self.name = name
        self.global_dim_0 = global_dim_0
        self.box_dim_0 = box_dim_0
        self.swizzle = swizzle
        self.tensor_data_type = tensor_data_type
        self.elem_bytes = elem_bytes
        self.buf_bytes = buf_bytes
        self.row_bytes = global_dim_0 * elem_bytes
        self.strip_bytes = box_dim_0 * elem_bytes

    def box_bytes(self, box_dim_1: int) -> int:
        return self.strip_bytes * box_dim_1

    def plan(self, box_dim_1: int, buf_bytes: int) -> dict:
        """Descriptor dims plus the walk's shift-and-mask encoding."""
        global_dim_1 = buf_bytes // self.row_bytes
        n_coord_0 = self.global_dim_0 // self.box_dim_0
        n_coord_1 = min(_floor_pow2(global_dim_1 // box_dim_1),
                        _floor_pow2(max(1, global_dim_1 // box_dim_1)))
        assert n_coord_0 & (n_coord_0 - 1) == 0, f"{n_coord_0} must be 2^k"
        return dict(
            global_dim_0=self.global_dim_0, global_dim_1=global_dim_1,
            box_dim_0=self.box_dim_0, box_dim_1=box_dim_1,
            swizzle=self.swizzle, tensor_data_type=self.tensor_data_type,
            elem_bytes=self.elem_bytes,
            mask0=n_coord_0 - 1, shift0=n_coord_0.bit_length() - 1,
            step0=self.box_dim_0, mask1=n_coord_1 - 1, step1=box_dim_1,
            n_coord_1=n_coord_1,
            # ADDRESSABLE range, NOT what the walk touches -- see touched_bytes.
            # Reporting this as "footprint" is what let 68 MB walks be labelled
            # 235 MB and read as cold DRAM against a 50 MB L2, which cost a
            # constant its credibility.
            addressable_bytes=n_coord_1 * box_dim_1 * self.row_bytes,
        )

    def touched_bytes(self, plan: dict, n_ctas: int, num_producers: int,
                      k_tile_count: int) -> int:
        """Bytes the walk ACTUALLY reaches, which is what L2 sees.

        `idx = base * k_tile_count + k` with base < n_ctas * num_producers, so
        idx tops out at n_ctas*num_producers*k_tile_count - 1 and the coord_1
        term only reaches `(idx_max >> shift0) + 1` of the plan's values.
        Everything above that is addressable and never visited.
        """
        idx_max = n_ctas * num_producers * k_tile_count - 1
        reached = min(plan["n_coord_1"], (idx_max >> plan["shift0"]) + 1)
        return reached * plan["box_dim_1"] * self.row_bytes


def encode(unit, ptr: int, plan: dict, swizzle: int = None):
    """Encode a descriptor through the unit's optional TMA extension.

    Returns ((buffer, address), rc). rc != 0 means the driver REJECTED the
    geometry, which is a result to record rather than an error to raise --
    enumerating what the driver accepts is how tma.bytes.txn.max was measured.
    """
    n = unit.lib.hut_tensor_map_bytes()
    raw = ctypes.create_string_buffer(n + 64)
    addr = ctypes.addressof(raw)
    off = (-addr) % 64          # cuTensorMapEncodeTiled needs 64 B alignment
    rc = unit.lib.hut_encode_tensor_map(
        ctypes.c_void_p(addr + off), ctypes.c_void_p(ptr),
        plan["global_dim_0"], plan["global_dim_1"],
        plan["box_dim_0"], plan["box_dim_1"],
        plan["swizzle"] if swizzle is None else swizzle,
        plan["tensor_data_type"], plan["elem_bytes"])
    return (raw, addr + off), rc


# The three patterns that appear in this repo's FFN task-loop, in the order the
# config table declares them.
def geoms(elem_bytes=2, tensor_data_type=DT_BF16):
    return {
        "contig":   Geom("contig", 64, tensor_data_type=tensor_data_type,
                         elem_bytes=elem_bytes),
        "stride2k": Geom("stride2k", 1024, tensor_data_type=tensor_data_type,
                         elem_bytes=elem_bytes),
        "stride8k": Geom("stride8k", 4096, tensor_data_type=tensor_data_type,
                         elem_bytes=elem_bytes),
    }
