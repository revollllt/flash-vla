"""The one binding to include/hut/unit.hpp.

Thirteen hand-written ctypes argtypes lists broke three times in one session,
every time by appending a trailing pointer to a positional signature. Parameters
travel in a Structure instead: adding an axis edits one class here and one
struct there, and a mismatch is a field name rather than a silent misread.

Field names follow references/vocabulary.md -- `stages` not `depth`,
`k_tile_count` not `trip`, and the three byte quantities `frame_b` used to
conflate kept apart.
"""

from __future__ import annotations

import ctypes

# Mirrors HutFlags.
NEEDS_COLD = 1 << 0
HAS_CHECK = 1 << 1
REPORTS_SM = 1 << 2
NO_SOURCE = 1 << 3

ERR = {-1: "bad cfg", -2: "bad param", -3: "smem", -4: "unit declares no check"}


class HutParams(ctypes.Structure):
    """Mirrors struct HutParams. Field ORDER must match the header."""
    _fields_ = [
        ("cfg", ctypes.c_int32),
        ("mode", ctypes.c_int32),
        ("n_ctas", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("num_producers", ctypes.c_int32),
        ("num_consumers", ctypes.c_int32),
        ("stages", ctypes.c_int32),
        ("k_tile_count", ctypes.c_int32),
        ("box_bytes", ctypes.c_int32),
        ("txn_bytes", ctypes.c_int32),
        ("stage_bytes", ctypes.c_int32),
        ("mask0", ctypes.c_int32),
        ("shift0", ctypes.c_int32),
        ("step0", ctypes.c_int32),
        ("mask1", ctypes.c_int32),
        ("step1", ctypes.c_int32),
        ("opt", ctypes.c_int32 * 4),
        ("tensor_map", ctypes.c_void_p),
        ("operand_a", ctypes.c_void_p),
        ("operand_b", ctypes.c_void_p),
    ]


class HutBuffers(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in
                ("cycles_a", "cycles_b", "sink", "dbg", "sm_id", "out")]


def _ptr(t):
    """Device pointer of a torch tensor, or None."""
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def buffers(**kw) -> HutBuffers:
    return HutBuffers(*[_ptr(kw.get(n)) for n in
                        ("cycles_a", "cycles_b", "sink", "dbg", "sm_id", "out")])


class Unit:
    """A loaded unit .so, bound through the uniform ABI."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        pp, pb = ctypes.POINTER(HutParams), ctypes.POINTER(HutBuffers)
        for name, res, args in (
            ("hut_name", ctypes.c_char_p, []),
            ("hut_flags", ctypes.c_uint32, []),
            ("hut_cfg_count", ctypes.c_int32, []),
            ("hut_cfg", ctypes.c_int32, [ctypes.c_int32] * 2),
            ("hut_cfg_name", ctypes.c_char_p, [ctypes.c_int32]),
            ("hut_opt_name", ctypes.c_char_p, [ctypes.c_int32]),
            ("hut_smem", ctypes.c_int32, [pp]),
            ("hut_check", ctypes.c_int32, [pp, pb, ctypes.c_void_p]),
            ("hut_launch", ctypes.c_int32, [pp, pb, ctypes.c_void_p]),
        ):
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = res, args
        # Optional TMA extension: present only in units that drive TMA.
        if hasattr(self.lib, "hut_encode_tensor_map"):
            self.lib.hut_encode_tensor_map.restype = ctypes.c_int32
            self.lib.hut_encode_tensor_map.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
            self.lib.hut_tensor_map_bytes.restype = ctypes.c_int32
        self.name = self.lib.hut_name().decode()
        self.flags = self.lib.hut_flags()
        self.n_cfg = self.lib.hut_cfg_count()
        self.cfg_fields = self._names(self.lib.hut_cfg_name)
        self.opt_fields = self._names(self.lib.hut_opt_name)

    @staticmethod
    def _names(fn):
        out = []
        for i in range(8):
            s = fn(i)
            if not s:
                break
            out.append(s.decode())
        return out

    def cfg(self, i) -> dict:
        return {n: self.lib.hut_cfg(i, f)
                for f, n in enumerate(self.cfg_fields)}

    def _call(self, fn, params, bufs, what):
        rc = fn(ctypes.byref(params), ctypes.byref(bufs), None)
        if rc != 0:
            raise RuntimeError(
                f"{self.name}.{what} rc={rc} ({ERR.get(rc, 'cudaError')})")

    def launch(self, params: HutParams, bufs: HutBuffers):
        self._call(self.lib.hut_launch, params, bufs, "launch")

    def check(self, params: HutParams, bufs: HutBuffers):
        if not self.flags & HAS_CHECK:
            raise RuntimeError(
                f"{self.name} declares no correctness check; protocol rule 11 "
                f"requires one for any unit whose instruction computes or moves "
                f"something. If that is deliberate, say why in the unit's "
                f"isolation: field.")
        self._call(self.lib.hut_check, params, bufs, "check")
