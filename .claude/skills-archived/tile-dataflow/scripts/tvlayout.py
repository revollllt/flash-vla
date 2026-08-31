"""Layout / swizzle / thread-value core for the L4 access checker.

Self-contained on purpose: these are design-time tools that must run on the
login node (python 3.6, no torch, no GPU). `tests/test_tvlayout.py`
cross-validates every function here against CUTLASS's own `pycute` when that
checkout is present, so the reimplementation is checked, not trusted.

Three objects:

  Layout(shape, stride)   a map (coord...) -> element offset. Nested shapes are
                          supported, matching CuTe: shape ((8,8),2) with stride
                          ((1,16),128) is a legal layout here.
  Swizzle(bits,base,shift)  CuTe's Swizzle<B,M,S>, applied to the ELEMENT offset.
                          The XOR layer is why this file exists: an affine
                          layout composed with an XOR is not affine, and the
                          composition is what nobody computes correctly by hand.
  TV(threads, vals, fn)   (tid, vid) -> coord, the map from a lane to what it
                          touches. `fn` returns a coord tuple for Layout.

Conventions, stated once because getting them wrong silently changes answers:

  * Layout offsets are in ELEMENTS. Byte addresses are `offset * dtype_bytes`.
  * Swizzle is applied to the element offset, before the byte scaling. This is
    CuTe's convention, and it is why Swizzle<3,3,3> is the 128 B atom for bf16
    (base 3 => 8 elements => 16 B granule) while fp32 needs Swizzle<3,2,3>.
  * A "word" is 4 bytes: the shared-memory bank granule.
"""

from __future__ import division


# --------------------------------------------------------------- int tuples

def is_tuple(x):
    return isinstance(x, (tuple, list))


def flatten(x):
    """Flatten a nested int tuple to a flat tuple of ints."""
    if not is_tuple(x):
        return (x,)
    out = []
    for e in x:
        out.extend(flatten(e))
    return tuple(out)


def product(x):
    p = 1
    for e in flatten(x):
        p *= e
    return p


def crd2idx(crd, shape, stride):
    """Map a (possibly nested, possibly linear) coordinate to an offset.

    Mirrors CuTe: a coordinate may be given either as a tuple matching the
    shape's structure, or as a single integer, in which case it is unravelled
    against the shape in column-major (first mode fastest) order.
    """
    if is_tuple(shape):
        if is_tuple(crd):
            if len(crd) != len(shape):
                raise ValueError("coord rank %d != shape rank %d" % (len(crd), len(shape)))
            return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
        # linear coordinate unravelled against the shape
        idx = int(crd)
        total = 0
        for s, d in zip(shape, stride):
            n = product(s)
            total += crd2idx(idx % n, s, d)
            idx //= n
        return total
    return int(crd) * int(stride)


class Layout(object):
    """An element-offset map, `(coord...) -> offset`."""

    def __init__(self, shape, stride=None):
        self.shape = tuple(shape) if is_tuple(shape) else (shape,)
        if stride is None:
            stride = compact_col_major(self.shape)
        self.stride = tuple(stride) if is_tuple(stride) else (stride,)
        if len(self.shape) != len(self.stride):
            raise ValueError("shape/stride rank mismatch: %r vs %r" % (self.shape, self.stride))

    def __call__(self, *crd):
        if len(crd) == 1 and is_tuple(crd[0]):
            crd = crd[0]
        if len(crd) == 1 and not is_tuple(crd[0]):
            return crd2idx(crd[0], self.shape, self.stride)
        return crd2idx(tuple(crd), self.shape, self.stride)

    def size(self):
        return product(self.shape)

    def cosize(self):
        """One past the largest offset the layout can produce."""
        hi = 0
        for s, d in zip(flatten(self.shape), flatten(self.stride)):
            hi += (s - 1) * d
        return hi + 1

    def __repr__(self):
        return "%s:%s" % (_fmt(self.shape), _fmt(self.stride))


def compact_col_major(shape):
    out, acc = [], 1
    for s in shape:
        if is_tuple(s):
            out.append(compact_col_major(s))
            acc *= product(s)
        else:
            out.append(acc)
            acc *= s
    return tuple(out)


def _fmt(x):
    if is_tuple(x):
        return "(" + ",".join(_fmt(e) for e in x) + ")"
    return str(x)


# ------------------------------------------------------------------ swizzle

class Swizzle(object):
    """CuTe's Swizzle<B, M, S>, applied to an element offset.

    offset ^= ((offset >> (M + S)) & ((1 << B) - 1)) << M

    B (`bits`)  how many bits are permuted -> 2**B distinct patterns
    M (`base`)  low bits left untouched -> the vectorisation granule that
                survives the swizzle. Getting this smaller than your access
                width is how a swizzle breaks a 128-bit store.
    S (`shift`) distance from the untouched low bits to the bits used as the
                XOR key -- in a row-major tile this is the row index.
    """

    def __init__(self, bits, base, shift):
        if bits < 0 or base < 0 or abs(shift) < bits:
            raise ValueError("illegal Swizzle<%d,%d,%d>" % (bits, base, shift))
        self.bits, self.base, self.shift = bits, base, shift
        msk = (1 << bits) - 1
        self.yyy_msk = msk << (base + max(0, shift))
        self.zzz_msk = msk << (base - min(0, shift))

    def __call__(self, offset):
        offset = int(offset)
        if self.shift >= 0:
            return offset ^ ((offset & self.yyy_msk) >> self.shift)
        return offset ^ ((offset & self.yyy_msk) << (-self.shift))

    def __repr__(self):
        return "Swizzle<%d,%d,%d>" % (self.bits, self.base, self.shift)


class NoSwizzle(object):
    def __call__(self, offset):
        return int(offset)

    def __repr__(self):
        return "none"


def swizzle_for(mode, dtype_bytes):
    """The canonical CuTe swizzle atom for a named smem mode.

    `mode` is the byte width of the atom's row, which is how everyone names
    these ("128 B swizzle"). The three parameters are forced, not chosen:

      base  = log2(16 / dtype_bytes)   16 B stays contiguous, so a 128-bit
                                       access survives the swizzle. This is the
                                       only place the dtype enters, and it is
                                       why the same "128 B swizzle" is
                                       Swizzle<3,3,3> for bf16, <3,2,3> for f32
                                       and <3,4,3> for fp8.
      bits  = log2(mode / 16)          how many 16 B chunks a row holds
      shift = 3                        always: the XOR key is the 128 B-line
                                       index, which sits 3 chunk-bits above the
                                       granule regardless of mode. For a 64 B
                                       atom that makes the key row//2, and for
                                       32 B row//4 -- exactly the rows that
                                       share a 128 B bank cycle.
    """
    if mode in (None, "none", 0):
        return NoSwizzle()
    mode = int(mode)
    if mode not in (32, 64, 128):
        raise ValueError("swizzle mode must be 32, 64 or 128 bytes, got %r" % mode)
    if 16 % dtype_bytes:
        raise ValueError("dtype_bytes must divide 16, got %r" % dtype_bytes)
    return Swizzle(_log2(mode // 16), _log2(16 // dtype_bytes), 3)


SWIZZLE_ATOM_ROWS = 8       # every CuTe smem swizzle atom is 8 rows tall


class SwizzledTile(object):
    """A row-major smem tile tiled by CuTe swizzle atoms: `(row, col) -> offset`.

    Why this is not just `Swizzle(...)` applied to a flat row-major offset: the
    swizzle is an atom of 8 rows x `mode` bytes, and a tile whose rows are WIDER
    than the atom is a tiling of atoms, not one big swizzle. Applying the XOR to
    the flat offset of a 256 B-row tile keys off the wrong bits and silently
    reports conflicts that the real layout does not have.

    Bank results from this layout are independent of the order the atoms are
    tiled in, because an atom occupies 8 * mode bytes -- 1024, 512 or 256 -- all
    multiples of 128 B, so the atom base contributes nothing to the bank index.
    The order below (atoms row-major, K contiguous) therefore only fixes the
    absolute addresses, and matches `tile_to_shape(Layout_K_SW*_Atom, ...)`.
    """

    def __init__(self, rows, cols, dtype_bytes, mode):
        self.rows, self.cols, self.dtype_bytes = rows, cols, dtype_bytes
        self.mode = mode
        self.swizzle = swizzle_for(mode, dtype_bytes)
        if mode in (None, "none", 0):
            self.atom_cols = cols
        else:
            self.atom_cols = int(mode) // dtype_bytes
            if cols % self.atom_cols:
                raise ValueError(
                    "tile width %d elems (%d B) is not a whole number of %d B atoms"
                    % (cols, cols * dtype_bytes, mode))
            if rows % SWIZZLE_ATOM_ROWS:
                raise ValueError("tile height %d is not a multiple of the atom's 8 rows" % rows)
        self.atom_elems = SWIZZLE_ATOM_ROWS * self.atom_cols
        self.atoms_per_row = cols // self.atom_cols

    def __call__(self, *crd):
        if len(crd) == 1 and is_tuple(crd[0]):
            crd = crd[0]
        row, col = int(crd[0]), int(crd[1])
        atom_id = (row // SWIZZLE_ATOM_ROWS) * self.atoms_per_row + col // self.atom_cols
        within = (row % SWIZZLE_ATOM_ROWS) * self.atom_cols + col % self.atom_cols
        return atom_id * self.atom_elems + self.swizzle(within)

    def size(self):
        return self.rows * self.cols

    def __repr__(self):
        return "SwizzledTile(%d,%d,%dB elems,swizzle=%s -> %s)" % (
            self.rows, self.cols, self.dtype_bytes, self.mode, self.swizzle)


def _log2(n):
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("%r is not a positive power of two" % n)
    return n.bit_length() - 1


# ------------------------------------------------------------ thread/value

class TV(object):
    """The map `(tid, vid) -> coord` for one access.

    `threads` is the number of threads that participate in ONE instruction
    (32 for a warp-scoped access, 128 for a warp-group-wide epilogue store).
    `vals` is how many separate instructions each thread issues.
    `vec` is how many ELEMENTS one instruction moves per thread; the checker
    verifies those elements are actually contiguous after swizzling rather than
    taking the claim on faith.
    """

    def __init__(self, threads, vals, fn, vec=1, name=""):
        self.threads, self.vals, self.fn, self.vec, self.name = threads, vals, fn, vec, name

    def coord(self, tid, vid, elem=0):
        return self.fn(tid, vid, elem)


# --------------------------------------------------------------- TV atoms

def wgmma_acc(n, vec=1, project="both", threads=128):
    """The fp32 accumulator fragment of `wgmma.mma_async.m64nNk*`.

    One warp group (128 threads) holds a 64 x N f32 tile. Warp w owns rows
    [16w, 16w+16); within a warp, lane l owns rows 16w + l//4 and 16w + l//4 + 8,
    and columns 2*(l%4) + 8*j + {0,1} for j in [0, N/8).

    This is the single most error-prone TV map in a Hopper kernel -- it decides
    the epilogue's store pattern, the softmax reduction's span, and which scale
    vector a thread needs -- so it ships as an atom rather than as an expression
    each spec rewrites.

    `project`:
      both  -> coord (row, col), for a 2-D accumulator-shaped buffer
      row   -> coord (row,),     for a per-row vector (an A-scale, a running max)
      col   -> coord (col,),     for a per-column vector (a B-scale, a bias)

    `vec` is elements per instruction along the column axis, so vec=2 is the
    float2 / st.shared.b64 form the accumulator's natural pairing gives you.
    """
    if n % 8:
        raise ValueError("wgmma N must be a multiple of 8, got %d" % n)
    if project not in ("both", "row", "col"):
        raise ValueError("project must be both|row|col")

    if project == "row":
        # A per-row vector: the column index is irrelevant, so a lane issues one
        # access per row half. vec > 1 is meaningless here -- the two rows a lane
        # owns are 8 apart, never contiguous.
        if vec != 1:
            raise ValueError("wgmma_acc project=row has no contiguous run to vectorise")

        def fn_row(tid, vid, elem):
            w, l = tid // 32, tid % 32
            return (16 * w + l // 4 + 8 * vid,)

        return TV(threads, 2, fn_row, vec=1, name="wgmma_acc[m64n%d].row" % n)

    per_half = _ncols_vals(n, vec)

    if project == "col":
        def fn_col(tid, vid, elem):
            return (_col_of(vid, vec, tid % 32) + elem,)

        return TV(threads, per_half, fn_col, vec=vec,
                  name="wgmma_acc[m64n%d].col" % n)

    # both: vid enumerates (row-half h, column group c), c fastest
    def fn_both(tid, vid, elem):
        w, l = tid // 32, tid % 32
        h, c = vid // per_half, vid % per_half
        return (16 * w + l // 4 + 8 * h, _col_of(c, vec, l) + elem)

    return TV(threads, 2 * per_half, fn_both, vec=vec, name="wgmma_acc[m64n%d]" % n)


def _ncols_vals(n, step):
    """How many instructions per row-half, when each moves `step` columns.

    The accumulator gives each lane 2 contiguous columns at 2*(l%4) + 8j. A
    step of 1 or 2 stays inside one such pair; a wider step would have to cross
    the 8-column gap, which is not contiguous, so it is rejected.
    """
    if step > 2:
        raise ValueError("wgmma accumulator columns come in contiguous pairs; "
                         "vec > 2 would cross the 8-column stride and is not contiguous")
    return (n // 8) * (2 // step)


def _col_of(c, step, l):
    j = c // (2 // step)
    off = (c % (2 // step)) * step
    return 2 * (l % 4) + 8 * j + off


def linear(cols, threads, vals, vec=1, rows=None):
    """The plain coalesced pattern: flatten the tile, hand thread t the run
    [t*vec, t*vec+vec), stride the whole CTA forward each instruction."""
    def fn(tid, vid, elem):
        flat = (vid * threads + tid) * vec + elem
        if rows is None:
            return (flat,)
        return (flat // cols, flat % cols)
    return TV(threads, vals, fn, vec=vec, name="linear(vec=%d)" % vec)


def expression(src, threads, vals, vec=1, name="expr"):
    """Escape hatch: a python snippet with `tid`, `vid`, `elem` bound, which
    must assign `coord`. Restricted namespace -- no imports, no builtins."""
    code = compile(src, "<tv>", "exec")
    env = {"__builtins__": {"range": range, "min": min, "max": max, "abs": abs, "int": int}}

    def fn(tid, vid, elem):
        loc = {"tid": tid, "vid": vid, "elem": elem}
        exec(code, env, loc)
        if "coord" not in loc:
            raise ValueError("tv expression must assign `coord`")
        c = loc["coord"]
        return tuple(c) if is_tuple(c) else (c,)

    return TV(threads, vals, fn, vec=vec, name=name)


ATOMS = {"wgmma_acc": wgmma_acc, "linear": linear}
