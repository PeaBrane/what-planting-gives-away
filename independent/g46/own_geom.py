"""Independent block geometry (construction notes Secs. 4.2, 8.3 at S = 12) and a box-indexed lock graph.  Couplings of the
background are taken from the construction instance rule (they are instance data) but their ground states and locks are
recomputed here from the raw 12 couplings."""
import itertools
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_cube as O  # noqa: E402

S, H = 12, 6
SLAB = (-1, 0)
TPLUS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
DIRS = [tuple(2 * c - 1 for c in O.CORN[k]) for k in range(8)]
DIR_K = {a: k for k, a in enumerate(DIRS)}


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def iscell(u):
    return u[0] % 2 == u[1] % 2 == u[2] % 2


def a_perp(i, tau):
    a = next(t for t in TPLUS if t[i] == tau)
    return tuple(0 if j == i else a[j] for j in range(3))


def outer_cell(C, i, sg, tau):
    """Outer crossing cell of face (i, sg) of the block centred at C, for travel tau e_i (Sec. 4.2)."""
    if sg == 1:
        return add(add(C, a_perp(i, tau)), tuple((H - 1) if j == i else 0 for j in range(3)))
    return add(C, tuple(-H if j == i else 0 for j in range(3)))


def crossing_cells(g):
    C = tuple(S * x for x in g)
    return {outer_cell(C, i, sg, tau) for i in range(3) for sg in (1, -1) for tau in (1, -1)}


def block_foam(g):
    C = tuple(S * x for x in g)
    cr = crossing_cells(g)
    out = []
    for d in itertools.product(range(-H, H), repeat=3):
        if any(x in (-H, H - 1) for x in d):
            u = add(C, d)
            if iscell(u) and u not in cr:
                out.append(u)
    return out


def block_interior(g, exclude=()):
    C = tuple(S * x for x in g)
    ex = set(exclude)
    return [add(C, d) for d in itertools.product(range(-H + 1, H - 1), repeat=3)
            if iscell(add(C, d)) and add(C, d) not in ex]


_BG_LOCKS = {}


def bg_locks(ax):
    """Lock pairs (j, k, parity) of the instance background coupling for plate axis ax, from raw couplings."""
    if ax not in _BG_LOCKS:
        os.environ.setdefault("TILE_S", "12")
        sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "construction"))
        import layout46 as G  # construction: only the instance's coupling values are read
        u = {0: (0, 0, 0), 1: (0, 0, 4), 2: (2, 0, 0)}[ax]
        assert sum(u) % 3 == ax
        W = G.bg_coupling(u)
        _BG_LOCKS[ax] = locks_of(W)
    return _BG_LOCKS[ax]


def locks_of(W):
    Gm = O.gamma(W)
    cl, par, aff = O.lock_pairs(Gm)
    assert aff and O.cls(W) == "F42", (W, O.cls(W))
    rep = {}
    out = []
    for k in range(8):
        if cl[k] in rep:
            j = rep[cl[k]]
            out.append((j, k, par[k] ^ par[j]))
        else:
            rep[cl[k]] = k
    return out


class Box:
    """Vertices of a finite non-periodic box, array indexed."""

    def __init__(self, lo, hi):
        self.lo = tuple(lo)
        self.n = tuple(h - l + 1 for l, h in zip(lo, hi))

    def vid(self, v):
        x, y, z = (v[j] - self.lo[j] for j in range(3))
        assert all(0 <= c < n for c, n in zip((x, y, z), self.n)), v
        return (x * self.n[1] + y) * self.n[2] + z

    def size(self):
        return self.n[0] * self.n[1] * self.n[2]


class LockGraph:
    def __init__(self, box):
        self.box = box
        self.src, self.dst = [], []
        self.n_par1 = 0

    def add_cell(self, u, locks):
        for j, k, p in locks:
            if p:
                self.n_par1 += 1  # never happens for 0-in-Gamma couplings; asserted below
            self.src.append(self.box.vid(add(u, O.CORN[j])))
            self.dst.append(self.box.vid(add(u, O.CORN[k])))

    def add_bg(self, cells):
        for u in cells:
            self.add_cell(u, bg_locks(sum(u) % 3))

    def solve(self):
        assert self.n_par1 == 0
        n = self.box.size()
        A = coo_matrix((np.ones(len(self.src), dtype=np.int8), (np.array(self.src), np.array(self.dst))),
                       shape=(n, n))
        self.ncomp, self.lab = connected_components(A, directed=False)
        return self

    def comp(self, v):
        return int(self.lab[self.box.vid(v)])
