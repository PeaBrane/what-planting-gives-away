"""Lattice geometry, gadget couplings and the TILE checker for the L^3 torus.

Cells: unit cubes with lower corner u, all coordinates of equal parity.  Corner k of cell u (k = dx + 2dy + 4dz) is
the vertex u + d_k; the other cell through that vertex is u + (2 d_k - 1).  A direction a in {+-1}^3 from cell u
names the corner d = (a + 1) / 2.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import CORNERS, CUBE_E, class_name  # noqa: E402

DIRS = [tuple(2 * c - 1 for c in CORNERS[k]) for k in range(8)]  # direction of corner k
DIR_TO_K = {a: k for k, a in enumerate(DIRS)}

# ---------------------------------------------------------------- ground-state table for all 4096 couplings
_S = np.array([[(-1 if (m >> k) & 1 else 1) for k in range(8)] for m in range(256)])
_PROD = np.stack([_S[:, a] * _S[:, b] for a, b in CUBE_E], axis=1)  # (256, 12)
ALLW = np.array(list(itertools.product([1, -1], repeat=12)))
_EN = -(_PROD @ ALLW.T)  # (256, 4096)
GS_OF = []
for j in range(ALLW.shape[0]):
    col = _EN[:, j]
    GS_OF.append(frozenset(np.nonzero(col == col.min())[0].tolist()))
BY_GS = {}
for j, g in enumerate(GS_OF):
    BY_GS.setdefault(g, []).append(j)


def gs(W):
    W = np.asarray(W)
    en = -(_PROD @ W)
    return frozenset(np.nonzero(en == en.min())[0].tolist())


def coupling_with_gs(target, cls=None):
    target = frozenset(target)
    for j in BY_GS.get(target, []):
        W = tuple(int(x) for x in ALLW[j])
        if cls is None or class_name(W) == cls:
            return W
    raise ValueError(f"no coupling with ground-state set {sorted(target)} (class {cls})")


def mask(ks):
    m = 0
    for k in ks:
        m |= 1 << k
    return m


def add(u, a, L):
    return tuple((u[i] + a[i]) % L for i in range(3))


def is_cell(u):
    return u[0] % 2 == u[1] % 2 == u[2] % 2


def all_cells(L):
    return [u for u in itertools.product(range(L), repeat=3) if is_cell(u)]


def corner_vertex(u, k, L):
    return add(u, CORNERS[k], L)


def neighbour(u, k, L):
    return add(u, DIRS[k], L)


def dir_between(u, w, L):
    """Direction a in {+-1}^3 with w = u + a (mod L), or None."""
    d = []
    for i in range(3):
        x = (w[i] - u[i]) % L
        if x == 1:
            d.append(1)
        elif x == L - 1:
            d.append(-1)
        else:
            return None
    return tuple(d)


# ---------------------------------------------------------------- standard gadget couplings (logical frame = spin frame)
FULL = 255
W_BACKGROUND = coupling_with_gs({0, FULL}, "F21")


def wire_coupling(k1, k2):
    D = mask([k1, k2])
    assert bin(k1 ^ k2).count("1") == 1, "wire corners must be a domino"
    return coupling_with_gs({0, D, FULL, FULL ^ D}, "F41")


def fanout_coupling(face):
    F = mask(face)
    return coupling_with_gs({0, F, FULL, FULL ^ F}, "F42")


# ---------------------------------------------------------------- TILE as an ILP (independent of the gadget theory)
class ParityUF:
    def __init__(self):
        self.parent, self.par = {}, {}

    def find(self, v):
        if v not in self.parent:
            self.parent[v], self.par[v] = v, 0
            return v, 0
        p, acc = v, 0
        while self.parent[p] != p:
            acc ^= self.par[p]
            p = self.parent[p]
        # path compression
        q, a2 = v, acc
        while self.parent[q] != p:
            nq, nb = self.parent[q], self.par[q]
            self.parent[q], self.par[q] = p, a2
            a2 ^= nb
            q = nq
        return p, acc

    def union(self, a, b, x):
        """impose bit(a) xor bit(b) = x; return False on contradiction."""
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == x
        self.parent[ra], self.par[ra] = rb, pa ^ pb ^ x
        return True


def lock_classes(G):
    """Partition of the 8 corners into classes whose pairwise xor is constant over the state set G."""
    classes = []
    for k in range(8):
        for cls in classes:
            j = cls[0]
            if len({((m >> k) ^ (m >> j)) & 1 for m in G}) == 1:
                cls.append(k)
                break
        else:
            classes.append([k])
    return classes


def tile_milp(L, Wall, fix=None, time_limit=60):
    """Decide TILE: exists s with s|_C in GS(W_C) for every cell C.  fix: {vertex: bit} extra constraints.
    Exact preprocessing: a cell whose ground-state set is {m, ~m} imposes fixed XORs between its corners; these are
    merged by a parity union-find.  The remaining cells become a MILP over class representatives.
    Returns (feasible, bits)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    uf = ParityUF()
    ZERO = "zero"
    uf.find(ZERO)
    rest = []
    for u, W in Wall.items():
        G = sorted(gs(W))
        locks = lock_classes(G)
        if len(G) == 2 ** len(locks):
            # the ground-state set is exactly the solution set of its pairwise locks (an affine relation)
            m = G[0]
            for cls in locks:
                k0 = cls[0]
                for k in cls[1:]:
                    x = ((m >> k) ^ (m >> k0)) & 1
                    if not uf.union(corner_vertex(u, k0, L), corner_vertex(u, k, L), x):
                        return False, None
        else:
            rest.append((u, G))
    if fix:
        for v, b in fix.items():
            if not uf.union(v, ZERO, b):
                return False, None
    verts = list(itertools.product(range(L), repeat=3))
    reps = {}
    for v in verts:
        r, _ = uf.find(v)
        reps.setdefault(r, len(reps))
    rz, _ = uf.find(ZERO)
    reps.setdefault(rz, len(reps))
    cols = len(reps)
    rows, cidx, vals, lo, hi = [], [], [], [], []
    r = 0
    for u, G in rest:
        ys = list(range(cols, cols + len(G)))
        cols += len(G)
        for y in ys:
            rows.append(r); cidx.append(y); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
        for k in range(8):
            rv, pv = uf.find(corner_vertex(u, k, L))
            # bit(v) = rep xor pv ;  bit(v) = sum_y y [m_k]
            if pv == 0:
                rows.append(r); cidx.append(reps[rv]); vals.append(-1.0); rhs = 0.0
            else:
                rows.append(r); cidx.append(reps[rv]); vals.append(1.0); rhs = 1.0
            for y, m in zip(ys, G):
                if (m >> k) & 1:
                    rows.append(r); cidx.append(y); vals.append(1.0)
            lo.append(rhs); hi.append(rhs); r += 1
    lb = np.zeros(cols); ub = np.ones(cols)
    lb[reps[rz]] = ub[reps[rz]] = 0
    if r == 0:
        rows, cidx, vals, lo, hi, r = [0], [0], [0.0], [0.0], [0.0], 1
    A = coo_matrix((vals, (rows, cidx)), shape=(r, cols)).tocsr()
    res = milp(np.zeros(cols), constraints=LinearConstraint(A, lo, hi), integrality=np.ones(cols),
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "presolve": True})
    if res.status == 2:
        return False, None
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    x = np.round(res.x).astype(int)
    bits = {}
    for v in verts:
        rv, pv = uf.find(v)
        bits[v] = int(x[reps[rv]]) ^ pv
    return True, bits


def check_solution(L, Wcell, bits):
    for u, W in Wcell.items():
        m = 0
        for k in range(8):
            if bits[corner_vertex(u, k, L)]:
                m |= 1 << k
        if m not in gs(W):
            return False
    return True
