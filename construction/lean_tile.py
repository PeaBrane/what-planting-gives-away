"""Memory-lean exact TILE decision for larger tori (independent of the gadget theory).

Every cell's ground-state set is recomputed from its raw coupling. A cell whose ground-state set is exactly the
solution set of its pairwise corner locks (an affine relation) contributes XOR constraints to a parity union-find
over array-indexed vertices; every other cell enters a 0/1 MILP (scipy HiGHS) over union-find representatives.
"""
import itertools
import os
import sys
from array import array

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiles import CORNERS, gs, lock_classes  # noqa: E402

_GS = {}


def gs_info(W):
    if W not in _GS:
        G = sorted(gs(W))
        locks = lock_classes(G)
        affine = len(G) == 2 ** len(locks)
        lk = []
        if affine:
            m = G[0]
            for cls in locks:
                k0 = cls[0]
                for k in cls[1:]:
                    lk.append((k0, k, ((m >> k) ^ (m >> k0)) & 1))
        _GS[W] = (G, affine, lk)
    return _GS[W]


class UF:
    def __init__(self, n):
        self.parent = array("i", range(n))
        self.par = bytearray(n)

    def find(self, v):
        parent, par = self.parent, self.par
        acc = 0
        r = v
        while parent[r] != r:
            acc ^= par[r]
            r = parent[r]
        # compress
        a2 = acc
        while parent[v] != r:
            nv, nb = parent[v], par[v]
            parent[v], par[v] = r, a2
            a2 ^= nb
            v = nv
        return r, acc

    def union(self, a, b, x):
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == x
        self.parent[ra] = rb
        self.par[ra] = pa ^ pb ^ x
        return True


def all_cells(L):
    for par in (0, 1):
        for x in range(par, L, 2):
            for y in range(par, L, 2):
                for z in range(par, L, 2):
                    yield (x, y, z)


def corner_index(L, u, k):
    d = CORNERS[k]
    return ((u[0] + d[0]) % L) + L * (((u[1] + d[1]) % L) + L * ((u[2] + d[2]) % L))


def vindex(L, v):
    return (v[0] % L) + L * ((v[1] % L) + L * (v[2] % L))


def decide(L, W_of, fix=None, time_limit=60, want_bits=False):
    """W_of(cell) -> coupling tuple.  fix: {vertex tuple: bit}.  Returns (feasible, bits or None)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n = L ** 3
    ZERO = n
    uf = UF(n + 1)
    rest = []
    for u in all_cells(L):
        G, affine, lk = gs_info(W_of(u))
        if affine:
            for k0, k, x in lk:
                if not uf.union(corner_index(L, u, k0), corner_index(L, u, k), x):
                    return False, None
        else:
            rest.append((u, G))
    for v, b in (fix or {}).items():
        if not uf.union(vindex(L, v), ZERO, b):
            return False, None
    reps = {}
    rz, pz = uf.find(ZERO)
    reps[rz] = 0
    rows, cidx, vals, lo, hi = [], [], [], [], []
    r = 0
    corner_refs = []
    for u, G in rest:
        refs = []
        for k in range(8):
            rv, pv = uf.find(corner_index(L, u, k))
            if rv not in reps:
                reps[rv] = len(reps)
            refs.append((reps[rv], pv))
        corner_refs.append(refs)
    cols = len(reps)
    for (u, G), refs in zip(rest, corner_refs):
        ys = list(range(cols, cols + len(G)))
        cols += len(G)
        for y in ys:
            rows.append(r); cidx.append(y); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
        for k in range(8):
            ci, pv = refs[k]
            # bit = rep xor pv = sum_y y [m_k]
            if pv == 0:
                rows.append(r); cidx.append(ci); vals.append(-1.0); rhs = 0.0
            else:
                rows.append(r); cidx.append(ci); vals.append(1.0); rhs = 1.0
            for y, m in zip(ys, G):
                if (m >> k) & 1:
                    rows.append(r); cidx.append(y); vals.append(1.0)
            lo.append(rhs); hi.append(rhs); r += 1
    lb = np.zeros(cols)
    ub = np.ones(cols)
    lb[0] = ub[0] = pz  # the class of the constant 0
    if r == 0:
        rows, cidx, vals, lo, hi, r = [0], [0], [0.0], [0.0], [0.0], 1
    A = coo_matrix((vals, (rows, cidx)), shape=(r, cols)).tocsr()
    res = milp(np.zeros(cols), constraints=LinearConstraint(A, lo, hi), integrality=np.ones(cols),
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "presolve": True})
    if res.status == 2:
        return False, None
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    if not want_bits:
        return True, None
    x = np.round(res.x).astype(int)
    bits = bytearray(n)
    for v in range(n):
        rv, pv = uf.find(v)
        bits[v] = (int(x[reps[rv]]) if rv in reps else 0) ^ pv
    return True, bits


def check_bits(L, W_of, bits):
    """Independent check of a returned state: every cell at a local minimum."""
    for u in all_cells(L):
        G, _, _ = gs_info(W_of(u))
        m = 0
        for k in range(8):
            if bits[corner_index(L, u, k)]:
                m |= 1 << k
        if m not in G:
            return False
    return True
