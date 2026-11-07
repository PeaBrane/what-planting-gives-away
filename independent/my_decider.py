"""Independent TILE decider.  Written from the definitions only; imports nothing from construction/.

Input: even L >= 4, a default 12-coupling W_default and a dict {cell (x,y,z): W} for the other cells.
Cells are lower corners u with u1 = u2 = u3 (mod 2) on the L^3 torus; corner k = dx + 2dy + 4dz is the vertex
u + d; edges of a cell are ordered as the pairs (a, b), a < b, a ^ b a power of two.

Method (exact):
  1. per distinct coupling: ground-state set Gamma from the raw 12 couplings (256-state enumeration);
     "pairwise-affine" if Gamma equals the solution set of its constant pairwise xors (|Gamma| = 2^#classes);
     otherwise the cell is a "search" cell;
  2. each pairwise-affine cell contributes one node per lock class; every vertex joins the class nodes of its two
     cells with a parity; parity-0 joins by scipy connected components, parity-1 joins by a parity union-find;
  3. search cells: exact backtracking over their ground states, variables = union-find roots (or bare vertices);
  4. a solution is expanded to a full spin state and re-verified: every cell at a local minimum, and the total
     torus energy recomputed edge by edge equals E0 = sum_C min H_C.
"""
import itertools

import numpy as np


def connected_components_np(n, src, dst):
    """Connected components by root hooking + pointer jumping (int32, low memory).  Returns (ncomp, labels)."""
    lab = np.arange(n, dtype=np.int32)
    while True:
        ls, ld = lab[src], lab[dst]
        diff = ls != ld
        if not diff.any():
            break
        ls, ld = ls[diff], ld[diff]
        np.minimum.at(lab, np.maximum(ls, ld), np.minimum(ls, ld))
        del ls, ld, diff
        while True:
            nl = lab[lab]
            if np.array_equal(nl, lab):
                break
            lab = nl
    roots, labels = np.unique(lab, return_inverse=True)
    return len(roots), labels.astype(np.int32)

CORN = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
FULL = 255
_SP = np.array([[(-1 if (m >> k) & 1 else 1) for k in range(8)] for m in range(256)])
_PR = np.stack([_SP[:, a] * _SP[:, b] for a, b in EDGES], axis=1)


def cell_energy_table(W):
    return -(_PR @ np.asarray(W, dtype=np.int64))       # (256,)


def lock_structure(G):
    """Lock classes (pairwise-constant xor over G), class index and parity of each corner."""
    G = sorted(G)
    g0 = G[0]
    cls, reps = [-1] * 8, []
    for k in range(8):
        for ci, r in enumerate(reps):
            if len({((m >> k) ^ (m >> r)) & 1 for m in G}) == 1:
                cls[k] = ci
                break
        else:
            cls[k] = len(reps)
            reps.append(k)
    par = [((g0 >> k) ^ (g0 >> reps[cls[k]])) & 1 for k in range(8)]
    return cls, par, len(reps)


class ParityUF:
    def __init__(self):
        self.parent, self.par = {}, {}

    def find(self, v):
        if v not in self.parent:
            return v, 0
        acc, p = 0, v
        while self.parent.get(p, p) != p:
            acc ^= self.par[p]
            p = self.parent[p]
        return p, acc

    def union(self, a, b, x):
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == x
        self.parent.setdefault(ra, ra)
        self.parent.setdefault(rb, rb)
        self.par.setdefault(rb, 0)
        self.parent[ra], self.par[ra] = rb, pa ^ pb ^ x
        return True


class Torus:
    def __init__(self, L):
        assert L % 2 == 0 and L >= 4
        self.L = L
        h = L // 2
        self.h = h
        self.ncells = 2 * h ** 3
        i, j, k = np.indices((h, h, h), dtype=np.int32).reshape(3, -1)
        self.ux = np.concatenate([2 * i, 2 * i + 1])
        self.uy = np.concatenate([2 * j, 2 * j + 1])
        self.uz = np.concatenate([2 * k, 2 * k + 1])

    def cell_index(self, x, y, z):
        h = self.h
        p = x % 2
        return p * h ** 3 + ((x // 2) * h + (y // 2)) * h + (z // 2)

    def vid(self, x, y, z):
        L = self.L
        return (x % L * L + y % L) * L + z % L

    def corner_vid(self, k):
        d = CORN[k]
        return self.vid(self.ux + d[0], self.uy + d[1], self.uz + d[2])

    def nbr(self, k):
        """Index of the other cell through corner k of every cell (direction a = 2d - 1)."""
        d = CORN[k]
        L = self.L
        return self.cell_index((self.ux + 2 * d[0] - 1) % L, (self.uy + 2 * d[1] - 1) % L,
                               (self.uz + 2 * d[2] - 1) % L)


class Instance:
    def __init__(self, L, W_default, Wcells):
        self.T = Torus(L)
        self.L = L
        types = [tuple(int(x) for x in W_default)]
        tid = {types[0]: 0}
        self.tcell = np.zeros(self.T.ncells, dtype=np.int32)
        for c, W in Wcells.items():
            W = tuple(int(x) for x in W)
            if W not in tid:
                tid[W] = len(types)
                types.append(W)
            self.tcell[self.T.cell_index(*c)] = tid[W]
        self.types = types
        nt = len(types)
        self.E = np.zeros((nt, 256), dtype=np.int64)
        self.isgs = np.zeros((nt, 256), dtype=bool)
        self.gam = []
        self.aff = np.zeros(nt, dtype=bool)
        self.ncls = np.zeros(nt, dtype=np.int32)
        self.cls = np.zeros((nt, 8), dtype=np.int32)
        self.par = np.zeros((nt, 8), dtype=np.int32)
        for t, W in enumerate(types):
            e = cell_energy_table(W)
            self.E[t] = e
            G = frozenset(np.nonzero(e == e.min())[0].tolist())
            self.gam.append(G)
            self.isgs[t, sorted(G)] = True
            cl, pa, nc = lock_structure(G)
            if len(G) == 2 ** nc:
                self.aff[t] = True
                self.ncls[t] = nc
                self.cls[t] = cl
                self.par[t] = pa
        self.E0 = int(sum(self.E[t].min() * cnt for t, cnt in enumerate(np.bincount(self.tcell, minlength=nt))))

    # ------------------------------------------------------------------ stage 1-2: affine part
    def solve_affine(self):
        T, tc = self.T, self.tcell
        ncl_cell = self.ncls[tc]
        base = np.zeros(T.ncells + 1, dtype=np.int32)
        np.cumsum(ncl_cell, out=base[1:])
        del ncl_cell
        self.base = base
        nnodes = int(base[-1])
        src0, dst0, s1, d1 = [], [], [], []
        aff_cell = self.aff[tc]
        for k in (4, 5, 6, 7):
            c2 = T.nbr(k).astype(np.int32)
            both = aff_cell & self.aff[tc[c2]]
            ia = np.nonzero(both)[0].astype(np.int32)
            ib = c2[ia]
            del c2, both
            ta, tb = tc[ia], tc[ib]
            a = base[ia] + self.cls[ta, k]
            b = base[ib] + self.cls[tb, 7 - k]
            pr = self.par[ta, k] ^ self.par[tb, 7 - k]
            src0.append(a[pr == 0]); dst0.append(b[pr == 0])
            s1.append(a[pr == 1]); d1.append(b[pr == 1])
            del ia, ib, ta, tb, a, b, pr
        src0 = np.concatenate(src0)
        dst0 = np.concatenate(dst0)
        self.ncomp, self.comp = connected_components_np(nnodes, src0, dst0)
        del src0, dst0
        self.uf = ParityUF()
        s1 = np.concatenate(s1)
        d1 = np.concatenate(d1)
        self.n_par1 = len(s1)
        for a, b in zip(self.comp[s1].tolist(), self.comp[d1].tolist()):
            if not self.uf.union(a, b, 1):
                return False
        return True

    # ------------------------------------------------------------------ stage 3: search cells
    def ref_of(self, c, k):
        """Variable reference for corner k of cell c, seen from an affine cell: (key, parity)."""
        t = self.tcell[c]
        if self.aff[t]:
            node = int(self.base[c] + self.cls[t, k])
            r, p = self.uf.find(int(self.comp[node]))
            return ("c", r), p ^ int(self.par[t, k])
        return None

    def corner_ref(self, c, k):
        T = self.T
        r = self.ref_of(c, k)
        if r is not None:
            return r
        u = (int(T.ux[c]), int(T.uy[c]), int(T.uz[c]))
        d = CORN[k]
        u2 = tuple((u[i] + 2 * d[i] - 1) % self.L for i in range(3))
        c2 = int(T.cell_index(*u2))
        r = self.ref_of(c2, 7 - k)
        if r is not None:
            return r
        v = (u[0] + d[0], u[1] + d[1], u[2] + d[2])
        return ("v", int(T.vid(*v))), 0

    def vertex_ref(self, v):
        """Reference of a vertex (x,y,z): it is corner k in {0..7} of the cell u = v - d with equal parities."""
        x, y, z = v
        for k in range(8):
            d = CORN[k]
            u = ((x - d[0]) % self.L, (y - d[1]) % self.L, (z - d[2]) % self.L)
            if u[0] % 2 == u[1] % 2 == u[2] % 2:
                return self.corner_ref(int(self.T.cell_index(*u)), k)
        raise AssertionError

    def prepare_search(self):
        sc = np.nonzero(~self.aff[self.tcell])[0]
        self.search_cells = []
        for c in sc.tolist():
            refs = [self.corner_ref(c, k) for k in range(8)]
            self.search_cells.append((c, refs, sorted(self.gam[self.tcell[c]])))

    def search(self, pins=None):
        assign = {}
        if pins:
            for v, b in pins.items():
                key, p = self.vertex_ref(v)
                want = b ^ p
                if assign.get(key, want) != want:
                    return None
                assign[key] = want
        cells = self.search_cells
        # order: greedy by overlap with already-ordered variables
        order, seen = [], set(assign)
        rem = list(range(len(cells)))
        while rem:
            best = max(rem, key=lambda i: sum(r[0] in seen for r in cells[i][1]))
            rem.remove(best)
            order.append(best)
            seen.update(r[0] for r in cells[best][1])

        def rec(pos):
            if pos == len(order):
                return True
            _, refs, G = cells[order[pos]]
            for m in G:
                newly, ok = [], True
                for k, (key, p) in enumerate(refs):
                    b = ((m >> k) & 1) ^ p
                    cur = assign.get(key)
                    if cur is None:
                        assign[key] = b
                        newly.append(key)
                    elif cur != b:
                        ok = False
                        break
                if ok and rec(pos + 1):
                    return True
                for key in newly:
                    del assign[key]
            return False

        return dict(assign) if rec(0) else None

    # ------------------------------------------------------------------ expansion and verification
    def expand(self, assign):
        T, tc, L = self.T, self.tcell, self.L
        compval = np.zeros(self.ncomp, dtype=np.int64)
        keys = set(self.uf.parent) | {k[1] for k in assign if k[0] == "c"}
        for cid in keys:
            r, p = self.uf.find(cid)
            compval[cid] = assign.get(("c", r), 0) ^ p
        sigma = np.zeros(L ** 3, dtype=np.int8)
        aff_cell = self.aff[tc]
        idx = np.nonzero(aff_cell)[0]
        for k in range(8):
            vids = T.corner_vid(k)[idx]
            node = self.base[idx] + self.cls[tc[idx], k]
            sigma[vids] = (compval[self.comp[node]] ^ self.par[tc[idx], k]).astype(np.int8)
        for c, refs, _ in self.search_cells:
            u = (int(T.ux[c]), int(T.uy[c]), int(T.uz[c]))
            for k, (key, p) in enumerate(refs):
                d = CORN[k]
                v = int(T.vid(u[0] + d[0], u[1] + d[1], u[2] + d[2]))
                sigma[v] = assign.get(key, 0) ^ p
        return sigma

    def verify(self, sigma):
        """All cells at a local minimum, and total torus energy (edge by edge) equals E0."""
        T, tc, L = self.T, self.tcell, self.L
        m = np.zeros(T.ncells, dtype=np.int32)
        for k in range(8):
            m |= sigma[T.corner_vid(k)].astype(np.int32) << k
        local_ok = bool(self.isgs[tc, m].all())
        Hcell = int(self.E[tc, m].sum())
        # edge-by-edge energy: edge (v, v + e_i); owner cell u = v - d with d_i = 0, d_j = (v_j - v_i) mod 2
        W = np.array(self.types, dtype=np.int64)        # (nt, 12)
        etab = np.array([[EDGE_INDEX.get((a, b), -1) for b in range(8)] for a in range(8)])
        Hedge = 0
        step = 8
        for x0 in range(0, L, step):
            xs = np.arange(x0, min(L, x0 + step), dtype=np.int64)
            x, y, z = [a.reshape(-1) for a in np.meshgrid(xs, np.arange(L), np.arange(L), indexing="ij")]
            v = [x, y, z]
            sv = 1 - 2 * sigma[(x * L + y) * L + z].astype(np.int64)
            for i in range(3):
                d = [np.zeros_like(x) if j == i else (v[j] - v[i]) % 2 for j in range(3)]
                u = [(v[j] - d[j]) % L for j in range(3)]
                c = T.cell_index(u[0], u[1], u[2])
                ka = d[0] + 2 * d[1] + 4 * d[2]
                eidx = etab[ka, ka + (1 << i)]
                assert (eidx >= 0).all()
                w = [(v[j] + (1 if j == i else 0)) % L for j in range(3)]
                sw = 1 - 2 * sigma[(w[0] * L + w[1]) * L + w[2]].astype(np.int64)
                Hedge += int(-(W[tc[c], eidx] * sv * sw).sum())
        return local_ok, Hcell, Hedge


def decide(L, W_default, Wcells, pins=None):
    inst = Instance(L, W_default, Wcells)
    inst.affine_ok = inst.solve_affine()
    if not inst.affine_ok:
        return inst, None
    inst.prepare_search()
    return inst, inst.search(pins)
