"""Independent TILE decider (adapted from independent/my_decider.py; imports nothing construction).

Input: an instance file with L, a table of raw 12-couplings and the coupling type of every cell (the independent check cell order:
parity block, then (x//2, y//2, z//2) row-major).
Method (exact):
  1. per coupling type: ground-state set from the raw couplings (256 states); "pairwise-affine" if it equals the
     solution set of its constant pairwise xors;
  2. affine cells: one node per lock class; each vertex joins the class nodes of its two cells (parity); parity-0
     joins by root hooking + pointer jumping (numpy), parity-1 joins by a parity union-find;
  3. the other cells are table constraints over boolean variables (component roots or bare vertices); exact DPLL
     with generalised arc consistency (forced values propagated, chronological backtracking, complete);
     without pins the global spin flip is factored out (every ground-state set is complement-closed: checked);
  4. a solution is expanded to a full spin state and re-verified: every cell at a local minimum, and the torus
     energy recomputed edge by edge equals E0 = sum_C min H_C.
Run: python -I own_decider.py INST.npz [PINS=1]"""
import itertools
import json
import os
import resource
import sys
import time

import numpy as np

CORN = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
_SP = np.array([[(-1 if (m >> k) & 1 else 1) for k in range(8)] for m in range(256)])
_PR = np.stack([_SP[:, a] * _SP[:, b] for a, b in EDGES], axis=1)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def connected_components_np(n, src, dst):
    lab = np.arange(n, dtype=np.int32)
    it = 0
    while True:
        it += 1
        ls, ld = lab[src], lab[dst]
        diff = ls != ld
        if not diff.any():
            break
        ls, ld = ls[diff], ld[diff]
        del diff
        np.minimum.at(lab, np.maximum(ls, ld), np.minimum(ls, ld))
        del ls, ld
        while True:
            nl = lab[lab]
            if np.array_equal(nl, lab):
                break
            lab = nl
    roots, labels = np.unique(lab, return_inverse=True)
    return len(roots), labels.astype(np.int32), it


def lock_structure(G):
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
        self.parent[ra], self.par[ra] = rb, pa ^ pb ^ x
        return True


class Torus:
    def __init__(self, L):
        assert L % 2 == 0 and L >= 4
        self.L, self.h = L, L // 2
        h = self.h
        self.ncells = 2 * h ** 3
        i, j, k = np.indices((h, h, h), dtype=np.int32).reshape(3, -1)
        self.ux = np.concatenate([2 * i, 2 * i + 1])
        self.uy = np.concatenate([2 * j, 2 * j + 1])
        self.uz = np.concatenate([2 * k, 2 * k + 1])

    def cell_index(self, x, y, z):
        h = self.h
        return (x % 2) * h ** 3 + ((x // 2) * h + (y // 2)) * h + (z // 2)

    def vid(self, x, y, z):
        L = self.L
        return (x % L * L + y % L) * L + z % L

    def corner_vid(self, k):
        d = CORN[k]
        return self.vid(self.ux + d[0], self.uy + d[1], self.uz + d[2])

    def nbr(self, k):
        d = CORN[k]
        L = self.L
        return self.cell_index((self.ux + 2 * d[0] - 1) % L, (self.uy + 2 * d[1] - 1) % L,
                               (self.uz + 2 * d[2] - 1) % L)


class Instance:
    def __init__(self, L, types, tcell):
        self.T = Torus(L)
        self.L = L
        self.types = [tuple(int(x) for x in W) for W in types]
        self.tcell = tcell.astype(np.int32)
        assert self.tcell.size == self.T.ncells
        nt = len(self.types)
        self.E = np.zeros((nt, 256), dtype=np.int64)
        self.isgs = np.zeros((nt, 256), dtype=bool)
        self.gam, self.aff = [], np.zeros(nt, dtype=bool)
        self.ncls = np.zeros(nt, dtype=np.int32)
        self.cls = np.zeros((nt, 8), dtype=np.int32)
        self.par = np.zeros((nt, 8), dtype=np.int32)
        for t, W in enumerate(self.types):
            e = -(_PR @ np.asarray(W, dtype=np.int64))
            self.E[t] = e
            G = frozenset(np.nonzero(e == e.min())[0].tolist())
            assert all((255 ^ m) in G for m in G)  # complement-closed (zero field)
            self.gam.append(G)
            self.isgs[t, sorted(G)] = True
            cl, pa, nc = lock_structure(G)
            if len(G) == 2 ** nc:
                self.aff[t], self.ncls[t], self.cls[t], self.par[t] = True, nc, cl, pa
        cnt = np.bincount(self.tcell, minlength=nt)
        self.E0 = int(sum(int(self.E[t].min()) * int(c) for t, c in enumerate(cnt)))
        self.type_counts = {t: int(c) for t, c in enumerate(cnt)}

    def nbr_slice(self, k, s, e):
        T, L, d = self.T, self.L, CORN[k]
        return T.cell_index((T.ux[s:e] + 2 * d[0] - 1) % L, (T.uy[s:e] + 2 * d[1] - 1) % L,
                            (T.uz[s:e] + 2 * d[2] - 1) % L)

    def edge_chunks(self, chunk):
        """Class-node joins, streamed: every vertex is corner k in {4..7} of exactly one of its two cells.
        Yields (a, b, parity) int64 arrays for cell slices; nothing is stored."""
        tc, n = self.tcell, self.T.ncells
        for k in (4, 5, 6, 7):
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                c2 = self.nbr_slice(k, s, e).astype(np.int64)
                ta, tb = tc[s:e], tc[c2]
                both = self.aff[ta] & self.aff[tb]
                ia = np.nonzero(both)[0]
                ib = c2[ia]
                ta, tb = ta[ia], tb[ia]
                a = self.base[s + ia].astype(np.int64) + self.cls[ta, k]
                b = self.base[ib].astype(np.int64) + self.cls[tb, 7 - k]
                pr = self.par[ta, k] ^ self.par[tb, 7 - k]
                yield a, b, pr

    def solve_affine(self, chunk=1 << 17):
        T, tc = self.T, self.tcell
        base = np.zeros(T.ncells + 1, dtype=np.int32)
        np.cumsum(self.ncls[tc], out=base[1:])
        self.base = base
        nnodes = int(base[-1])
        lab = np.arange(nnodes, dtype=np.int32)
        s1, d1 = [], []
        it = 0
        while True:  # root hooking (max root -> min root) + pointer jumping, edges streamed per chunk
            it += 1
            changed = False
            for a, b, pr in self.edge_chunks(chunk):
                if it == 1 and pr.any():
                    s1.append(a[pr == 1]); d1.append(b[pr == 1])
                a, b = a[pr == 0], b[pr == 0]
                la, lb = lab[a], lab[b]
                dif = la != lb
                if not dif.any():
                    continue
                changed = True
                la, lb = la[dif], lb[dif]
                np.minimum.at(lab, np.maximum(la, lb), np.minimum(la, lb))
                while True:
                    nl = lab[lab]
                    if np.array_equal(nl, lab):
                        break
                    lab = nl
                del nl
            if not changed:
                break
        self.comp = lab  # component id = root node index
        del lab
        self.ncomp = int(np.count_nonzero(self.comp == np.arange(nnodes, dtype=np.int32)))
        self.cc_iters = it
        self.uf = ParityUF()
        s1 = np.concatenate(s1) if s1 else np.zeros(0, dtype=np.int64)
        d1 = np.concatenate(d1) if d1 else np.zeros(0, dtype=np.int64)
        self.n_par1 = int(s1.size)
        for a, b in zip(self.comp[s1].tolist(), self.comp[d1].tolist()):
            if not self.uf.union(a, b, 1):
                return False
        return True

    def ref_of(self, c, k):
        t = self.tcell[c]
        if self.aff[t]:
            r, p = self.uf.find(int(self.comp[int(self.base[c] + self.cls[t, k])]))
            return ("c", r), p ^ int(self.par[t, k])
        return None

    def corner_ref(self, c, k):
        T = self.T
        r = self.ref_of(c, k)
        if r is not None:
            return r
        u = (int(T.ux[c]), int(T.uy[c]), int(T.uz[c]))
        d = CORN[k]
        c2 = int(T.cell_index(*[(u[i] + 2 * d[i] - 1) % self.L for i in range(3)]))
        r = self.ref_of(c2, 7 - k)
        if r is not None:
            return r
        return ("v", int(T.vid(u[0] + d[0], u[1] + d[1], u[2] + d[2]))), 0

    def vertex_ref(self, v):
        x, y, z = v
        for k in range(8):
            d = CORN[k]
            u = ((x - d[0]) % self.L, (y - d[1]) % self.L, (z - d[2]) % self.L)
            if u[0] % 2 == u[1] % 2 == u[2] % 2:
                return self.corner_ref(int(self.T.cell_index(*u)), k)
        raise AssertionError

    # ------------------------------------------------------------------ table-constraint DPLL with GAC
    def prepare_search(self):
        sc = np.nonzero(~self.aff[self.tcell])[0].tolist()
        self.var_id = {}
        self.cells = []
        for c in sc:
            refs = []
            for k in range(8):
                key, p = self.corner_ref(c, k)
                if key not in self.var_id:
                    self.var_id[key] = len(self.var_id)
                refs.append((self.var_id[key], p))
            masks = []
            for m in sorted(self.gam[self.tcell[c]]):  # drop states inconsistent with repeated variables
                seen, ok = {}, True
                for k, (v, p) in enumerate(refs):
                    b = ((m >> k) & 1) ^ p
                    if seen.setdefault(v, b) != b:
                        ok = False
                        break
                if ok:
                    masks.append(m)
            self.cells.append((c, refs, masks))
        nv = len(self.var_id)
        self.watch = [[] for _ in range(nv)]
        for i, (_, refs, _) in enumerate(self.cells):
            for v in {v for v, _ in refs}:
                self.watch[v].append(i)
        self.order = sorted(range(nv), key=lambda v: -len(self.watch[v]))

    def search(self, pins=None, break_symmetry=True):
        nv = len(self.var_id)
        assign = [-1] * nv
        trail = []
        cells, watch = self.cells, self.watch
        stats = dict(decisions=0, conflicts=0, forced=0)

        def setv(v, b, queue):
            assign[v] = b
            trail.append(v)
            queue.append(v)

        def propagate(queue):
            while queue:
                x = queue.pop()
                for i in watch[x]:
                    _, refs, masks = cells[i]
                    known = val = 0
                    for k, (v, p) in enumerate(refs):
                        a = assign[v]
                        if a >= 0:
                            known |= 1 << k
                            val |= (a ^ p) << k
                    andm, orm, any_ = 255, 0, False
                    for m in masks:
                        if (m ^ val) & known == 0:
                            andm &= m
                            orm |= m
                            any_ = True
                    if not any_:
                        return False
                    for k, (v, p) in enumerate(refs):
                        if assign[v] < 0:
                            if (andm >> k) & 1:
                                setv(v, 1 ^ p, queue)
                                stats["forced"] += 1
                            elif not (orm >> k) & 1:
                                setv(v, p, queue)
                                stats["forced"] += 1
            return True

        def undo(n):
            while len(trail) > n:
                assign[trail.pop()] = -1

        q = []
        if pins:
            for v, b in pins.items():
                key, p = self.vertex_ref(v)
                if key not in self.var_id:
                    continue  # vertex in no constraint: a free affine class, any value
                vi = self.var_id[key]
                want = b ^ p
                if assign[vi] >= 0 and assign[vi] != want:
                    return None, stats
                if assign[vi] < 0:
                    setv(vi, want, q)
        elif break_symmetry and nv:
            setv(self.order[0], 0, q)  # global flip symmetry
        if not propagate(q):
            return None, stats
        frames = []  # (var, next value to try, trail length before the decision)
        pos = 0
        while True:
            while pos < nv and assign[self.order[pos]] >= 0:
                pos += 1
            if pos == nv:
                return {key: assign[vi] for key, vi in self.var_id.items()}, stats
            x = self.order[pos]
            frames.append((x, 1, len(trail), pos))
            stats["decisions"] += 1
            q = []
            setv(x, 0, q)
            ok = propagate(q)
            while not ok:
                stats["conflicts"] += 1
                # backtrack: flip the most recent frame that still has a value to try
                while frames:
                    x, nxt, tl, p0 = frames.pop()
                    undo(tl)
                    if nxt == 1:
                        frames.append((x, 2, tl, p0))
                        q = []
                        setv(x, 1, q)
                        pos = p0
                        ok = propagate(q)
                        break
                else:
                    return None, stats

    def expand(self, assign):
        T, tc, L = self.T, self.tcell, self.L
        compval = np.zeros(self.comp.size, dtype=np.int8)
        for key, b in assign.items():
            if key[0] == "c":
                compval[key[1]] = b
        for cid in list(self.uf.parent):
            r, p = self.uf.find(cid)
            compval[cid] = compval[r] ^ p
        sigma = np.zeros(L ** 3, dtype=np.int8)
        n, ch = T.ncells, 1 << 17
        for s0 in range(0, n, ch):
            e0 = min(n, s0 + ch)
            idx = s0 + np.nonzero(self.aff[tc[s0:e0]])[0]
            for k in range(8):
                d = CORN[k]
                vids = T.vid(T.ux[idx] + d[0], T.uy[idx] + d[1], T.uz[idx] + d[2])
                node = self.base[idx] + self.cls[tc[idx], k]
                sigma[vids] = (compval[self.comp[node]] ^ self.par[tc[idx], k]).astype(np.int8)
                del vids, node
            del idx
        for c, refs, _ in self.cells:
            u = (int(T.ux[c]), int(T.uy[c]), int(T.uz[c]))
            for k, (vi, p) in enumerate(refs):
                d = CORN[k]
                v = int(T.vid(u[0] + d[0], u[1] + d[1], u[2] + d[2]))
                sigma[v] = self._val(vi, assign) ^ p
        return sigma

    def _val(self, vi, assign):
        if not hasattr(self, "_inv"):
            self._inv = {i: k for k, i in self.var_id.items()}
        return assign[self._inv[vi]]

    def verify(self, sigma):
        T, tc, L = self.T, self.tcell, self.L
        local_ok, Hcell = True, 0
        n, ch = T.ncells, 1 << 17
        for s0 in range(0, n, ch):
            e0 = min(n, s0 + ch)
            m = np.zeros(e0 - s0, dtype=np.int32)
            for k in range(8):
                d = CORN[k]
                m |= sigma[T.vid(T.ux[s0:e0] + d[0], T.uy[s0:e0] + d[1], T.uz[s0:e0] + d[2])].astype(np.int32) << k
            local_ok &= bool(self.isgs[tc[s0:e0], m].all())
            Hcell += int(self.E[tc[s0:e0], m].sum())
            del m
        W = np.array(self.types, dtype=np.int64)
        etab = np.array([[EDGE_INDEX.get((a, b), -1) for b in range(8)] for a in range(8)])
        Hedge = 0
        for x0 in range(0, L, 2):
            xs = np.arange(x0, min(L, x0 + 2), dtype=np.int64)
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


def nae(clauses, a):
    return all(len({a[v] ^ ng for v, ng in cl}) > 1 for cl in clauses)


def main(path):
    t0 = time.time()
    z = np.load(path)
    L = int(z["L"])
    meta = json.loads(str(z["meta"]))
    clauses = [[tuple(l) for l in cl] for cl in meta["clauses"]]
    n = meta["n_vars"]
    rep = z["rep"]
    netv = list(zip(z["net_vid"].tolist(), z["net_lab"].tolist())) if "net_vid" in z.files else []
    inst = Instance(L, z["types"], z["tcell"])
    del z
    cls_count = {}
    for t, c in inst.type_counts.items():
        nm = "affine" if inst.aff[t] else "search"
        cls_count[nm] = cls_count.get(nm, 0) + c
    print(f"[{meta['name']}] L={L} cells={inst.T.ncells} types={len(inst.types)} {cls_count} E0={inst.E0}"
          f"  [{time.time()-t0:.1f}s RSS {rss():.0f}MB]", flush=True)
    t1 = time.time()
    ok = inst.solve_affine()
    print(f"  affine: consistent={ok} components={inst.ncomp} cc_iters={inst.cc_iters} parity1_joins={inst.n_par1}"
          f"  [{time.time()-t1:.1f}s RSS {rss():.0f}MB]", flush=True)
    sat = any(nae(clauses, a) for a in itertools.product((0, 1), repeat=n))
    if not ok:
        print(f"  TILE planted=False (affine contradiction); NAE-sat={sat}; agree={not sat}", flush=True)
        return
    t1 = time.time()
    inst.prepare_search()
    print(f"  search cells={len(inst.cells)} variables={len(inst.var_id)}  [{time.time()-t1:.1f}s RSS {rss():.0f}MB]",
          flush=True)
    t1 = time.time()
    sol, st = inst.search()
    planted = sol is not None
    print(f"  TILE planted={planted}; NAE-sat={sat}; agree={planted == sat}; search {st}  [{time.time()-t1:.1f}s]",
          flush=True)
    if planted:
        t1 = time.time()
        sigma = inst.expand(sol)
        local_ok, Hc, He = inst.verify(sigma)
        a = [int(sigma[inst.T.vid(*rep[v])]) if rep[v, 0] >= 0 else 0 for v in range(n)]
        nv_ = netv
        const = {}
        for vid_, lab_ in nv_:
            const.setdefault(int(lab_), set()).add(int(sigma[vid_]))
        n_const = sum(len(x) == 1 for x in const.values())
        print(f"  read-off: sigma constant on {n_const}/{len(const)} networks ({len(nv_)} link vertices)", flush=True)
        print(f"  certificate: all cells at local minimum={local_ok}; H(cells)={Hc}; H(edges)={He}; E0={inst.E0};"
              f" ground state={local_ok and He == inst.E0}; read-off {a} NAE-satisfies={nae(clauses, a)}"
              f"  [{time.time()-t1:.1f}s RSS {rss():.0f}MB]", flush=True)
        del sigma
    if os.environ.get("PINS", "1") == "1":
        t1 = time.time()
        agree, tot, bad, pinned_gs = 0, 0, [], 0
        for a in itertools.product((0, 1), repeat=n):
            pins = {tuple(int(x) for x in rep[v]): a[v] for v in range(n) if rep[v, 0] >= 0}
            s, _ = inst.search(pins=pins)
            tot += 1
            if s is not None:  # re-verify every pinned solution as a full ground state
                sig = inst.expand(s)
                lok, _, He = inst.verify(sig)
                got = [int(sig[inst.T.vid(*rep[v])]) for v in range(n) if rep[v, 0] >= 0]
                if not (lok and He == inst.E0 and got == [a[v] for v in range(n) if rep[v, 0] >= 0]):
                    bad.append(("unverified", a))
                const = {}
                for vid_, lab_ in netv:
                    const.setdefault(int(lab_), set()).add(int(sig[vid_]))
                if any(len(x) != 1 for x in const.values()):
                    bad.append(("network not constant", a))
                pinned_gs += 1
                del sig
            if (s is not None) == nae(clauses, a):
                agree += 1
            else:
                bad.append(a)
        print(f"  pinned assignments: TILE(pinned) == NAE(a) for {agree}/{tot}; {pinned_gs} pinned ground states"
              f" re-verified (local minima, E0, read-off = a, every network constant); mismatches {bad}"
              f"  [{time.time()-t1:.1f}s RSS {rss():.0f}MB]", flush=True)
    print(f"  total {time.time()-t0:.1f}s, peak RSS {rss():.0f}MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
