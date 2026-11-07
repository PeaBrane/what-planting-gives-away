"""Local (finite) lemmas for the G46 construction at S = 12 (every block has the same background phase, since
12 = 0 mod 3).

(L1) Shell lemma: on the 2x2x2-block torus (L = 24), remove from every block every possible outer crossing cell
     (6 faces x 2 travel directions); the remaining *shell* cells (cells in an extreme layer of their block) are
     background F42 cells, and their plate locks join all their corners into ONE component.
(L2) Template lemma: for each of the 30 sparse templates in a block whose shell background is anchored, the
     propagated anchoring (background locks inside the block, then F6 cells whose anchored corners force
     {0, 1_D}) pins every template cell, for some choice of F6 gauges (found greedily, then re-verified).
(L3) Cluster lemma: the same inside the G46 cluster region, with the region's extreme layers anchored.
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
import templates46 as T6  # noqa: E402
from tiles import CORNERS, DIR_TO_K, DIRS, gs, mask  # noqa: E402

assert B.S == 12, "run with TILE_S=12"
S, H = B.S, B.H


class DUF:
    def __init__(self):
        self.p = {}

    def find(self, v):
        self.p.setdefault(v, v)
        while self.p[v] != v:
            self.p[v] = self.p[self.p[v]]
            v = self.p[v]
        return v

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def corner(u, k, L=None):
    v = tuple(x + d for x, d in zip(u, CORNERS[k]))
    return tuple(x % L for x in v) if L else v


def plate_locks(u, W):
    """Pairs of corners locked equal by an affine F42 cell in ferromagnetic gauge (lock classes of Gamma)."""
    Gs = sorted(gs(W))
    out = []
    for k in range(1, 8):
        for j in range(k):
            if len({((m >> k) ^ (m >> j)) & 1 for m in Gs}) == 1:
                assert all(((m >> k) ^ (m >> j)) & 1 == 0 for m in Gs)
                out.append((j, k))
                break
    return out


def shell_lemma():
    L = 2 * S
    cells = [u for u in itertools.product(range(L), repeat=3) if u[0] % 2 == u[1] % 2 == u[2] % 2]

    def off(u):
        return [((x + H) % S) - H for x in u]  # offset from the nearest block centre, in [-H, H)

    shell = [u for u in cells if any(o in (-H, H - 1) for o in off(u))]
    crossing = set()
    for g in itertools.product(range(2), repeat=3):
        C = tuple(S * x for x in g)
        for f in B.FACES6:
            for tau in (1, -1):
                o, _ = B.boundary_pair(C, f, tau)
                crossing.add(tuple(x % L for x in o))
    keep = [u for u in shell if u not in crossing]
    uf = DUF()
    for u in keep:
        for j, k in plate_locks(u, G.bg_coupling(u)):
            uf.union(corner(u, j, L), corner(u, k, L))
    roots = {uf.find(corner(u, k, L)) for u in keep for k in range(8)}
    return dict(L=L, shell_cells=len(shell), crossing_removed=len(crossing & set(shell)),
                kept=len(keep), components=len(roots))


def propagate(cells_f6, links, bg_cells, anchored_vertices, extra_affine=()):
    """cells_f6: {cell: domino mask} wire cells; links: {cell: set of link corner indices}; bg_cells: affine F42
    cells (background or fan-outs) with their couplings; anchored_vertices: vertices known 0.  Returns (pinned,
    unpinned, gauges)."""
    uf = DUF()
    for u, W in bg_cells.items():
        for j, k in plate_locks(u, W):
            uf.union(corner(u, j), corner(u, k))
    anch = {uf.find(v) for v in anchored_vertices}
    pending = dict(cells_f6)
    gauges = {}
    while pending:
        newly = []
        for c, D in pending.items():
            Pm = 0
            for k in range(8):
                if k not in links[c] and uf.find(corner(c, k)) in anch:
                    Pm |= 1 << k
            W = G.self_pinnable(D, Pm)
            if W is not None:
                newly.append((c, W))
        if not newly:
            break
        for c, W in newly:
            gauges[c] = W
            del pending[c]
            for k in range(8):
                if k not in links[c]:
                    anch.add(uf.find(corner(c, k)))
    return len(gauges), sorted(pending), gauges


def template_lemma():
    T = T6.templates46()
    C = (0, 0, 0)
    res = {}
    for (f1, f2), cells in sorted(T.items()):
        idx = {c: t for t, c in enumerate(cells)}
        links = {}
        for t, c in enumerate(cells):
            ks = set()
            if t > 0:
                ks.add(DIR_TO_K[B.vsub(cells[t - 1], c)])
            else:
                ks.add(DIR_TO_K[B.neg(B.cross_step(f1, B.entry_tau(f1)))])
            if t < len(cells) - 1:
                ks.add(DIR_TO_K[B.vsub(cells[t + 1], c)])
            else:
                ks.add(DIR_TO_K[B.cross_step(f2, f2[1])])
            links[c] = ks
        f6 = {c: mask(links[c]) for c in cells}
        # background: every other cell of the block and of the 26 neighbouring blocks' facing layers
        bg = {}
        anchored = set()
        for d in itertools.product(range(-H - 2, H + 2), repeat=3):
            u = B.vadd(C, d)
            if not (u[0] % 2 == u[1] % 2 == u[2] % 2) or u in idx:
                continue
            inside = B.in_block(C, u)
            if not inside and any(abs(x) > H for x in d):
                pass
            bg[u] = G.bg_coupling(u)
            ext = (not inside) or any(x in (-H, H - 1) for x in d)
            if ext:
                # shell background (own extreme layer, or a neighbouring block's): anchored by the shell lemma.
                # The neighbours' linked crossing cells are F6 wire cells, not background: exclude them.
                for k in range(8):
                    anchored.add(corner(u, k))
        # the neighbouring blocks' crossing cells linked to our outer cells are wires, not background
        for f, tau in ((f1, B.entry_tau(f1)), (f2, f2[1])):
            i, sg = f
            o2, _ = B.boundary_pair(B.vsc(B.unit(i), S * sg), (i, -sg), tau)
            bg.pop(o2, None)
            for k in range(8):
                anchored.discard(corner(o2, k))
        npin, unp, _ = propagate(f6, links, bg, anchored)
        res[f1, f2] = (len(cells), npin, unp)
    return res


def cluster_lemma():
    lk, assign, wires, u = G.cluster_template46()
    R = B.cluster_region((0, 0, 0))
    links = {c: {DIR_TO_K[a] for a in lk.net[c]} for c in lk.used}
    f6 = {}
    bg = {}
    for c in lk.used:
        r = lk.role[c]
        if r.startswith("clause"):
            continue
        if r.startswith("side"):
            m = mask(links[c])
            bg[c] = B.coupling_with_gs({0, m, 0xFF ^ m, 0xFF}, "F42")
        else:
            f6[c] = mask(links[c])
    # socket outer cells link upward to the socket blocks: add that link corner
    for ij, w in wires.items():
        o = w[-1]
        assert len(links[o]) == 1
        tau = ij[1]
        links[o].add(DIR_TO_K[B.cross_step((2, 1), tau)] if tau == 1 else DIR_TO_K[B.neg(B.cross_step((2, 1), tau))])
        f6[o] = mask(links[o])
    anchored = set()
    (x0, x1), (y0, y1), (z0, z1) = R
    for d in itertools.product(range(x0 - 1, x1 + 1), range(y0 - 1, y1 + 1), range(z0 - 1, z1 + 1)):
        c = d
        if not (c[0] % 2 == c[1] % 2 == c[2] % 2) or c in lk.used:
            continue
        inside = B.in_region(c, R)
        bg[c] = G.bg_coupling(c)
        if (not inside) or c[0] in (x0, x1 - 1) or c[1] in (y0, y1 - 1) or c[2] in (z0, z1 - 1):
            for k in range(8):
                anchored.add(corner(c, k))
    # the socket blocks' outer cells (linked to our socket outer cells) are wires: not background
    for ij, w in wires.items():
        o = w[-1]
        tau = ij[1]
        step = B.cross_step((2, 1), tau) if tau == 1 else B.neg(B.cross_step((2, 1), tau))
        o2 = B.vadd(o, step)
        bg.pop(o2, None)
        for k in range(8):
            anchored.discard(corner(o2, k))
    npin, unp, _ = propagate(f6, links, bg, anchored)
    return dict(cluster_cells=len(lk.used), f6_wire_cells=len(f6), pinned=npin, unpinned=unp[:5],
                n_unpinned=len(unp))


if __name__ == "__main__":
    t0 = time.time()
    print("(L1) shell lemma:", shell_lemma(), f"[{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    res = template_lemma()
    bad = {k: v for k, v in res.items() if v[2]}
    print(f"(L2) template lemma: {len(res)} templates; all cells pinned in {len(res) - len(bad)};"
          f" failures: {bad}  [{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    print("(L3) cluster lemma:", cluster_lemma(), f"[{time.time()-t0:.1f}s]", flush=True)
