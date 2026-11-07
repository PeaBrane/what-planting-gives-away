"""Non-periodic versions of the G46 foam lemmas at S = 12 (no torus-cover subtleties).

(P) Pair lemma: for blocks g, g' = g + e_i (i = x, y, z), in the foam of the two blocks alone (their extreme-layer
    cells minus all possible crossing cells), every reference vertex v of g's foam that lies in g's own giant
    component is lock-connected to v + S e_i.  Then the giant foam components of any face-connected set of blocks
    form one component.
(R) Region lemma: the G46 cluster region (6x4x2 blocks) inside a non-periodic box with a one-block ring of foam
    blocks around it: propagated anchoring from the ring's giant component pins all F6 cells of the cluster.
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
from g46_local import DUF, plate_locks  # noqa: E402
from tiles import CORNERS, DIR_TO_K, mask  # noqa: E402

assert B.S == 12, "run with TILE_S=12"
S, H = B.S, B.H


def corner(u, k):
    return tuple(x + d for x, d in zip(u, CORNERS[k]))


def iscell(u):
    return u[0] % 2 == u[1] % 2 == u[2] % 2


def block_foam(g):
    C = tuple(S * x for x in g)
    crossing = {B.boundary_pair(C, f, tau)[0] for f in B.FACES6 for tau in (1, -1)}
    out = []
    for d in itertools.product(range(-H, H), repeat=3):
        if any(x in (-H, H - 1) for x in d):
            u = B.vadd(C, d)
            if iscell(u) and u not in crossing:
                out.append(u)
    return out


def lock(uf, cells):
    for u in cells:
        for j, k in plate_locks(u, G.bg_coupling(u)):
            uf.union(corner(u, j), corner(u, k))


def comps(uf, cells):
    cnt = {}
    for u in cells:
        for k in range(8):
            r = uf.find(corner(u, k))
            cnt[r] = cnt.get(r, 0) + 1
    return cnt


def pair_lemma():
    out = {}
    g = (0, 0, 0)
    fg = block_foam(g)
    uf1 = DUF()
    lock(uf1, fg)
    c1 = comps(uf1, fg)
    big1 = max(c1, key=c1.get)
    refs = [corner(u, k) for u in fg for k in range(8) if uf1.find(corner(u, k)) == big1]
    for i in range(3):
        g2 = tuple(S * (1 if j == i else 0) for j in range(3))
        f2 = block_foam(tuple(1 if j == i else 0 for j in range(3)))
        uf = DUF()
        lock(uf, fg + f2)
        bad = [v for v in refs if uf.find(v) != uf.find(B.vadd(v, g2))]
        out["xyz"[i]] = dict(single_block_giant_incidences=c1[big1], single_block_components=len(c1),
                             refs=len(refs), refs_not_joined_to_translate=len(bad))
    return out


def region_lemma():
    lk, assign, wires, u = G.cluster_template46()
    R = B.cluster_region((0, 0, 0))
    # region blocks: coarse x 0..5, y 0..3, z -1..0 ; ring: x -1..6, y -1..4, z -2..1
    ring = [g for g in itertools.product(range(-1, 7), range(-1, 5), range(-2, 2))
            if not (0 <= g[0] <= 5 and 0 <= g[1] <= 3 and -1 <= g[2] <= 0)]
    foam = []
    for g in ring:
        foam += block_foam(g)
    links = {c: {DIR_TO_K[a] for a in lk.net[c]} for c in lk.used}
    for ij, w in wires.items():
        tau = ij[1]
        step = B.cross_step((2, 1), tau) if tau == 1 else B.neg(B.cross_step((2, 1), tau))
        links[w[-1]].add(DIR_TO_K[step])
    f6 = {}
    fan = []
    for c in lk.used:
        r = lk.role[c]
        if r.startswith("clause"):
            continue
        if r.startswith("side"):
            fan.append(c)
        else:
            assert len(links[c]) == 2
            f6[c] = mask(links[c])
    region_bg = [c for c in itertools.product(*[range(lo, hi) for lo, hi in R]) if iscell(c) and c not in lk.used]
    uf = DUF()
    lock(uf, foam)
    cnt = comps(uf, foam)
    big = [r for r, c in cnt.items() if c > 64]
    lock(uf, region_bg)
    for c in fan:
        m = mask(links[c])
        W = B.coupling_with_gs({0, m, 0xFF ^ m, 0xFF}, "F42")
        for j, k in plate_locks(c, W):
            uf.union(corner(c, j), corner(c, k))
    anch = {uf.find(r) for r in big}
    pending = dict(f6)
    rounds = 0
    while pending:
        rounds += 1
        newly = []
        for c, D in pending.items():
            Pm = 0
            for k in range(8):
                if k not in links[c] and uf.find(corner(c, k)) in anch:
                    Pm |= 1 << k
            if G.self_pinnable(D, Pm) is not None:
                newly.append(c)
        if not newly:
            break
        for c in newly:
            del pending[c]
            for k in range(8):
                if k not in links[c]:
                    anch.add(uf.find(corner(c, k)))
    return dict(ring_blocks=len(ring), ring_foam_giant_components=len(big), f6_cells=len(f6),
                unpinned=len(pending), rounds=rounds)


if __name__ == "__main__":
    t0 = time.time()
    print("(P) pair lemma:", pair_lemma(), f"[{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    print("(R) region lemma:", region_lemma(), f"[{time.time()-t0:.1f}s]", flush=True)
