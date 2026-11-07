"""Local lemmas for G46 at S = 12 with worst-case environments (supersedes the shell part of g46_local.py, whose
thin-shell 'all shell corners anchored' assumption is false: the shell foam has one giant lock component plus
isolated single plates).

Foam F: on a torus of blocks, every cell in an extreme layer of its block, minus every possible outer crossing cell
(6 faces x 2 travel directions per block).  These cells are background in every layout (template condition C1),
whatever the blocks contain.
(L1) The lock graph of F (F42 plates) has exactly one component with more than 64 corner incidences, on the
     4x4x4-block torus (L = 48) -- so the giant component does not split under a larger cover (2x2x2 vs 4x4x4).
(L2) For each of the 30 sparse templates in block (1,1,1) of that torus, with that block's interior background
     added: propagated anchoring from the giant foam component pins every template cell.
(L3) For the G46 cluster region placed in an 8x6x4-block torus (region = 6x4x2 blocks, all its cells known), with
     the foam outside the region: propagated anchoring from the giant component pins every F6 cell of the cluster.
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
import templates46 as T6  # noqa: E402
from g46_local import DUF, plate_locks  # noqa: E402
from tiles import CORNERS, DIR_TO_K, mask  # noqa: E402

assert B.S == 12, "run with TILE_S=12"
S, H = B.S, B.H


def corner(u, k, P):
    return tuple((x + d) % p for x, d, p in zip(u, CORNERS[k], P))


def iscell(u):
    return u[0] % 2 == u[1] % 2 == u[2] % 2


def foam(nb, exclude_blocks=()):
    """Foam cells on a torus of nb = (nx, ny, nz) blocks; blocks in exclude_blocks contribute nothing."""
    P = tuple(S * n for n in nb)
    crossing = set()
    for g in itertools.product(*[range(n) for n in nb]):
        C = tuple(S * x for x in g)
        for f in B.FACES6:
            for tau in (1, -1):
                o, _ = B.boundary_pair(C, f, tau)
                crossing.add(tuple(x % p for x, p in zip(o, P)))
    ex = set(exclude_blocks)
    cells = []
    for g in itertools.product(*[range(n) for n in nb]):
        if g in ex:
            continue
        C = tuple(S * x for x in g)
        for d in itertools.product(range(-H, H), repeat=3):
            if not any(x in (-H, H - 1) for x in d):
                continue
            u = tuple((c + x) % p for c, x, p in zip(C, d, P))
            if iscell(u) and u not in crossing:
                cells.append(u)
    return P, cells


def lock(uf, cells, P, W_of):
    for u in cells:
        for j, k in plate_locks(u, W_of(u)):
            uf.union(corner(u, j, P), corner(u, k, P))


def giant(uf, cells, P, thresh=64):
    cnt = {}
    for u in cells:
        for k in range(8):
            r = uf.find(corner(u, k, P))
            cnt[r] = cnt.get(r, 0) + 1
    big = [r for r, c in cnt.items() if c > thresh]
    return big, cnt


def propagate(uf, anch, f6, links, P):
    pending = dict(f6)
    n = 0
    while pending:
        newly = []
        for c, D in pending.items():
            Pm = 0
            for k in range(8):
                if k not in links[c] and uf.find(corner(c, k, P)) in anch:
                    Pm |= 1 << k
            if G.self_pinnable(D, Pm) is not None:
                newly.append(c)
        if not newly:
            break
        for c in newly:
            del pending[c]
            n += 1
            for k in range(8):
                if k not in links[c]:
                    anch.add(uf.find(corner(c, k, P)))
    return n, sorted(pending)


def L1():
    P, cells = foam((4, 4, 4))
    uf = DUF()
    lock(uf, cells, P, G.bg_coupling)
    big, cnt = giant(uf, cells, P)
    sizes = sorted(cnt.values(), reverse=True)
    return dict(P=P, foam_cells=len(cells), components=len(cnt), giant=len(big), sizes_top=sizes[:3],
                largest_small=max([c for r, c in cnt.items() if r not in big], default=0)), (P, cells, uf, big)


def L2(env):
    P, cells, uf0, big = env
    T = T6.templates46()
    C = (S, S, S)
    fails = {}
    for (f1, f2), tc in sorted(T.items()):
        uf = DUF()
        uf.p = dict(uf0.p)
        cells_t = [B.vadd(C, c) for c in tc]
        idx = set(cells_t)
        links = {}
        for t, c in enumerate(cells_t):
            ks = set()
            ks.add(DIR_TO_K[B.vsub(cells_t[t - 1], c)] if t > 0 else DIR_TO_K[B.neg(B.cross_step(f1, B.entry_tau(f1)))])
            ks.add(DIR_TO_K[B.vsub(cells_t[t + 1], c)] if t < len(cells_t) - 1 else DIR_TO_K[B.cross_step(f2, f2[1])])
            links[c] = ks
        interior = []
        for d in itertools.product(range(-H + 1, H - 1), repeat=3):
            u = B.vadd(C, d)
            if iscell(u) and u not in idx:
                interior.append(u)
        lock(uf, interior, P, G.bg_coupling)
        anch = {uf.find(r) for r in big}
        f6 = {c: mask(links[c]) for c in cells_t}
        n, unp = propagate(uf, anch, f6, links, P)
        if unp:
            fails[f1, f2] = unp
    return dict(templates=len(T), failures=fails)


def L3(nb=(8, 6, 4), off=(1, 1, 2)):
    """nb = (6, 4, n) makes the region periodic in x and y with the patch pitch of layout A: the torus then models
    the slab of adjacent cluster regions with foam blocks above and below."""
    lk, assign, wires, u = G.cluster_template46()
    region_blocks = [((off[0] + x) % nb[0], (off[1] + y) % nb[1], off[2] - 1 + z)
                     for x in range(6) for y in range(4) for z in range(2)]
    P, cells = foam(nb, exclude_blocks=region_blocks)
    shift = tuple(S * o for o in off)
    used = {tuple((c + s) % p for c, s, p in zip(x, shift, P)): x for x in lk.used}
    R = B.cluster_region((0, 0, 0))
    region_cells = []
    for d in itertools.product(*[range(lo, hi) for lo, hi in R]):
        c = tuple((x + s) % p for x, s, p in zip(d, shift, P))
        if iscell(c) and c not in used:
            region_cells.append(c)
    links = {}
    f6 = {}
    fan = []
    for c, c0 in used.items():
        r = lk.role[c0]
        links[c] = {DIR_TO_K[a] for a in lk.net[c0]}
        if r.startswith("clause"):
            continue
        if r.startswith("side"):
            fan.append((c, mask(links[c])))
        else:
            f6[c] = None
    for ij, w in wires.items():
        o = tuple((x + s) % p for x, s, p in zip(w[-1], shift, P))
        tau = ij[1]
        step = B.cross_step((2, 1), tau) if tau == 1 else B.neg(B.cross_step((2, 1), tau))
        links[o].add(DIR_TO_K[step])
    for c in f6:
        assert len(links[c]) == 2, (c, links[c])
        f6[c] = mask(links[c])
    uf = DUF()
    lock(uf, cells, P, G.bg_coupling)
    big, _ = giant(uf, cells, P)
    lock(uf, region_cells, P, G.bg_coupling)
    for c, m in fan:
        W = B.coupling_with_gs({0, m, 0xFF ^ m, 0xFF}, "F42")
        for j, k in plate_locks(c, W):
            uf.union(corner(c, j, P), corner(c, k, P))
    anch = {uf.find(r) for r in big}
    n, unp = propagate(uf, anch, f6, links, P)
    return dict(torus_blocks=nb, foam_cells=len(cells), giant_components=len(big), region_bg_cells=len(region_cells),
                f6_cells=len(f6), pinned=n, unpinned=len(unp), example=unp[:3])


if __name__ == "__main__":
    t0 = time.time()
    r1, env = L1()
    print("(L1) foam lemma:", r1, f"[{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    print("(L2) template lemma:", L2(env), f"[{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    print("(L3) cluster lemma, isolated region (8x6x4-block torus):", L3(), f"[{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    print("(L3') cluster lemma, slab of adjacent regions (6x4x6-block torus, region periodic in x, y):",
          L3((6, 4, 6), (0, 0, 2)), f"[{time.time()-t0:.1f}s]", flush=True)
