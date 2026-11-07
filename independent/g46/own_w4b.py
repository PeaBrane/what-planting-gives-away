"""Probe (layout B socket face): the (W4) model with ONE socket template of any shape -- + port
T[(2,-1), f], - port T[f, (2,-1)], f any of the 5 other faces -- and the other five socket blocks foam-only (their
cluster wires end at an absent template).  Worst case otherwise as in (W4).  Window margin MARGIN (default 1:
a sub-model of the margin-2 window, so a pass here implies a pass there by monotonicity).
Run: TILE_S=12 python -I -B own_w4b.py SOCKET_INDEX ..."""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_cube as O  # noqa: E402
import own_geom as Gm  # noqa: E402
from own_geom import DIR_K, S, SLAB, add, sub  # noqa: E402
from own_windows import (cluster_data, end_link, periodic_giant, present_robust_refs, propagate, rss,  # noqa: E402
                         template_links, window_box)

os.environ.setdefault("TILE_S", "12")
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "construction"))
import templates46 as T6  # noqa: E402
import tiles as TF  # noqa: E402

FACES = [(i, s) for i in range(3) for s in (1, -1)]


def w4b(offs, sock, face, margin):
    lk, assign, wires, R, links, roles = cluster_data()
    T = T6.templates46()
    used = set(lk.used)
    blocks = list(itertools.product(range(-margin, 6 + margin), range(-margin, 4 + margin), range(-1 - margin, 4)))
    pres = [g for g in blocks if g[2] not in SLAB]
    lg = Gm.LockGraph(window_box(blocks))
    for g in pres:
        lg.add_bg(Gm.block_foam(g))
    lg.add_bg([c for c in itertools.product(*[range(lo, hi) for lo, hi in R]) if Gm.iscell(c) and c not in used])
    f6, fan = {}, []
    for c in used:
        if roles[c].startswith("clause"):
            continue
        if len(links[c]) == 4:
            m = O.mask(links[c])
            lg.add_cell(c, Gm.locks_of(TF.coupling_with_gs({0, m, 255 ^ m, 255}, "F42")))
        else:
            f6[c] = None
    (i, pol), (a, bb) = sock
    g = (2 * a, 2 * bb, 1)
    f1, f2 = ((2, -1), face) if pol == 1 else (face, (2, -1))
    C = tuple(S * x for x in g)
    cells = [add(C, c) for c in T[f1, f2]]
    tl = template_links(cells)
    if pol == 1:
        bot, tau_b, top, tau_t = cells[0], 1, cells[-1], f2[1]
    else:
        bot, tau_b, top, tau_t = cells[-1], -1, cells[0], -f1[1]
    w = wires[i, pol][-1]
    assert w == end_link(bot, g, tau_b)
    tl[bot].add(DIR_K[sub(w, bot)])
    tl[top].add(DIR_K[sub(end_link(top, g, tau_t), top)])
    for c, ks in tl.items():
        links[c] = ks
        f6[c] = None
    lg.add_bg(Gm.block_interior(g, cells))
    for ij, wl in wires.items():  # every cluster wire end gets its across-face link (absent partner if not chosen)
        o = wl[-1]
        go = tuple((o[j] + 6) // S for j in range(3))
        links[o] = set(links[o]) | {DIR_K[sub(end_link(o, go, ij[1]), o)]}
    for c in f6:
        assert len(links[c]) == 2 and O.mask(links[c]) in O.DOMINOES, (c, links[c])
        f6[c] = O.mask(links[c])
    lg.solve()
    anch = {lg.comp(v) for v in present_robust_refs(pres, offs)}
    unp, _ = propagate(lg, anch, f6, links)
    tset = set(cells)
    return len([c for c in unp if c in tset]), len([c for c in unp if c not in tset])


if __name__ == "__main__":
    margin = int(os.environ.get("MARGIN", "1"))
    _, offs = periodic_giant()
    lk, assign, wires, R, links, roles = cluster_data()
    socks = sorted(assign.items())
    for si in map(int, sys.argv[1:]):
        sock = socks[si]
        res = {}
        t0 = time.time()
        for f in FACES:
            if f == (2, -1):
                continue
            res[f] = w4b(offs, sock, f, margin)
        print(f"socket {sock} (port (literal, polarity) -> socket (a, b)); margin {margin}: "
              f"(unpinned template cells, unpinned cluster cells) per face {res}  [{time.time()-t0:.1f}s RSS {rss():.0f}MB]",
              flush=True)
