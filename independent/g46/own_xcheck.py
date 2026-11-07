"""model-instance consistency on fresh formulas, including fan-outs (skipped by W3's xcheck46).
Instances come from the construction builder (layout_A46 -> expand46 -> to_torus46); the models are rebuilt here.
Run: TILE_S=12 python -I own_xcheck.py NAME [...]"""
import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_cube as O  # noqa: E402
import own_geom as Gm  # noqa: E402
from own_geom import DIR_K, H, S, SLAB, add, sub  # noqa: E402
from own_windows import cluster_data, end_link, rss, template_links  # noqa: E402
from a5_formulas import FORMULAS  # noqa: E402

os.environ.setdefault("TILE_S", "12")
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "construction"))
import layout46 as G  # noqa: E402
import templates46 as T6  # noqa: E402
from layout import face_of  # noqa: E402


def w4_model(margin=2):
    lk, assign, wires, R, links, roles = cluster_data()
    T = T6.templates46()
    used = set(lk.used)
    blocks = list(itertools.product(range(-margin, 6 + margin), range(-margin, 4 + margin), range(-1 - margin, 4)))
    pres = [g for g in blocks if g[2] not in SLAB]
    bg = set()
    for g in pres:
        bg.update(Gm.block_foam(g))
    bg.update(c for c in itertools.product(*[range(lo, hi) for lo, hi in R]) if Gm.iscell(c) and c not in used)
    f6, fan, clause = {}, {}, None
    for c in used:
        if roles[c].startswith("clause"):
            clause = c
        elif len(links[c]) == 4:
            fan[c] = frozenset(links[c])
        else:
            f6[c] = set(links[c])
    for (i, pol), (a, bb) in assign.items():
        g = (2 * a, 2 * bb, 1)
        f1, f2 = ((2, -1), (2, 1)) if pol == 1 else ((2, 1), (2, -1))
        C = tuple(S * x for x in g)
        cells = [add(C, c) for c in T[f1, f2]]
        tl = template_links(cells)
        bot, top = (cells[0], cells[-1]) if pol == 1 else (cells[-1], cells[0])
        w = wires[i, pol][-1]
        assert w == end_link(bot, g, pol)
        tl[bot].add(DIR_K[sub(w, bot)])
        f6[w].add(DIR_K[sub(bot, w)])
        tl[top].add(DIR_K[sub(end_link(top, g, pol), top)])
        f6.update(tl)
        bg.update(Gm.block_interior(g, cells))
    assert not (bg & set(f6)) and not (bg & set(fan))
    return bg, {c: frozenset(v) for c, v in f6.items()}, fan, clause, blocks


def build(name, margin=1):
    n, clauses = FORMULAS[name]
    origins, edges, paths = G.layout_A46(clauses)
    lk, negs = G.expand46(clauses, origins, edges, paths)
    lo = [min(c[j] for c in lk.used) for j in range(3)]
    kshift = tuple(margin - math.floor(l / S) for l in lo)
    L, netc, role = G.to_torus46(lk, negs, margin)
    return n, clauses, origins, edges, paths, kshift, L, netc, role


def main(names):
    bg, f6, fan, clause, wblocks = w4_model()
    T = T6.templates46()
    print(f"W4 model: {len(bg)} background, {len(f6)} F6, {len(fan)} fan-outs  [RSS {rss():.0f}MB]", flush=True)
    for name in names:
        t0 = time.time()
        n, clauses, origins, edges, paths, ksh, L, netc, role = build(name)
        nb = L // S
        P = lambda c, sh: tuple((c[j] + sh[j]) % L for j in range(3))  # noqa: E731
        bad = dict(bg_not_bg=0, bg_mod3=0, f6_missing=0, f6_links=0, f6_role=0, fan_missing=0, fan_links=0,
                   fan_role=0, clause=0)
        for j, Oj in enumerate(origins):
            sh = tuple(S * (Oj[i] + ksh[i]) for i in range(3))
            assert sum(sh) % 3 == 0 and L % 3 == 0
            for c in bg:
                if P(c, sh) in netc:
                    bad["bg_not_bg"] += 1
            for c, ks in f6.items():
                p = P(c, sh)
                if p not in netc:
                    bad["f6_missing"] += 1
                    continue
                if frozenset(netc[p]) != ks:
                    bad["f6_links"] += 1
                if role[p].startswith("side") or role[p].startswith("clause"):
                    bad["f6_role"] += 1
            for c, ks in fan.items():
                p = P(c, sh)
                if p not in netc:
                    bad["fan_missing"] += 1
                    continue
                if frozenset(netc[p]) != ks:
                    bad["fan_links"] += 1
                if not role[p].startswith("side"):
                    bad["fan_role"] += 1
            if role.get(P(clause, sh)) != f"clause{j}":
                bad["clause"] += 1
        # model z-range of the W4 window and of W3 windows must not wrap onto the slab layers
        hmax = max(g[2] for p in paths for g in p)
        slab_t = {(z + ksh[2]) % nb for z in SLAB}
        w4_wrap = sum(1 for z in range(-3, 4) if z not in SLAB and (z + ksh[2]) % nb in slab_t)
        w3_wrap = sum(1 for z in range(1, hmax + 3) if (z + ksh[2]) % nb in slab_t)
        # path blocks: templates, links and interiors (the W3 model), and z-structure
        regions = {((Oj[0] + a + ksh[0]) % nb, (Oj[1] + b + ksh[1]) % nb, (z + ksh[2]) % nb)
                   for Oj in origins for a in range(6) for b in range(4) for z in SLAB}
        tb = dict(link_mismatch=0, interior_not_bg=0, z_lt_1=0, z1_nonsocket=0, extra_cells=0)
        accounted = set()
        sh0 = tuple(S * ksh[i] for i in range(3))
        pblocks = set()
        for path in paths:
            for t, g in enumerate(path):
                f1 = (2, -1) if t == 0 else face_of(sub(path[t - 1], g))
                f2 = (2, -1) if t == len(path) - 1 else face_of(sub(path[t + 1], g))
                if g[2] < 1:
                    tb["z_lt_1"] += 1
                if g[2] == 1 and (f1, f2) not in (((2, -1), (2, 1)), ((2, 1), (2, -1))):
                    tb["z1_nonsocket"] += 1
                pblocks.add(tuple((g[i] + ksh[i]) % nb for i in range(3)))
                C = tuple(S * x for x in g)
                cells = [add(C, c) for c in T[f1, f2]]
                tl = template_links(cells)
                for c, tau in ((cells[0], -f1[1]), (cells[-1], f2[1])):
                    tl[c].add(DIR_K[sub(end_link(c, g, tau), c)])
                for c in cells:
                    p = P(c, sh0)
                    accounted.add(p)
                    if frozenset(netc.get(p, {})) != frozenset(tl[c]):
                        tb["link_mismatch"] += 1
                for c in Gm.block_interior(g, cells):
                    if P(c, sh0) in netc:
                        tb["interior_not_bg"] += 1
                # foam of the path block itself: crossing cells other than the template's own are background
                for c in Gm.block_foam(g):
                    if P(c, sh0) in netc:
                        tb["interior_not_bg"] += 1
        # every construction cell is in a region block or is a template cell of a path block
        for p in netc:
            blk = tuple(((p[i] + H) // S) % nb for i in range(3))
            if p not in accounted and blk not in regions:
                tb["extra_cells"] += 1
        # W1 model: the foam of every non-slab block of the torus is background
        foam_bad, nonslab = 0, 0
        for g in itertools.product(range(nb), repeat=3):
            if g[2] in slab_t:
                continue
            nonslab += 1
            for c in Gm.block_foam(g):
                if tuple(x % L for x in c) in netc:
                    foam_bad += 1
        print(f"[{name}] m={len(clauses)} origins={origins} L={L} nb={nb} hmax={hmax} (nb >= hmax+4: "
              f"{nb >= hmax + 4}); W4 per-patch mismatches {bad}; window z-wrap onto slab: W4 {w4_wrap}, W3 {w3_wrap};"
              f" path blocks {len(pblocks)} {tb}; non-slab blocks {nonslab}, foam cells in netc {foam_bad}"
              f"  [{time.time()-t0:.1f}s RSS {rss():.0f}MB]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
