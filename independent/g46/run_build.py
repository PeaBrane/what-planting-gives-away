"""Builder: run the construction reduction layout_A46 -> expand46 -> to_torus46 -> assign_couplings46 and write
the instance (raw couplings of every cell) for the independent decider.
MODE=construction: the construction assign_couplings46 itself (fits locally only at L = 96).
MODE=fast:   the same algorithm (A0 = most-B-incidence lock component, greedy propagated anchoring in netc order,
             gauge = construction self_pinnable), with the construction Python union-find replaced by vectorised components.
             Validated against MODE=construction at L = 96 (identical couplings).
Run: TILE_S=12 MODE=fast python -I run_build.py NAME OUT.npz"""
import json
import os
import resource
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a5_formulas import FORMULAS  # noqa: E402

os.environ.setdefault("TILE_S", "12")
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "construction"))
import layout46 as G  # noqa: E402
import lean_tile as LT  # noqa: E402  (only gs_info, as the construction assign_couplings46 uses it)
from tiles import CORNERS, DIRS, FULL, coupling_with_gs, mask  # noqa: E402


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def cell_index(L, x, y, z):
    """The independent decider's canonical cell order: parity block, then (x//2, y//2, z//2) row-major."""
    h = L // 2
    return (x % 2) * h ** 3 + ((x // 2) * h + (y // 2)) * h + (z // 2)


def all_cell_coords(L):
    h = L // 2
    i, j, k = np.indices((h, h, h), dtype=np.int32).reshape(3, -1)
    return (np.concatenate([2 * i, 2 * i + 1]), np.concatenate([2 * j, 2 * j + 1]),
            np.concatenate([2 * k, 2 * k + 1]))


def fast_assign(L, netc, role, negs):
    """Replica of layout46.assign_couplings46 (same decisions), vectorised lock components."""
    Wc = {}
    for c, lab in netc.items():
        r = role[c]
        if r.startswith("clause"):
            Wc[c] = G.clause_coupling46(negs[int(r[len("clause"):])])
        elif len(lab) == 4:
            m = mask(lab.keys())
            Wc[c] = coupling_with_gs({0, m, FULL ^ m, FULL}, "F42")
    ux, uy, uz = all_cell_coords(L)
    n_cells = ux.size
    # per-cell lock pattern id: 0..2 background axis, 3.. fan-out patterns; -1 F6 (wire or clause)
    pat = ((ux.astype(np.int64) + uy + uz) % 3).astype(np.int8)
    pats = [LT.gs_info(G.bg_coupling(u))[2] for u in ((0, 0, 0), (0, 0, 4), (2, 0, 0))]
    for ax, lks in enumerate(pats):
        assert all(x == 0 for _, _, x in lks)
    for c, lab in netc.items():
        ci = cell_index(L, *c)
        if c in Wc and len(lab) == 4:
            lks = LT.gs_info(Wc[c])[2]
            assert LT.gs_info(Wc[c])[1] and all(x == 0 for _, _, x in lks)
            if lks not in pats:
                pats.append(lks)
            pat[ci] = pats.index(lks)
        else:
            pat[ci] = -1
    # vertex ids: construction corner_index convention
    src, dst = [], []
    for p, lks in enumerate(pats):
        idx = np.nonzero(pat == p)[0]
        x, y, z = ux[idx].astype(np.int64), uy[idx].astype(np.int64), uz[idx].astype(np.int64)
        for k0, k, _ in lks:
            a, b = CORNERS[k0], CORNERS[k]
            src.append(((x + a[0]) % L + L * (((y + a[1]) % L) + L * ((z + a[2]) % L))).astype(np.int32))
            dst.append(((x + b[0]) % L + L * (((y + b[1]) % L) + L * ((z + b[2]) % L))).astype(np.int32))
        del idx, x, y, z
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    n = L ** 3
    A = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()
    del src, dst
    _, lab = connected_components(A, directed=False)
    del A
    lab = lab.astype(np.int32)
    # B incidences per component: every corner k of every cell with k not a link corner
    cnt = np.zeros(int(lab.max()) + 1, dtype=np.int64)
    for k in range(8):
        d = CORNERS[k]
        v = ((ux.astype(np.int64) + d[0]) % L + L * (((uy.astype(np.int64) + d[1]) % L)
                                                   + L * ((uz.astype(np.int64) + d[2]) % L)))
        np.add.at(cnt, lab[v], 1)
        del v
    for c, labc in netc.items():  # remove link-corner incidences
        for k in labc:
            cnt[lab[LT.corner_index(L, c, k)]] -= 1
    root = int(np.argmax(cnt))
    n_comp = int((cnt > 0).sum())
    anchored = {root}
    pending = [c for c in netc if c not in Wc]
    order = []
    while pending:
        progress = []
        for c in pending:
            labc = netc[c]
            assert len(labc) == 2
            D = mask(labc.keys())
            Pm = 0
            for k in range(8):
                if k not in labc and int(lab[LT.corner_index(L, c, k)]) in anchored:
                    Pm |= 1 << k
            W = G.self_pinnable(D, Pm)
            if W is not None:
                Wc[c] = W
                progress.append(c)
                order.append((c, Pm))
                for k in range(8):
                    if k not in labc:
                        anchored.add(int(lab[LT.corner_index(L, c, k)]))
        if not progress:
            raise RuntimeError(f"{len(pending)} F6 wire cells not pinnable, e.g. {pending[0]} ({role[pending[0]]})")
        done = set(progress)
        pending = [c for c in pending if c not in done]
    info = dict(lock_components_with_B=n_comp, main_component_B_incidences=int(cnt[root]),
                anchored_components=len(anchored))
    return Wc, info


def main(name, out):
    t0 = time.time()
    n_vars, clauses = FORMULAS[name]
    origins, edges, paths = G.layout_A46(clauses)
    lk, negs = G.expand46(clauses, origins, edges, paths)
    L, netc, role = G.to_torus46(lk, negs, int(os.environ.get("MARGIN", "1")))
    print(f"[{name}] L={L} construction cells {len(netc)}  [{time.time()-t0:.1f}s RSS {rss():.0f}MB]", flush=True)
    mode = os.environ.get("MODE", "fast")
    t1 = time.time()
    if mode == "construction":
        Wc, info, uf, _ = G.assign_couplings46(L, netc, role, negs)
        del uf
    else:
        Wc, info = fast_assign(L, netc, role, negs)
    print(f"  assign ({mode}): {info}  [{time.time()-t1:.1f}s RSS {rss():.0f}MB]", flush=True)
    # instance file: raw couplings of every cell
    types = [G.bg_coupling(u) for u in ((0, 0, 0), (0, 0, 4), (2, 0, 0))]  # sum mod 3 = 0, 1, 2
    tid = {w: i for i, w in enumerate(types)}
    ux, uy, uz = all_cell_coords(L)
    tcell = ((ux.astype(np.int64) + uy + uz) % 3).astype(np.int16)
    for c, W in Wc.items():
        if W not in tid:
            tid[W] = len(types)
            types.append(W)
        tcell[cell_index(L, *c)] = tid[W]
    for c in netc:
        assert c in Wc
    # read-off representatives: one link vertex per variable network
    rep = -np.ones((n_vars, 3), dtype=np.int64)
    for c, labc in netc.items():
        for k, nt in labc.items():
            if nt.startswith("x"):
                v = int(nt[1:])
                if rep[v, 0] < 0:
                    rep[v] = [(c[j] + CORNERS[k][j]) % L for j in range(3)]
    labels = sorted({nt for labc in netc.values() for nt in labc.values()})
    lid = {nt: i for i, nt in enumerate(labels)}
    nvert = {}
    for c, labc in netc.items():
        for k, nt in labc.items():
            v = tuple((c[j] + CORNERS[k][j]) % L for j in range(3))
            nvert[(v[0] * L + v[1]) * L + v[2]] = lid[nt]  # independent decider vertex id convention
    wc_list = sorted((c, W) for c, W in Wc.items())
    np.savez_compressed(out, L=L, types=np.array(types, dtype=np.int8), tcell=tcell, rep=rep,
                        meta=json.dumps(dict(name=name, n_vars=n_vars, clauses=clauses, mode=mode, info=info,
                                             origins=origins)),
                        wc_cells=np.array([c for c, _ in wc_list], dtype=np.int32),
                        wc_W=np.array([W for _, W in wc_list], dtype=np.int8),
                        net_vid=np.array(list(nvert), dtype=np.int64), net_lab=np.array(list(nvert.values())),
                        net_names=np.array(labels))
    print(f"  wrote {out}: {len(types)} coupling types, {len(Wc)} non-background cells  "
          f"[total {time.time()-t0:.1f}s RSS {rss():.0f}MB]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
