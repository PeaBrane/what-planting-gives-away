"""Lemma 8.1 hypotheses on the fresh instances, re-derived here from netc/roles of the construction builder
and from the raw couplings in the instance files.
  - every coupling type is F42 or F6 (own frustrated-face count); background/fan-out types contain 0 in Gamma;
  - links symmetric with equal labels; wire cells: 2 link corners (a domino), F6, 0 and 1_D in Gamma;
    fan-outs: 4 link corners (a face), F42 with Gamma = {0, 1_F, 1_F^c, 1}; clause: 8 corners, the four dominoes
    {01, 26, 37, 45} carry four distinct labels;
  - every network connected through wire and fan-out cells only (clause cells excluded).
Run: TILE_S=12 python -I -B own_hyp_check.py NAME ..."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_cube as O  # noqa: E402
from own_geom import DIRS  # noqa: E402
from own_xcheck import build  # noqa: E402

DOMS = [(0, 1), (2, 6), (3, 7), (4, 5)]


def main(names):
    for name in names:
        n, clauses, origins, edges, paths, ksh, L, netc, role = build(name)
        z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"inst/{name}.npz")) if name != "a5_single" else \
            np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst/a5_single_frozen.npz"))
        assert int(z["L"]) == L
        types = [tuple(int(x) for x in w) for w in z["types"]]
        tcls = {i: O.cls(w) for i, w in enumerate(types)}
        Wc = {tuple(int(x) for x in c): tuple(int(x) for x in w) for c, w in zip(z["wc_cells"], z["wc_W"])}
        bad = dict(class_not_F6_F42=sum(c not in ("F6", "F42") for c in tcls.values()), bg_0_missing=0,
                   asym_link=0, wire=0, fanout=0, clause=0, net_disconnected=0)
        for i in range(3):
            if 0 not in O.gamma(types[i]):
                bad["bg_0_missing"] += 1
        for c, lab in netc.items():
            for k, nt in lab.items():
                c2 = tuple((c[j] + DIRS[k][j]) % L for j in range(3))
                if netc.get(c2, {}).get(7 - k) != nt:
                    bad["asym_link"] += 1
            W = Wc[c]
            G = O.gamma(W)
            m = O.mask(lab.keys())
            if role[c].startswith("clause"):
                labs = [lab.get(a) for a, _ in DOMS]
                if not (len(lab) == 8 and O.cls(W) == "F6" and all(lab[a] == lab[b] for a, b in DOMS)
                        and len(set(labs)) == 4):
                    bad["clause"] += 1
            elif len(lab) == 4:
                if not (O.cls(W) == "F42" and frozenset(lab) in O.FACE_SETS and G == frozenset({0, m, 255 ^ m, 255})
                        and len(set(lab.values())) == 1):
                    bad["fanout"] += 1
            else:
                if not (len(lab) == 2 and m in O.DOMINOES and O.cls(W) == "F6" and 0 in G and m in G
                        and len(set(lab.values())) == 1):
                    bad["wire"] += 1
        # network connectivity through wire and fan-out cells only
        verts, adj = {}, {}
        for c, lab in netc.items():
            vs = [(tuple((c[j] + O.CORN[k][j]) % L for j in range(3)), nt) for k, nt in lab.items()]
            for v, nt in vs:
                verts.setdefault(nt, set()).add(v)
            if role[c].startswith("clause"):
                continue
            for v, _ in vs:
                for w, _ in vs:
                    if v != w:
                        adj.setdefault(v, set()).add(w)
        for nt, vs in verts.items():
            st = next(iter(vs))
            seen, stack = {st}, [st]
            while stack:
                v = stack.pop()
                for w in adj.get(v, ()):
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            if seen != vs:
                bad["net_disconnected"] += 1
        print(f"[{name}] L={L} types {sorted(set(tcls.values()))}; networks {len(verts)}; cells {len(netc)};"
              f" violations {bad}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TILE_S", "12")
    main(sys.argv[1:])
