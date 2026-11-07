"""Cycle LP on Hamze et al. 3D tile-planted instances (arXiv:1711.04083, Sec. III).

Cells: unit cubes whose lower corner has all-equal coordinate parities (edge-disjoint, every spin in two
cells). Each cell gets a random member of class F6 (prob p6) or F22, from enumerating all +-1 cube
couplings with ferromagnetic ground state, classified by frustrated facets and ground-state count.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cycle_lp import cycle_lp  # noqa: E402

CORNERS = list(itertools.product([0, 1], repeat=3))
CUBE_E = [(a, b) for a, b in itertools.combinations(range(8), 2)
          if sum(abs(p - q) for p, q in zip(CORNERS[a], CORNERS[b])) == 1]
FACES = []
for ax in range(3):
    for val in (0, 1):
        vs = {i for i, c in enumerate(CORNERS) if c[ax] == val}
        FACES.append([k for k, (a, b) in enumerate(CUBE_E) if a in vs and b in vs])


def classify():
    S = np.array(list(itertools.product([1, -1], repeat=8)))
    prod = np.stack([S[:, a] * S[:, b] for a, b in CUBE_E], axis=1)  # (256, 12)
    classes = {}
    for signs in itertools.product([1, -1], repeat=12):
        J = np.array(signs, float)
        if all(J[f].prod() > 0 for f in FACES) and J.min() > 0:
            continue  # pure ferromagnet
        En = -(prod @ J)
        if En[0] > En.min():
            continue  # +1 must be a ground state
        nfr = sum(J[f].prod() < 0 for f in FACES)
        gs = int((En == En.min()).sum()) // 2
        classes.setdefault((nfr, gs), []).append(J)
    return classes


def instance(rng, L, members_by_class, probs):
    idx = lambda x, y, z: (x % L) * L * L + (y % L) * L + (z % L)
    E, J = [], []
    keys = list(members_by_class)
    for x, y, z in itertools.product(range(L), repeat=3):
        if not (x % 2 == y % 2 == z % 2):
            continue
        cls = keys[rng.choice(len(keys), p=probs)]
        Jc = members_by_class[cls][rng.integers(len(members_by_class[cls]))]
        verts = [idx(x + c[0], y + c[1], z + c[2]) for c in CORNERS]
        for k, (a, b) in enumerate(CUBE_E):
            E.append((verts[a], verts[b]))
            J.append(Jc[k])
    J = np.array(J)
    return E, J, -J.sum()  # E0 = H(+1)


def main():
    classes = classify()
    print("ferro-GS +-1 cube classes (frustrated facets, GS count up to Z2): members")
    for k in sorted(classes):
        print(f"  {k}: {len(classes[k])}")
    F6, F22 = classes[(6, 8)], classes[(2, 4)]
    rng = np.random.default_rng(2)
    for L in (6,):
        N = L ** 3
        for p6 in (1.0,):
            tight = integral = 0
            trials = 1
            for _ in range(trials):
                E, J, E0 = instance(rng, L, {"F6": F6, "F22": F22}, [p6, 1 - p6])
                eu = np.array([e[0] for e in E]); ev = np.array([e[1] for e in E])
                t = rng.choice([-1, 1], N)
                lp, x, ncuts = cycle_lp(N, E, J * t[eu] * t[ev], max_rounds=60, max_new=600, partial=True)
                print(f'  LP bound {lp:.4f} vs E0 {E0}, cuts {ncuts}, frac entries {(np.minimum(x,1-x)>1e-6).sum()}', flush=True)
                tight += lp is not None and abs(lp - E0) < 1e-5
                integral += bool(np.all(np.minimum(x, 1 - x) < 1e-6))
            print(f"L={L} p6={p6}: LP == E0 {tight}/{trials}, integral LP vertex {integral}/{trials}", flush=True)


if __name__ == "__main__":
    main()
