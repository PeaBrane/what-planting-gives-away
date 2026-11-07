"""Exact plaquette cover on the L^3 torus: x_e in {0,1}, sum_{e in p} x_e = 1 for every plaquette p.
If feasible, J = -1 on the cover and +1 elsewhere is fully frustrated with frustration exactly 1/4,
and is a frustrated-plaquette-loop planted instance (H = (1/4) sum_p H_p)."""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def run(L):
    idx = lambda x, y, z: (x % L) * L * L + (y % L) * L + (z % L)
    eid = {}
    for x, y, z in itertools.product(range(L), repeat=3):
        for ax in range(3):
            d = [0, 0, 0]; d[ax] = 1
            eid[(idx(x, y, z), ax)] = len(eid)
    rows = []
    for x, y, z in itertools.product(range(L), repeat=3):
        for a, b in itertools.combinations(range(3), 2):
            p = [x, y, z]; pa = list(p); pa[a] += 1; pb = list(p); pb[b] += 1
            row = np.zeros(len(eid))
            for k in (eid[(idx(*p), a)], eid[(idx(*p), b)], eid[(idx(*pa), b)], eid[(idx(*pb), a)]):
                row[k] += 1
            rows.append(row)
    A = np.array(rows)
    res = milp(np.zeros(len(eid)), constraints=LinearConstraint(A, 1, 1), integrality=np.ones(len(eid)), bounds=Bounds(0, 1))
    if res.status == 0:
        x = np.round(res.x)
        print(f"L={L}: exact plaquette cover exists, |cover| = {int(x.sum())} of {len(eid)} edges (= {x.sum()/len(eid):.4f})")
    else:
        print(f"L={L}: no exact plaquette cover (status {res.status}: {res.message})")

for L in (2, 4, 6):
    run(L)
