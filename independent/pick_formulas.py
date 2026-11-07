"""Pick fresh random NAE-3SAT formulas (own RNG) and write formulas.json."""
import os
import itertools
import json
import numpy as np


def nae_count(cl, n):
    return sum(all(len({a[v] ^ g for v, g in c}) > 1 for c in cl) for a in itertools.product((0, 1), repeat=n))


def rand_formula(rng, n, m):
    out = []
    for _ in range(m):
        vs = rng.choice(n, size=3, replace=False)
        out.append([[int(v), int(rng.integers(2))] for v in vs])
    return out


F = {}
F["single_mixed"] = [[[2, 1], [0, 0], [1, 1]]]
F["unsat_perm"] = [[[0, 0], [1, 0], [2, 0]], [[1, 0], [2, 1], [0, 0]], [[2, 0], [0, 1], [1, 0]], [[1, 1], [0, 0], [2, 0]]]
F["pair_shift"] = [[[0, 0], [1, 1], [2, 0]], [[1, 0], [2, 1], [3, 0]]]
F["var4x"] = [[[0, 0], [1, 0], [2, 0]], [[1, 0], [0, 1], [3, 0]], [[2, 1], [3, 1], [0, 0]], [[0, 1], [1, 1], [3, 0]]]
picked = {}
for n in (4, 5, 6):
    for seed in range(1000, 1400):
        rng = np.random.default_rng(seed)
        cl = rand_formula(rng, n, 4)
        c = nae_count(cl, n)
        key = (n, c > 0)
        if key not in picked:
            picked[key] = (seed, cl, c)
for (n, sat), (seed, cl, c) in sorted(picked.items()):
    F[f"rand_n{n}_m4_s{seed}_{'sat' if sat else 'unsat'}"] = cl
    print(n, sat, seed, c, cl)
for k, cl in F.items():
    n = 1 + max(v for c in cl for v, _ in c)
    print(k, "n", n, "m", len(cl), "#NAE solutions", nae_count(cl, n))
json.dump(F, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "formulas.json"), "w"), indent=1)
