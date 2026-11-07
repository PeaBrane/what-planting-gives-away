"""Fresh NAE-3SAT formulas (distinct from the construction's own test formulas).  Literal (v, neg)."""
import itertools
import random


def rand_formula(n, m, seed):
    rng = random.Random(seed)
    return n, [[(v, rng.randint(0, 1)) for v in rng.sample(range(n), 3)] for _ in range(m)]


FORMULAS = {
    # m = 1, L = 96: fresh sign pattern
    "a5_single": (3, [[(0, 1), (1, 1), (2, 0)]]),
    # m = 4, 2x2 patch array, unsatisfiable: with x0 fixed, the four clauses forbid all four (x1, x2) patterns;
    # x0 sits at a different literal position in each clause
    "a5_unsat4": (3, [[(2, 0), (0, 0), (1, 0)], [(1, 1), (2, 0), (0, 0)], [(0, 0), (2, 1), (1, 0)],
                      [(1, 1), (0, 0), (2, 1)]]),
    # m = 3: partial last patch row (q = 2: row 0 holds 2 patches, row 1 holds 1)
    "a5_sat3p": (4, [[(0, 1), (1, 0), (2, 1)], [(1, 0), (3, 1), (0, 0)], [(2, 0), (3, 0), (1, 1)]]),
    # random, m = 4
    "a5_rand4": rand_formula(5, 4, 2026),
    # random, m = 3 (partial last row)
    "a5_rand3p": rand_formula(4, 3, 7),
}


def nae(clauses, a):
    return all(len({a[v] ^ ng for v, ng in cl}) > 1 for cl in clauses)


def satisfiable(n, clauses):
    return any(nae(clauses, a) for a in itertools.product((0, 1), repeat=n))


if __name__ == "__main__":
    for k, (n, cl) in FORMULAS.items():
        print(k, n, cl, "NAE-sat:", satisfiable(n, cl),
              "#sat:", sum(nae(cl, a) for a in itertools.product((0, 1), repeat=n)))
