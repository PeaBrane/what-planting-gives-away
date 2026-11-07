"""Search for a single-F6-cell clause: the 8 corners split into 4 dominoes (a perfect matching of the cube), three
carrying variables a, b, c and one a private bit d.  For every gauge g, R = {s in {0,1}^4 : g xor sum s_i D_i in
Gamma_F6}; report matchings/gauges where exists_d R is NP-hard-type (non-Schaefer), in particular NAE.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import CUBE_E, coupling_from_class, gs_set  # noqa: E402


def matchings():
    edges = [(a, b) for a, b in CUBE_E]
    out = []
    for combo in itertools.combinations(edges, 4):
        used = set()
        ok = True
        for a, b in combo:
            if a in used or b in used:
                ok = False
                break
            used |= {a, b}
        if ok:
            out.append(combo)
    return out


def closed(R, op, ar):
    for t in itertools.product(R, repeat=ar):
        if op(*t) not in R:
            return False
    return True


def schaefer_type(R):
    R = set(R)
    return dict(
        horn=closed(R, lambda a, b: a & b, 2), dualhorn=closed(R, lambda a, b: a | b, 2),
        bij=closed(R, lambda a, b, c: (a & b) | (b & c) | (a & c), 3), aff=closed(R, lambda a, b, c: a ^ b ^ c, 3))


def main():
    G0, _ = gs_set(coupling_from_class("F6"))
    Ms = matchings()
    print("perfect matchings of Q3:", len(Ms))
    found = {}
    for M in Ms:
        D = [(1 << a) | (1 << b) for a, b in M]
        dirs = [bin(a ^ b).index("1") if False else (a ^ b).bit_length() - 1 for a, b in M]
        for perm in itertools.permutations(range(4)):  # roles a, b, c, d
            Ds = [D[i] for i in perm]
            for g in range(256):
                R4 = set()
                for s in range(16):
                    m = g
                    for i in range(4):
                        if (s >> i) & 1:
                            m ^= Ds[i]
                    if m in G0:
                        R4.add(s)
                P = sorted({s & 7 for s in R4})
                if not P:
                    continue
                st = schaefer_type(P)
                hard = not any(st.values())
                key = (tuple(sorted(dirs)), tuple(P))
                if hard and key not in found:
                    found[key] = (M, perm, g, sorted(R4))
    print("non-Schaefer projected ternary relations found:", len(found))
    for (dirs, P), (M, perm, g, R4) in sorted(found.items())[:40]:
        print(f"  domino axes {dirs}  exists_d R = {[format(p, '03b')[::-1] for p in P]}  M={M} roles={perm} g={g}")


if __name__ == "__main__":
    main()
