"""Catalogue of all 2^12 +-1 cube couplings: ground-state sets, flip structure, classes, relation properties."""
import itertools
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict

from cube import (CORNERS, CUBE_E, FACES, all_couplings, class_name, coupling_from_class, gs_set, oh_perms,
                  permute_W, gauge_W, spin)

ADJ = {(a, b) for a, b in CUBE_E} | {(b, a) for a, b in CUBE_E}


def popcount(x):
    return bin(x).count("1")


def is_delta_matroid(R, n, even=False):
    R = set(R)
    for x in R:
        for y in R:
            d = x ^ y
            for i in range(n):
                if not (d >> i) & 1:
                    continue
                ok = False
                for j in range(n):
                    if not (d >> j) & 1:
                        continue
                    if even and j == i:
                        continue
                    z = x ^ (1 << i) ^ ((1 << j) if j != i else 0)
                    if z in R:
                        ok = True
                        break
                if not ok:
                    return False
    return True


def closed(R, op):
    R = list(R)
    S = set(R)
    for a in R:
        for b in R:
            if op.__code__.co_argcount == 2:
                if op(a, b) not in S:
                    return False
            else:
                for c in R:
                    if op(a, b, c) not in S:
                        return False
    return True


def schaefer(R, n):
    full = (1 << n) - 1
    return dict(
        horn=closed(R, lambda a, b: a & b),
        dualhorn=closed(R, lambda a, b: a | b),
        bijunctive=closed(R, lambda a, b, c: (a & b) | (b & c) | (a & c)),
        affine=closed(R, lambda a, b, c: a ^ b ^ c),
    )


def main():
    Ws = all_couplings()
    cls_count = Counter()
    gs_count = defaultdict(set)
    odd_flip = 0
    nondomino_2flip = 0
    for W in Ws:
        G, e0 = gs_set(W)
        c = class_name(W)
        cls_count[c] += 1
        gs_count[c].add(len(G) // 2)
        for x in G:
            for y in G:
                d = x ^ y
                if popcount(d) % 2:
                    odd_flip += 1
                if popcount(d) == 2:
                    a, b = [k for k in range(8) if (d >> k) & 1]
                    if (a, b) not in ADJ:
                        nondomino_2flip += 1
    print("class sizes over all 4096 couplings:", dict(cls_count))
    print("ground-state pairs |Gamma|/2 per class:", {k: sorted(v) for k, v in gs_count.items()})
    print("odd flips between ground states:", odd_flip)
    print("2-flips between ground states that are not dominoes:", nondomino_2flip)

    # flip sets (modulo complement) per class, canonical representative in ferromagnetic gauge
    for name in ["F0", "F21", "F22", "F41", "F42", "F6"]:
        W = coupling_from_class(name)
        G, e0 = gs_set(W)
        assert 0 in G and class_name(W) == name, name
        flips = sorted({min(x ^ y, 0xFF ^ x ^ y) for x in G for y in G if x != y and x != 0xFF ^ y})
        desc = []
        for f in flips:
            ks = [k for k in range(8) if (f >> k) & 1]
            desc.append("{" + ",".join("".join(map(str, CORNERS[k])) for k in ks) + "}")
        print(f"{name}: W={W} |Gamma|={len(G)} flip sets (mod complement): {desc}")
        dm = is_delta_matroid(G, 8)
        edm = is_delta_matroid(G, 8, even=True)
        print(f"   delta-matroid: {dm}, even delta-matroid: {edm}, Schaefer: {schaefer(G, 8)}")

    # F22 column structure: frustrated faces z0,z1 -> vertical columns (x,y)
    W = coupling_from_class("F22")
    G, _ = gs_set(W)
    cols = [(0, 4), (1, 5), (3, 7), (2, 6)]  # c00, c10, c11, c01 (corner index = dx + 2dy + 4dz)
    colrel = set()
    for m in G:
        bits = []
        for a, b in cols:
            assert ((m >> a) & 1) == ((m >> b) & 1), "column not locked"
            bits.append((m >> a) & 1)
        colrel.add("".join(map(str, bits)))
    print("F22 column relation (c00,c10,c11,c01):", sorted(colrel))
    proj = sorted({s[:3] for s in colrel})
    print("  exists c01: (c00,c10,c11) in", proj, " missing:", sorted(set("".join(p) for p in itertools.product("01", repeat=3)) - set(proj)))
    R4 = {int(s[::-1], 2) for s in colrel}
    print("  column relation delta-matroid:", is_delta_matroid(R4, 4), " Schaefer:", schaefer(R4, 4))

    # orbit sizes under Oh x gauge
    perms = oh_perms()
    seen = {}
    for name in ["F0", "F21", "F22", "F41", "F42", "F6"]:
        W0 = coupling_from_class(name)
        orb = set()
        for p in perms:
            Wp = permute_W(W0, p)
            for t in range(256):
                orb.add(gauge_W(Wp, t))
        seen[name] = len(orb)
    print("orbit sizes under Oh x gauge:", seen, "total", sum(seen.values()))


if __name__ == "__main__":
    main()
