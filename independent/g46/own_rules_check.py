"""recompute the F6/F42 lock and pinning rules from raw couplings and compare with the construction code.
Run: TILE_S=12 python -I own_rules_check.py"""
import itertools
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_cube as O  # noqa: E402

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "construction"))
os.environ.setdefault("TILE_S", "12")
import layout46 as G  # noqa: E402  (construction; only for comparison)
import tiles as TF  # noqa: E402

out = {}
# 1. catalogue
G_all = O.all_gammas()
cnt, gsz = Counter(), {}
for W, Gm in zip(O.ALL_W, G_all):
    c = O.cls(W)
    cnt[c] += 1
    gsz.setdefault(c, set()).add(len(Gm))
out["class_sizes"] = dict(cnt)
out["gamma_sizes"] = {k: sorted(v) for k, v in gsz.items()}
# F42 ground-state sets with 0: always {0, 1_F, 1_F^c, 1} for the face pair of the unfrustrated axis
bad42 = 0
for W, Gm in zip(O.ALL_W, G_all):
    if O.cls(W) != "F42" or 0 not in Gm:
        continue
    cl, par, aff = O.lock_pairs(Gm)
    if not aff or len(set(cl)) != 2 or any(par):
        bad42 += 1
        continue
    plate = frozenset(k for k in range(8) if cl[k] == cl[0])
    if plate not in O.FACE_SETS:
        bad42 += 1
out["F42_with_0_not_two_face_plates"] = bad42

# 2. background couplings used by the instance (construction bg_coupling), re-derived here
bg = {}
for u in [(0, 0, 0), (1, 1, 1), (2, 2, 2), (2, 0, 0), (1, 1, 3), (0, 0, 4)]:
    ax = sum(u) % 3
    W = G.bg_coupling(u)
    Gm = O.gamma(W)
    cl, par, aff = O.lock_pairs(Gm)
    plates = sorted({frozenset(k for k in range(8) if cl[k] == c) for c in set(cl)}, key=min)
    want = sorted([O.FACE_SETS[2 * ax], O.FACE_SETS[2 * ax + 1]], key=min)
    bg[u] = dict(ax=ax, cls=O.cls(W), affine=aff, parities0=not any(par), plates_perp_ax=plates == want)
out["background"] = bg
assert all(v["cls"] == "F42" and v["affine"] and v["parities0"] and v["plates_perp_ax"] for v in bg.values())

# 3. fan-out couplings
fan = {}
for f in O.FACE_SETS:
    m = O.mask(f)
    W = TF.coupling_with_gs({0, m, 255 ^ m, 255}, "F42")
    Gm = O.gamma(W)
    fan[m] = (O.cls(W), Gm == frozenset({0, m, 255 ^ m, 255}))
out["fanouts_ok"] = all(c == "F42" and ok for c, ok in fan.values())

# 4. self_pinnable equivalence, exhaustive over dominoes D and anchored sets P within the complement of D
mism, n, npin, wbad = 0, 0, 0, 0
for D in O.DOMINOES:
    comp = [k for k in range(8) if not (D >> k) & 1]
    for r in range(7):
        for sub in itertools.combinations(comp, r):
            P = O.mask(sub)
            n += 1
            mine = O.pinnable(D, P)
            Wf = G.self_pinnable(D, P)
            npin += mine
            if mine != (Wf is not None):
                mism += 1
            if Wf is not None:
                Gm = O.gamma(Wf)
                if not (O.cls(Wf) == "F6" and 0 in Gm and D in Gm and {s for s in Gm if s & P == 0} == {0, D}):
                    wbad += 1
out["self_pinnable"] = dict(pairs=n, pinnable=npin, mismatches_vs_frozen=mism, construction_W_invalid=wbad)
# monotonicity in P and Fact 8.2 (<= 1 floating B corner suffices)
mono_bad, fact82_bad = 0, 0
for D in O.DOMINOES:
    comp = [k for k in range(8) if not (D >> k) & 1]
    for r in range(7):
        for sub in itertools.combinations(comp, r):
            P = O.mask(sub)
            if O.pinnable(D, P):
                for k in comp:
                    if not O.pinnable(D, P | (1 << k)):
                        mono_bad += 1
            if r >= 5 and not O.pinnable(D, P):
                fact82_bad += 1
out["pinnable_monotone_violations"] = mono_bad
out["fact82_violations"] = fact82_bad
# minimal pinning sets with 2 floating corners: which floating pairs are tolerated
two_float = Counter()
for D in O.DOMINOES:
    comp = [k for k in range(8) if not (D >> k) & 1]
    for fl in itertools.combinations(comp, 2):
        P = O.mask([k for k in comp if k not in fl])
        two_float[O.pinnable(D, P)] += 1
out["two_floating_corners_pinnable_count"] = dict(two_float)

# 5. clause couplings: F6, and exists private bit: relation = NAE with negations, for all 8 patterns
DOMS = [(0, 1), (2, 6), (3, 7), (4, 5)]
cl_ok = {}
for negs in itertools.product((0, 1), repeat=3):
    W = G.clause_coupling46(negs)
    Gm = O.gamma(W)
    Ds = [O.mask(d) for d in DOMS]
    rel = set()
    for t in itertools.product((0, 1), repeat=4):
        m = 0
        for i in range(4):
            if t[i]:
                m ^= Ds[i]
        if m in Gm:
            rel.add(t[:3])
    want = {t for t in itertools.product((0, 1), repeat=3) if len({t[i] ^ negs[i] for i in range(3)}) > 1}
    # mixed matching: the four dominoes partition the corners
    part = sorted(k for d in DOMS for k in d) == list(range(8)) and all(bin(a ^ b).count("1") == 1 for a, b in DOMS)
    cl_ok[negs] = (O.cls(W), rel == want, part)
out["clause"] = cl_ok
assert all(c == "F6" and r and p for c, r, p in cl_ok.values())


# 6. O_h orbits inside the ferromagnetic gauge (support of Hamze's planted ensemble: one global gauge)
def oh():
    ps = []
    for p in itertools.permutations(range(3)):
        for r in itertools.product((0, 1), repeat=3):
            ps.append(tuple(sum(((O.CORN[k][p[i]] ^ r[i]) << i) for i in range(3)) for k in range(8)))
    return ps


EIDX = {e: i for i, e in enumerate(O.EDGES)}


def permW(W, pm):
    out_ = [0] * 12
    for i, (a, b) in enumerate(O.EDGES):
        x, y = pm[a], pm[b]
        out_[EIDX[(min(x, y), max(x, y))]] = W[i]
    return tuple(out_)


orb = {}
for c in ("F42", "F6"):
    ferro = {W for W, Gm in zip(O.ALL_W, G_all) if O.cls(W) == c and 0 in Gm}
    W0 = min(ferro)
    o = {permW(W0, pm) for pm in oh()}
    orb[c] = dict(ferro_gauge_members=len(ferro), oh_orbit_of_one=len(o), single_orbit=o == ferro)
out["ferro_gauge_orbits"] = orb
for k, v in out.items():
    print(f"{k}: {v}")
