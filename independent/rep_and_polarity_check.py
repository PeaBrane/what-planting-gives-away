"""Check Lemma 2.2's ferromagnetic-gauge representatives and Sec. 6.3's port-polarity claim for build.place_clause."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_decider as D  # noqa: E402
idx = {e: i for i, e in enumerate(D.EDGES)}


def gam(neg):
    W = [1] * 12
    for e in neg:
        W[idx[e]] = -1
    t = D.cell_energy_table(W)
    return sorted(int(m) for m in (t == t.min()).nonzero()[0])


msk = lambda ks: sum(1 << k for k in ks)
print("F21 rep -1 on (0,1):", gam([(0, 1)]), "== [0,255]:", gam([(0, 1)]) == [0, 255])
D04 = msk([0, 4])
print("F41 rep -1 on (0,2),(4,5):", gam([(0, 2), (4, 5)]), "== {0,1_D,1_Dc,255} with D={0,4}:",
      gam([(0, 2), (4, 5)]) == sorted({0, D04, 255 ^ D04, 255}))
F = msk([0, 1, 2, 3])
print("F42 rep -1 on (1,5),(2,6):", gam([(1, 5), (2, 6)]), "== face z=0:", gam([(1, 5), (2, 6)]) == sorted({0, F, 255 ^ F, 255}))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "construction"))
import build  # noqa: E402
inst = build.Instance(32)
ports = build.place_clause(inst, (16, 16, 16), 0, [(0, 0), (1, 0), (2, 0)])
pi = lambda a: a[0] * a[1] * a[2]
for pos, v in enumerate((0, 1, 2)):
    P, Q1, o1, Q2, o2 = ports[f"x{v}"]
    print(f"first-run builder, literal position {pos}: polarity (port1, port2) = ({pi(o1):+d}, {pi(o2):+d})")
