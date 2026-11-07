"""G46 cluster template at the current TILE_S: contact structure and per-cell self-pinnability (assuming every
corner shared with an F42 or background cell is anchored)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout46 as G  # noqa: E402

t0 = time.time()
lk, assign, wires, u = G.cluster_template46()
f6 = lambda c: G.is_f6_role(lk.role[c])  # noqa: E731
print(f"S={G.S} G46 cluster template: {len(lk.used)} cells, u={u}, (literal, polarity) -> socket {assign}")
hist = {}
for c in lk.used:
    r = lk.role[c]
    if r.startswith("clause") or r.startswith("side"):
        continue
    k = len(G.f6_contacts(lk, c, f6))
    hist[k] = hist.get(k, 0) + 1
print("F6 wire cells by number of F6 contacts:", dict(sorted(hist.items())))
print("not self-pinnable:", G.unpinnable_cells(lk, list(lk.used)))
for ij, w in sorted(wires.items()):
    print(f"  port {ij}: {len(w)} cells")
print(f"[{time.time()-t0:.1f}s]")
