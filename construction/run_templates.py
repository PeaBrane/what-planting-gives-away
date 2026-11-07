"""Machine check of the block-template lemma and the clause-cluster template (theorem T layout)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402


def main():
    t0 = time.time()
    T = B.templates()
    print(f"block side S={B.S}; {len(T)} oriented (pi=+1) face-pair templates found and checked:")
    for (f1, f2), cells in sorted(T.items()):
        print(f"  in {f1} out {f2}: {len(cells)} cells: {cells}")
    lk, assign, wires, u = B.cluster_template()
    R = B.cluster_region((0, 0, 0))
    print(f"cluster template: clause cell u={u}, region x,y,z ranges {R}")
    print(f"  cells used: {len(lk.used)}; (literal position, port polarity) -> socket: {assign}")
    for ij, w in sorted(wires.items()):
        print(f"  port {ij}: {len(w)} cells: {w}")
    assert all(B.in_region(c, R) for c in lk.used)
    print(f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
