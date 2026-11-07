"""Run verify_T.py (heuristic fine-level router, L=32 by default) on random NAE-3SAT formulas rand:<n>:<m>:<seed>."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_T  # noqa: E402
from verify_blocks import random_formula  # noqa: E402

if __name__ == "__main__":
    names = []
    for a in sys.argv[1:]:
        if a.startswith("rand:"):
            _, n, m, seed = a.split(":")
            verify_T.FORMULAS[a] = random_formula(int(n), int(m), int(seed))
        names.append(a)
    verify_T.main(names)
