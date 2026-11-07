# What a Planted Solution Gives Away: verification code

This code accompanies the manuscript *What a Planted Solution Gives Away: Known Energies, Hard Witnesses, and Planted
Spin Glasses on the Cubic Lattice* (Yan Ru Pei, 2026). The proofs in the paper are by hand except where a finite
object is checked by enumeration (the paper's Appendix B). This repository holds those checks, two independent
deciders for the end-to-end tests, and the small numerical experiments of Section 6.

Every check is exact (integer or bit arithmetic, or an LP or MILP whose answer is re-checked exactly), and each
recorded output (`*_out.txt`) is the log of the run cited in the paper.

## Requirements

Python 3.10 or later with `numpy` and `scipy` (`pip install -r requirements.txt`). The runs below were made with
Python 3.13, numpy 2.5 and scipy 1.18. Each script runs from its own directory.

## Layout and what each script checks

| Paper | Script | What it does |
|---|---|---|
| Lemma A.1 and the class table of Sec. 4.2; Lemma A.2; Thm 4.7(1) | `independent/cell_catalogue.py`, `construction/catalogue.py` | enumerate all 4096 ±1 cube couplings: ground-state sets, Hamze classes, T-join duality, the clause cell, affinity of classes with at most two ground-state pairs |
| Cor. 2.4, Prop. 3.2 | `construction/loopcone_q3.py`, `independent/cell_catalogue.py` | the 63 planted ±1 cube couplings lie in the loop cone of Q3 (LP feasibility over 168 loop atoms); the cycle LP equals the cell minimum for all 4096 couplings |
| Lemmas A.4, A.5 (domino rule, chirality, block and cluster templates) | `construction/run_templates.py`, `independent/geometry_check.py`, `independent/cross_block_contacts.py`, `independent/rep_and_polarity_check.py` | find and verify the 30 oriented block templates and the 195-cell clause cluster; recheck them, the gluing across faces and the port polarities with independent code |
| Thm 4.3 end to end (construction's verifier) | `construction/verify_blocks.py` (layout A), `construction/verify_T.py`, `construction/verify_T_rand.py` | build the layout-A instance for a formula, check typing and network connectivity, decide TILE by parity union-find plus exact search, compare with NAE-satisfiability, read off and pin assignments |
| Thm 4.3 end to end (independent decider) | `independent/my_decider.py`, `independent/e2e_check.py`, `independent/test_decider.py`, `independent/pick_formulas.py`, `independent/formulas.json` | a decider written from the definitions only (it imports nothing from `construction/`), cross-checked against a naive backtracker on random small tori; `e2e_check.py` uses the construction code only to generate instances |
| Sec. 5 (Ex. 5.4, lattices, Figure 1) | `frustration/exact_examples.py`, `frustration/bowlin_k4r.py`, `frustration/torus_cover.py`, `frustration/plaquette_cover.py`, `frustration/frustration_landscape.py` | exact checks of the K5 and K4,4 examples, Bowlin's K4,8 signing, the exact plaquette cover of the cubic torus, and the figure |
| Sec. 6 (exploratory numerics) | `numerics/cycle_lp.py`, `numerics/tile_lp.py`, `numerics/loop_cone.py` | the cycle LP on random frustrated-loop and tile instances; the 4-cycle loop cone of small RBMs |

Docstrings inside `construction/` and `independent/` use the numbering of the construction notes the code was
written against. In the paper's numbering:

| In the code | In the paper |
|---|---|
| Lemma 2.1 (T-join duality), Lemma 2.2 (class table) | Lemma A.1 and the class table of Sec. 4.2 |
| Lemma 2.3 (clause cell) | Lemma A.2 |
| Sec. 3.1, Lemma 3.1 (networks, typing) | Lemma A.3 (network lemma) |
| Lemmas 4.1, 4.2 (domino rule, chirality) | Lemma A.4 |
| Lemmas 4.3–4.5 (templates, gluing, clause cluster) | Lemma A.5 and the gluing remark after it |
| Lemmas 4.6, 4.7, "layout A" | Layout A in Appendix A |
| Theorem T | Theorem 4.3 |
| Theorem B | Theorem 4.7 |
| Corollary 2(b) | Proposition 3.2 |
| Corollary 4 | Corollary 4.6 (via Corollary 2.4) |
| Claim 6 (`bowlin_k4r.py`), Claim 7 (`torus_cover.py`) | Example 5.4; the "Lattices" paragraph of Sec. 5 |

## Running

```bash
cd independent && python cell_catalogue.py        # rewrites cell_catalogue_out.txt
cd construction && python loopcone_q3.py
cd construction && python run_templates.py
cd independent && python geometry_check.py
cd construction && python verify_blocks.py single pair sat3   # construction's verifier, layout A
cd independent && python e2e_check.py formulas.json rand_n4_m4_s1029_unsat 0   # independent decider
cd independent && python test_decider.py
```

Memory grows with L³. The end-to-end runs in the paper go up to four clauses (L = 96 at margin 0), using about
100 MB; the margin-1 runs need more.

## Limits

- End-to-end decisions were run up to four clauses. Layout A was checked for up to 32 clauses at the level of linked
  instances (typing and network connectivity).
- Layout B (the L = O(√m) wiring behind the ETH bound) is a hand proof via de Berg et al.'s Cube Wiring Theorem and
  is not implemented.

## Licence

MIT; see `LICENSE`.
