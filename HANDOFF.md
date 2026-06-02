# QBTC / TreeStack — project handoff

This archive is the **working code + experiments** for the QB Tree Compression
(QBTC) thesis chapter. It is self-contained: the full library source is here. This file maps the
codebase.

---

## 1. What the project is

Randomized rounding of tensors stored in **tree tensor network (TTN)** format.
Everything is **"Path A" = TTN-native**: methods take a TTN in and return a TTN
out, and cost is bounded by bond dimensions, *not* by the ambient dimension
`n^d`. The dense tensor is never formed (except inside `from_dense`, used only
to build small test inputs).

**QBTC** = leaves-to-root sweep that, at each node, (1) sketches the node
unfolding with a structured random tensor network, (2) thin-QR the sketch,
(3) extracts a rank-`r` isometry, (4) contracts it into the residual. Two
finishes: **plain-QB** (keep leading `r` columns) and **QB+SVD** (rotate onto
the top-`r` singular directions of the projected unfolding). Two sketch
families: **TreeStack** (topology-matched, on the actual tree) and **TTStack**
(path/tensor-train sketch). Naming: `QB-Tree`, `QB-Train`, `QB-TreeSVD`,
`QB-TrainSVD`.

Baselines compared against: **TTN-SVD** (deterministic hierarchical SVD, the
optimal reference), **TTN-HMT** (per-node randomized range finder, Halko–
Martinsson–Tropp), **TTNN** (Bucci–Verzella tree tensor network Nyström,
arXiv:2412.06111).

---

## 2. File map

```
ttn_format.py        CORE. TTN class, from_dense, qbtc_ttn, structured sketches,
                     residual update, above-Gram, materialize helpers.   (~2150 ln)
ttn_baselines.py     ttn_svd, ttn_hmt, ttnn.                              (~540 ln)
ttn_norms.py         TTN-native Frobenius norm / inner product / diff norm.(~340 ln)
tree_sketch.py       Node class + tree builders (figure1, balanced binary).(~160 ln)
cbc_tt.py            Cholesky-based compression (CBC) experiment helper.

experiments/
  experiment_plots.py   SHARED plotting/method machinery (METHODS, run_methods,
                        plot_four_panels, set_scale_mode). All experiments use it.
  experiment_1.py       Yukawa radial tensor, balanced binary tree (d=8).
  experiment_2.py       Synthetic prescribed-decay TTN, Figure-1 tree (6 leaves).
  experiment_3.py       Synthetic prescribed-decay TTN, balanced binary (d=4).
  experiment_4.py       CP-synthetic tensor, Figure-1 tree.
  synthetic.py          Test-tensor generators (see §5).
  sanity_check_hilbert_6d.py   6D Hilbert tensor, Figure-1 tree, 4-panel figure.
  experiment_cbc_pathological.py   (peripheral: CBC pathological TT; two q-regimes,
                                    CBC vs TT-SVD vs QBTC on the TT-graph caterpillar)
  README.md             Panel/budget/encoding documentation.

README.md, requirements.txt, .gitignore, Picture/
```

**Experiments.** Experiment 1 (Yukawa, balanced binary d=8); Experiment 2
(synthetic prescribed-decay TTN, Figure-1 tree); Experiment 3 (synthetic
prescribed-decay TTN, balanced binary d=4); Experiment 4 (CP-synthetic,
Figure-1 tree). All four share `experiment_plots.METHODS` and take
`large_scale` and `exclude` arguments. Experiment 3 defaults:
`n=100, d=4, R_input=80, decays=("quadratic",), ranks=(10,20,30,40,50)`.

---

## 3. Core API (signatures + meaning)

### `tree_sketch.py`
- `Node` — tree node (has `.children`, `.parent`, `.is_leaf`, `.physical_axis`, `.name`).
- `build_figure1_tree()` → 6-leaf irregular tree (the chapter's Figure-1 tree).
- `build_balanced_binary_tree(d)` → balanced binary tree with `d` leaves.
- `all_nodes`, `subtree_leaves`, `post_order`, `compression_order` — traversals.

### `ttn_format.py`
- `class TTN(root, cores)` — `cores` is `{Node: ndarray}`. Leaf core shape
  `(R_parent, n)`; internal `(R_parent, R_c1, …, R_cm)`; root `(R_c1, …, R_cm)`.
- `TTN.from_dense(T, root, max_rank=...)` — build a TTN from a dense array
  (test inputs only).
- `qbtc_ttn(ttn_input, target_r, finish="qb_svd", P=2, R_per_copy=None,`
  `sketch_kind="gaussian", above_mode="qr", rng_seed=None)` — **the algorithm.**
  - `finish`: `"qb"` (plain) | `"qb_svd"` (**SVD-of-Y**, n_above-free, the
    default) | `"qb_svd_exact"` (former Gram finish, retired/reference only) |
    `"qb_cbc"`.
  - `sketch_kind`: `"gaussian"` (dense, materializes n_above) | `"kr"` (Khatri-Rao,
    structured) | `"treestack"` | `"ttstack"` (both structured).
  - `P`, `R_per_copy`: stacked copies and per-copy bond dim; total sketch dim `PR`.
  - `above_mode`: `"qr"` (builds `above_V`) | `"gram"` (structured above-Gram,
    **no n_above**, ~1e-8 floor). Used ONLY by `qb_svd_exact` and by `ttn_svd`;
    `qb_svd` (SVD-of-Y) and `qb`/`qb_cbc` ignore it. See §4.
- `compute_above_gram(v)` — structured `R_v×R_v` above-Gram, **never forms n_above**.
- `apply_unfolding_KR / _TS / _TT_struct` — structured sketch contractions.
- `_materialize_above(v, leaf_override=None, order_only=False)` — builds the
  `R_v × n_above` environment. `order_only=True` returns only the column
  ordering with **no array** (used by structured paths to avoid the blowup).
- `residual_update(v, Q_v)` — contract committed isometry into the residual.

### `ttn_baselines.py`
- `ttn_svd(ttn_in, target_r, above_mode="orth")` — deterministic hierarchical SVD
  (full-precision canonical form; `"qr"`/`"gram"` kept for comparison).
- `ttn_hmt(ttn_in, target_r, oversample=0, sketch="gaussian", above_mode="qr",
  rng_seed=None)` — per-node randomized range finder. `sketch="kr"` +
  `above_mode="gram"` ⇒ fully n_above-free (and matches Bucci's KR sketch).
- `ttnn(ttn_in, target_r, p=0, rng_seed=None)` — Bucci–Verzella TTNN
  (one-pass two-sided Nyström). Already n_above-free.

### `ttn_norms.py` (all TTN-native, bond-bounded, no dense `X̂`)
- `frobenius_norm(A)` — `√⟨A,A⟩` via tree contraction.
- `frobenius_diff_norm(A, B)` — `‖A−B‖_F` via explicit difference TTN +
  leaves-to-root QR canonicalization (stable; no catastrophic cancellation).
- `ttn_inner`, `compute_below_gram`, `representative_spectrum`.

---

## 4. TTN-SVD back-ends (`above_mode`)

TTN-SVD is the deterministic, quasi-optimal reference. The default
`above_mode="orth"` is a **full-precision, n-independent** canonical hierarchical
SVD (`_hsvd_orth`): root-orthogonalise the cores (leaves→root QR), then truncate
root→leaves moving the orthogonality centre, so every truncation is an SVD of a
bond-sized core. No Gram, no `n^(d-1)` — backward stable (resolves singular
values down to `≈ε·σ_max`) AND bounded by bond dims. **Use this everywhere;** it
is what the experiments use at every scale, with no precision floor.

Two legacy back-ends are kept only for comparison. Both recompute the bond-sized
factor `R_a` with `R_aᵀR_a = G := above_V·above_Vᵀ` (the above-Gram) and then SVD
the small `B = subtree_Vᵀ·R_aᵀ` (whose singular values equal `σ(M_v)`):

- **`above_mode="qr"`:** QR of the dense `above_V` (`R_v × n^(d-1)`). Backward
  stable, but **memory `∝ n^(d-1)`** — on the 6-leaf tree a leaf has
  `n_above = n^5` (~14 GB at n=30, ~175 GB at n=50). Dies on wide trees at large n.
- **`above_mode="gram"`:** `G = compute_above_gram(v)` (structured, `O(R²)`,
  **never forms n_above**), then `R_a` = symmetric square root via `eigh`.
  Bounded memory, **but forming `G` squares the spectrum**, so directions below
  `√ε·σ_max ≈ 1e-8` are lost → a ~1e-8 floor. Kept only to *exhibit* that floor.

`TTN-HMT` additionally needs a structured **sketch** to be n_above-free: dense
Gaussian forces the unfolding; `sketch="kr"` does not (Bucci uses KR at scale).
This is the only thing `set_scale_mode(large=True)` now changes.

`experiment_plots.set_scale_mode(large: bool)` flips both globals at once:
`large=True` ⇒ `ABOVE_MODE="gram"`, `HMT_SKETCH="kr"` (n_above-free, Bucci
scale). Default/`False` ⇒ `"qr"`, `"gaussian"` (precise). Each experiment calls
it at entry so the mode never leaks across runs: every `experiment_N` takes a
`large_scale` flag (default `False`); `sanity_check_hilbert_6d` pins `False`.

---

## 5. Test-tensor generators (`experiments/synthetic.py`)

- `cp_synthetic(shape, K, sigma, rng)` — random rank-`K` CP tensor, weights
  `σ_k`. **Requires every mode dim ≥ K** (a CP factor matrix needs K independent
  columns). Dense build.
- `yukawa_radial_tensor(d, n, kappa, radius)` — screened-Coulomb radial tensor
  `exp(-κr)/r` on a grid. Smooth, fast spectrum. Dense build.
- `synthetic_ttn_with_decay(tree, n, R_input, sigma, rng)` — **builds the dense
  tensor** from a prescribed per-edge spectrum (needs `n ≥ R_input`). Avoid at
  large n on wide trees.
- `synthetic_ttn_decay_object(tree, n, R_input, sigma, rng)` — **builds the TTN
  directly** (cores only, never dense). Use this for large-n / Bucci-scale runs.
  Experiment 2/3 use this internally.

Decay profiles used: quadratic `σ_i=1/i²` (hard), cubic `1/i³`, geometric
`1/2^i` (easy). Quadratic is the kept profile.

---

## 6. Experiment machinery (`experiments/experiment_plots.py`)

- `METHODS` dict: name → `(builder(ttn_in, r, seed, d) → TTN, is_deterministic)`.
- `set_scale_mode(large)` — see §4.
- `run_methods(ttn_in, ranks, n_leaves, trials=5, seed=0)` → results dict
  `{name: {"mean":[...], "sd":[...]}}` (relative Frobenius error per rank).
- `plot_four_panels(results, ranks, N_steps, suptitle, save_path)` → 2×2 figure.

**Budget rule:** `R_per_copy = d` (= #leaves), `P = ⌈PR_target/d⌉`,
`PR_target ∈ {r, 1.2r, 2r}`. Note `PR=r` and `PR=1.2r` collapse to the same `P`
at many ranks when `d` is large (e.g. d=8) — visible only at d=4.

**The four panels:**
1. Accuracy — QB-TreeSVD(1.2r) vs TTN-HMT(p=3), TTNN(p=3), TTN-SVD.
2. Sketch topology (plain QB, PR=r) — QB-Tree vs QB-Train vs TTN-SVD.
3. QB finish (Tree, PR=1.2r) — QB-Tree vs QB-TreeSVD vs TTN-SVD.
4. Oversampling (QB-TreeSVD) — PR=r vs PR=1.2r vs TTN-SVD.

Style: Okabe–Ito, all solid lines; TTN-SVD black `^`; randomized methods get
±1σ shaded bands.

---

## 7. Implementation notes and design decisions

1. **`qb_svd` Gram-squaring bug → fixed.** The SVD finish used to eigendecompose
   `B Bᵀ = A G Aᵀ`, squaring the spectrum and floored the error at ~1e-8 at high
   rank (the QB-TreeSVD "jump"). Rewrote to QR of `above_V` + SVD of the small
   `C = A·R_aᵀ` (no squaring). The `gram` mode is the *opt-in* squaring path for
   scale (§4) — it reintroduces the 1e-8 floor deliberately and only when chosen.
2. **`ttn_svd` Gram-squaring → fixed earlier** the same way (QR-based).
3. **n_above-free ordering fix.** `ttnn`, `build_ttstack_omega`,
   `apply_unfolding_TT_struct` used to call `_materialize_above(v)` just for the
   column ordering, building the `n^(d-1)` array and discarding it. Added
   `order_only=True` fast path (pure tree bookkeeping, no array). TTNN, QB-Tree,
   QB-Train are now genuinely n_above-free.
4. **`above_mode="gram"` + KR HMT** added so the *full* method set runs at Bucci
   scale (n=500, 6-mode tree, 8 GB). Verified: gram ≡ qr to machine precision on
   slow decay; full stack runs at n=150 on the 6-mode tree with the array path
   forbidden.
5. **TTN-HMT vs TTN-SVD coincidence on Hilbert** explained: with oversample p=3
   the HMT subspace saturates on the fast Hilbert spectrum, so it matches
   TTN-SVD; the gap only shows on slow-decay tensors or at p=0 (Bucci's setting).
6. Plot/panel restyling iterations (strong blue for PR=r curves; panel-1 uses
   QB-TreeSVD; panel-4 uses PR=r vs PR=1.2r).
7. **`qb_svd` → SVD-of-Y.** The default SVD finish now takes the top-r left
   singular vectors of the range sketch Y itself (`Q_full @ U_R`, a one-pass
   randomized SVD of `B=Q^T M_v` through Omega). n_above-free, no R^4 Gram, full
   precision. The former exact (above-Gram) finish is kept as `qb_svd_exact`.
8. **`ttstack` contraction.** `apply_unfolding_TT_struct` assembles the
   above-network as a single einsum instead of the hand-rolled DFS (old DFS
   kept as `apply_unfolding_TT_struct_legacy`). The contraction *order* comes
   from `contract_network` (fix 10). numpy's own `greedy` was adequate on
   Figure-1, but on a branching tree (balanced binary, d=8) it picks a
   memory-cheap but flop-catastrophic order (~1.9e13 flops → multi-hour hang);
   `opt_einsum` finds a bond-bounded path (~1.4e6 flops). QB-Train is therefore
   bond-bounded (~QB-Tree cost) on every tree shape.
9. **TTNN structured row sketch.** `ttnn` previously used a DENSE row Gaussian
   `Y_v` of shape `(n_subtree(v), r+p)` = `n^(#leaves under v)` (n^3 on
   Figure-1), so it was n-dependent and crawled at large n (~10 min at n=200).
   Now BOTH sketches are Khatri-Rao; the row pass uses `S_v = subtree_V·Y_v`
   (recursive) and the child projector `P_c = S_c·Z_c`, so n_subtree is never
   formed. Bond-bounded, n-independent (n=200: ~0.5 s, ~140 MB). Old dense-row
   variant kept as `ttnn_dense` (with `col_factors`/`row_sketch` injection hooks
   used to cross-check `ttnn` to machine precision).
10. **Contraction paths standardized on `opt_einsum`.** Every multi-operand,
   path-choosing einsum (per-node QB-Tree sketch + Gram in `ttn_format`, TTNN
   core contractions in `ttn_baselines`, inner-product Grams in `ttn_norms`)
   routes through `contract_network` in `ttn_format`, which uses `opt_einsum`'s
   flop-minimizing path with a numpy fallback if `opt_einsum` is absent.
   Verified numerically identical to the previous code; `opt_einsum` added to
   `requirements.txt`.
11. **Full-precision canonical TTN-SVD (`above_mode="orth"`, new default).**
   The old default (`"qr"`) materialised the `n^(d-1)` above-environment, and
   the scale fallback (`"gram"`) squared the spectrum → a ~1e-8 floor that made
   the *deterministic* reference look worse than the randomised methods on
   fast-decaying spectra at high rank (e.g. exponential decay, R_input=80, r=60:
   TTN-SVD floored at ~2e-8 while QB-Tree reached ~1e-14). Added `_hsvd_orth`: a
   two-phase canonical-form hierarchical SVD (leaves→root QR to root-orthogonal
   form, then a root→leaves orthogonality-centre sweep truncating each edge by
   an SVD of a bond-sized core). No Gram, no `n_above`; backward stable and
   n-independent (n=200, r=20: 0.06 s). Verified: exact reconstruction to ~1e-14,
   matches `"qr"` to 3 digits, reaches the analytic tail (`≤ √N·tail`), and
   restores TTN-SVD as the smallest error at every rank. `set_scale_mode` no
   longer switches TTN-SVD (only the gaussian→KR HMT sketch).
12. **`qb_svd` finish made never-worse-than-`qb` down to machine precision.**
   The SVD-of-Y finish shrank the committed rank to the *numerical* rank of Y
   (`r_keep = #{sv > sigma_1*eps*maxshape}`). On fast-decaying spectra at high
   target rank the trailing singular directions of Y fall below that threshold,
   so QB-TreeSVD committed a lower-rank core than plain QB (which keeps all r),
   leaving a residual ~10-50x larger at the fp floor (e.g. exp decay R_input=80,
   r=60: QB-TreeSVD 5.6e-13 vs QB-Tree 1.5e-14) -- looking like QB+SVD was worse
   than plain QB, which the theory (Rem. "QB+SVD is no worse") forbids in exact
   arithmetic. Fix: keep `min(target_r, rank(Y))` columns (same as plain QB), and
   when no subspace selection actually happens (PR = r, the rotation spans all of
   col(Y) and is inert) use the QR basis directly so QB-TreeSVD coincides EXACTLY
   with QB-Tree. Verified: at PR = r they now match to the digit; at PR = 1.2r
   QB-TreeSVD <= QB-Tree at every rank; the lone residual gap is r=70/PR=72 at
   1.4x of 5e-15 (oversampled selection rounding, both already exact). Confirmed
   a finite-precision floor effect via the algebraic-decay control (quadratic
   sigma_i = i^-2 stays above the floor and shows no reversal at any rank).

---

## 8. Open items / known limitations

- **~~`above_mode="gram"` squares → 1e-8 floor~~ — RESOLVED (fix 11).** The
  default `above_mode="orth"` is a full-precision, n-independent canonical
  hierarchical SVD (orthogonality-centre sweep, SVD of bond-sized cores; §4).
  It needs neither `n_above` nor the Gram, so there is no scale-vs-precision
  tradeoff. The `"gram"` back-end is retained only to exhibit the floor.
- **STTNN** (Bucci's sequential Nyström) has no Path-A port; absent from experiments.
- **Literature positioning (raised, not yet written):** QBTC's plain-QB sweep is
  essentially the **Randomize-then-Orthogonalize** of Al Daas, Ballard et al.
  (arXiv:2110.04393, SIAM JSC 2023) generalized from TT to trees. **Cite it and
  confront it directly.** Defensible novelty = (i) TreeStack sketch on a general
  tree + its OSE proof, (ii) the orthogonal-error-decomposition additive `√N`
  bound for arbitrary topology, (iii) one-sided QB complement to two-sided TTNN.
  Suggested forward-citation check: papers citing 2110.04393 and 2412.06111.

---

## 9. Quick start

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments")
from tree_sketch import build_balanced_binary_tree, subtree_leaves
from synthetic import synthetic_ttn_decay_object
from ttn_format import qbtc_ttn
from ttn_baselines import ttn_svd, ttn_hmt, ttnn
from ttn_norms import frobenius_diff_norm, frobenius_norm
import numpy as np

tree = build_balanced_binary_tree(4); d = len(subtree_leaves(tree))
sigma = 1.0/np.arange(1, 81)**2
ttn = synthetic_ttn_decay_object(tree, n=100, R_input=80, sigma=sigma,
                                 rng=np.random.default_rng(0))
nrm = frobenius_norm(ttn)
xhat = qbtc_ttn(ttn, target_r=30, finish="qb_svd", sketch_kind="treestack",
                P=2, R_per_copy=d, above_mode="qr", rng_seed=0)
print(frobenius_diff_norm(ttn, xhat) / nrm)

# Full figure (balanced-binary synthetic, default n=100):
import experiment_3
experiment_3.run_experiment_3()
# Bucci-scale Figure-1 run:
import experiment_2
experiment_2.run_experiment_2(n=500, R_input=70, decays=("quadratic",),
                              ranks=(10,20,30,40,50,60), trials=5, large_scale=True)
```

Environment: `pip install -r requirements.txt` (numpy, scipy, matplotlib, opt_einsum).
