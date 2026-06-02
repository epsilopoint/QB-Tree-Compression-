# QBTC / TreeStack — TTN-native

Tree-tensor-network compression with structured randomized sketches, operating
**directly on the tree (TTN) structure** — no dense materialization of the
working tensor.

This repository implements **QB Tree Compression (QBTC)** — a leaves-to-root
sweep of QB factorizations driven by an *open-leaf TreeStack* sketch — together
with the baselines it is compared against, all in TTN-native form (a `TTN`
object in, a `TTN` object out). It also includes a standalone implementation of
**Cholesky-Based Compression (CBC)** for tensor trains.

> **Path-A only.** An earlier dense-tensor implementation (operating on full
> `np.ndarray` tensors) has been removed. Every compression method here takes
> and returns a `TTN`; cost and memory are bounded by bond dimensions, not by
> the ambient size `n^d`. The one method without a TTN-native port yet is
> **STTNN** (sequential TTN-Nyström) — see *Status* below.

## Layout

```
.
├── ttn_format.py        TTN data structure + qbtc_ttn + TreeStack sketch (TTN-native)
├── ttn_baselines.py     TTN-native baselines: ttn_svd, ttn_hmt, ttnn
├── ttn_norms.py         Frobenius norm / inner product / diff-norm via tree contraction
├── tree_sketch.py       Tree topology (Node, builders, traversals) — topology only
├── cbc_tt.py            Cholesky-Based Compression for tensor trains (standalone)
├── plot_helper.py       Shared two-row by-family plotting
└── experiments/         Larger drivers, incl. sanity_check_hilbert_6d.py
                         (End-to-end TTN-native benchmark on the 6D Hilbert tensor)
```

Run scripts from the repository root; modules import each other by plain name,
so no path configuration is needed:

```bash
pip install -r requirements.txt
python experiments/sanity_check_hilbert_6d.py
```

## Core API

### `ttn_format.py`

| object | description |
|---|---|
| `TTN(root, cores)` | A tree-tensor network: a tree `root` (a `Node`) plus a `{Node: core}` dict. |
| `TTN.from_dense(T, root, max_rank=None, rtol=0.0)` | Build an (exact, full-rank) TTN from a dense tensor over the given tree. |
| `ttn.to_dense()` | Contract the TTN back to a dense array (for checking / small cases). |
| `qbtc_ttn(ttn_input, target_r, finish="qb_svd", P=2, R_per_copy=None, sketch_kind="gaussian", rng_seed=None)` | QB Tree Compression on a TTN. `sketch_kind ∈ {"gaussian","kr","treestack","ttstack"}`, `finish ∈ {"qb","qb_svd","qb_svd_exact","qb_cbc"}` (`qb_svd` is the SVD-of-Y finish; `qb_svd_exact` is the retired Gram finish).

`sketch_kind="treestack"` is the open-leaf TreeStack on the tree topology;
`"ttstack"` is the caterpillar (TT path-graph) topology. Total per-step sketch
budget is `P · R_per_copy`.

### `ttn_baselines.py` — TTN in, TTN out

| function | description |
|---|---|
| `ttn_svd(ttn_in, target_r)` | Exact hierarchical TTN-SVD (the quasi-optimal reference): full-precision canonical form (`above_mode="orth"`) -- no Gram, no `n_above`. |
| `ttn_hmt(ttn_in, target_r, oversample=0, rng_seed=None)` | Per-node randomized range finder. |
| `ttnn(ttn_in, target_r, p=0, rng_seed=None)` | Bucci–Verzella TTN-Nyström. |

### `ttn_norms.py` — norms by tree contraction

| function | description |
|---|---|
| `frobenius_norm(A)` | `‖A‖_F` via tree contraction. |
| `frobenius_diff_norm(A, B)` | `‖A − B‖_F` via root-canonical orthogonalization (stable when the difference is tiny). |
| `ttn_inner(A, B)` | `⟨A, B⟩`. |
| `representative_spectrum(ttn, v)` | Singular values of the unfolding at node `v`, from `R_v × R_v` Grams. |

```python
import numpy as np
from tree_sketch import build_figure1_tree
from ttn_format import TTN, qbtc_ttn
from ttn_baselines import ttn_svd
from ttn_norms import frobenius_diff_norm, frobenius_norm

T      = ...                                   # dense 6-mode test tensor
root   = build_figure1_tree()
ttn_in = TTN.from_dense(T, root)               # exact TTN of the input

ttn_qb  = qbtc_ttn(ttn_in, target_r=8, finish="qb",
                   sketch_kind="treestack", P=2, R_per_copy=4, rng_seed=0)
ttn_svd_out = ttn_svd(ttn_in, target_r=8)

rel_err = frobenius_diff_norm(ttn_in, ttn_qb) / frobenius_norm(ttn_in)   # TTN-native
```

### `cbc_tt.py` — Cholesky-Based Compression for tensor trains

Operates on a TT given as a list of cores `(R_{k-1}, n_k, R_k)`, `R_0=R_d=1`.
`cbc_tt(cores, target_r)`, `tt_svd(cores, target_r)`, `reconstruct_tt(cores)`,
`right_canonicalize_tt(cores)`, `check_right_canonical(cores)`.

## Sanity check

`experiments/sanity_check_hilbert_6d.py` builds the 6D Hilbert tensor
`T[i₁,…,i₆] = 1/(1+i₁+⋯+i₆)`, converts it once with `TTN.from_dense`, runs every
method on the resulting TTN, and measures error TTN-natively with
`frobenius_diff_norm`. It saves a two-row by-family figure to
`Picture/sanity_check.png`. It is a smoke test of the full pipeline, not a
formal benchmark; knobs (`n`, `n_trials`, `ranks`) are at the top of `main()`.

## Status

Verified TTN-native and correct (round-trip against dense ground truth):
**QBTC** (`gaussian` / `treestack` / `ttstack`, `qb` / `qb_svd`),
**TTN-SVD**, **TTN-HMT**, **TTNN**.

Known follow-ups:
- **STTNN** (sequential TTN-Nyström) has no TTN-native port and is omitted from
  the sanity check. Its dense reference implementation is not part of this repo.
- The `qb_svd` finish is the n_above-free **SVD-of-Y** method (see below), so it
  scales like plain `qb`. `QB-Train`'s `ttstack` sketch is contracted with a
  flop-minimizing path (`contract_network`, backed by `opt_einsum`), so its peak
  cost is bond-bounded like `QB-Tree` on every tree shape (see *Contraction
  paths* below). The only paths that still form `n_above` are the dense
  `gaussian` sketch and the retired `qb_svd_exact` finish (kept for reference,
  not used by the experiments).
- All four `experiments/experiment_1..4.py` are TTN-native and share one method
  set (`experiment_plots.METHODS`); each takes `large_scale` and `exclude`
  arguments (see `experiments/README.md`).

### Numerical note — TTN-SVD basis (precision)

`ttn_svd` computes each node basis from an SVD of a small factor, **not** by
eigendecomposing the Gram `M_v M_vᵀ`. Going through the Gram squares the
singular values (`σ → σ²`), so directions with small `σ` fall below machine
epsilon and are resolved as noise — which made the *exact* TTN-SVD curve plateau
*above* the randomized methods at high target rank (e.g. ~4e-10 instead of
~1e-14 at r=12). With the SVD-based basis, TTN-SVD matches the direct dense-SVD
reference and is the best method at every rank, as it must be.

The **`qb_svd` finish** no longer touches a Gram at all. It is the *SVD-of-Y*
method: take the top-`r` left singular vectors of the range sketch `Y` directly
(via the small QR factor, `U_Y = Q_full @ U_R`). By the identity
`Y = Q_full (B Ω)` with `B = Q_fullᵀ M_v`, this is exactly a one-pass randomized
SVD of the projected core `B` through the sketch `Ω`. It is `n_above`-free, has
no `R⁴` Gram cost, and carries no spectrum-squaring floor. At `sketch_dim = r` it
coincides with plain `qb`; with oversampling it improves on `qb`, recovering most
(not all) of the exact-SVD gain. The former exact finish (which aligned `Q_v`
with the true singular vectors of the full unfolding via the `R_v × R_v`
above-Gram, the only one with the sharp `(1+ε)` per-node bound but an `R_input⁴`
cost) is kept as `qb_svd_exact` for reference and is not used by the experiments.

### Contraction paths (`opt_einsum`)

The sketch contractions are multi-operand tensor-network einsums. Every one
that needs a contraction *order* routes through a single helper,
`contract_network` in `ttn_format.py`, which uses `opt_einsum`'s flop-minimizing
path. This matters: numpy's own `einsum` optimizer minimizes intermediate
*memory*, and on a branching tree that can pick an order which is memory-cheap
but flop-catastrophic (≈ `R^(#leaves)` flops). `opt_einsum` returns a
bond-bounded path in milliseconds instead. If `opt_einsum` is not installed the
helper falls back to numpy, which is fine for small or path-like trees but can
be slow on branching ones — so `opt_einsum` is listed as a requirement.

## Requirements

Python 3.10+ (developed and tested on 3.12), with `numpy`, `scipy`,
`matplotlib`, and `opt_einsum` (see `requirements.txt`). `opt_einsum` is
pure-Python and strongly recommended; see *Contraction paths* above.

## Citation

If you use this code, please cite it. The repository ships a `CITATION.cff`, so
GitHub shows a **"Cite this repository"** button with the entry below.

```bibtex
@software{homza_qbtc_treestack,
  author = {Homza, Daniil},
  title  = {{QBTC / TreeStack: TTN-native randomized tensor-network compression}},
  year   = {2026},
  url    = {https://github.com/<your-username>/<your-repo>}
}
```

If you would rather direct citations to the accompanying thesis, adapt:

```bibtex
@mastersthesis{homza_qbtc_thesis,
  author = {Homza, Daniil},
  title  = {{<thesis title>}},
  school = {<institution>},
  year   = {2026},
  type   = {Master's thesis}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

