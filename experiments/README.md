# Experiments

These are larger drivers, set aside from the maintained core
(`ttn_format.py`, `ttn_norms.py`, `ttn_baselines.py`, `qbtc_ttn`). Each adds
the repository root to `sys.path`, so run them from inside this folder:

```bash
cd experiments
python experiment_1.py      # Picture/exp1.png                    (Yukawa, d=8)
python experiment_2.py      # Picture/experiment_2_<decay>.png    (one per decay)
python experiment_3.py      # Picture/experiment_3_<decay>.png    (one per decay)
python experiment_4.py      # Picture/exp4.png                    (CP-synthetic)
```

`sanity_check_hilbert_6d.py` (6D Hilbert tensor, Figure-1 tree, in this folder)
produces the **same** 4-panel figure and is the quickest smoke test of the whole
pipeline.

## Status

All experiments are TTN-native (Path A). `experiment_1` and `experiment_4`
build a dense test tensor and convert it once with
`TTN.from_dense(max_rank=R_input)`; `experiment_2` and `experiment_3` build the
TTN directly (`synthetic_ttn_decay_object`, no dense array, so large `n` is
feasible). All run every method on the resulting TTN and measure error with
`frobenius_diff_norm` (no dense reconstruction). Each takes `large_scale`
(`False` = precise default, `True` = bond-bounded for large `n`) and
`exclude=(...)` to skip methods by name.

- `experiment_1.py` — Yukawa-screened tensor, d=8, balanced binary tree.
- `experiment_2.py` — synthetic TTN with prescribed decay, Figure-1 tree; one figure per decay.
- `experiment_3.py` — synthetic TTN with prescribed decay, balanced binary tree
  (default d=4, n=100); one figure per decay.
- `experiment_4.py` — synthetic CP tensor (`sigma_k = k^-alpha`), Figure-1 tree.
- `experiment_cbc_pathological.py` — CBC pathological example: the track-routed
  `d=6` Prop. 3.2 tensor at two spread regimes (`q=1` tied, `q=1.2` spread),
  comparing CBC (`cbc_tt.py`), TT-SVD, and QBTC (TreeStack on the TT-graph
  caterpillar). Writes `Picture/CBC_q=1.png` and `Picture/CBC_q=12.png`.
- `synthetic.py` — synthetic tensor generators used by `experiment_1..4`.
- `experiment_plots.py` — **shared** method set + 4-panel figure (below).

`cp_synthetic`/`synthetic_ttn_with_decay` require `n >= K` / `n >= R_input`.

## The four-panel figure (`experiment_plots.py`)

Every driver calls `run_methods(ttn_in, ranks, n_leaves, trials, seed)` then
`plot_four_panels(results, ranks, N_steps, suptitle, save_path)`. The figure is
a 2x2 grid; all panels show log-y absolute relative error
`||X - Xhat||_F / ||X||_F`, draw TTN-SVD as a black reference, and shade
randomized methods with +/-1 sigma over `trials` independent sketches.

1. **Accuracy** — TTN-SVD, QB-TreeSVD (PR=1.2r), TTNN(p=3), TTN-HMT(p=3).
2. **Sketch topology** — Tree vs Train, plain-QB at PR=r, + TTN-SVD.
3. **QB finish** — plain-QB vs QB+SVD, Tree at PR=1.2r, + TTN-SVD.
4. **Oversampling** — QB-TreeSVD at PR=r and PR=1.2r, + TTN-SVD.

### Sketch budget (R = d / P = ceil(PR_target/d))

The per-copy sketch bond is fixed at `R = d` (the number of leaves of the tree),
and the number of stacked copies reaches the target total dimension:

      R_per_copy = d
      P          = ceil( PR_target / d ),     PR_target in { r, 1.2r, 2r }
      realized PR = P * d  (>= PR_target)

`d` is read from the tree as `len(subtree_leaves(tree))` (6 for the Figure-1
tree, 8 for the d=8 balanced tree, etc.) and passed to `run_methods` as
`n_leaves`. For `r < d` a single copy already gives `PR = d`, so the `r` and
`2r` curves coincide until `r` is large enough that `ceil(2r/d) > ceil(r/d)`.

### Encoding

Each series has its own colour (Okabe-Ito colourblind-safe palette) and **all
lines are solid** — series are distinguished by colour and marker, never by
linestyle. TTN-SVD black `^`, TTNN vermillion `o`, TTN-HMT purple `v`,
QB-Tree(PR=r) blue `d`, QB-Train orange `s`, QB-Tree(PR=1.2r) blue `d`,
QB-TreeSVD(PR=1.2r) green `D`, QB-TreeSVD(PR=r) blue `D`.

To change which methods appear in a panel, edit `PANELS`; to add/remove a
method edit `METHODS` and `STYLE`. `run_methods(..., methods=...)` accepts a
subset. To re-enable the quasi-optimal band `[sigma_opt, sqrt(N) sigma_opt]` in
a panel, set its `show_band` flag to `True` in `PANELS` (N = `N_steps`).

**STTNN** still has no TTN-native port and is not included.

## Adding a dense-input experiment

1. Build `T` and convert once: `ttn_in = TTN.from_dense(T, root, max_rank=R_input)`.
2. `qbtc_ttn(ttn_in, r, finish="qb"|"qb_svd", sketch_kind="treestack"|"ttstack", ...)`.
3. Use the `ttn_baselines` versions of `ttn_svd / ttn_hmt / ttnn`.
4. Error: `frobenius_diff_norm(ttn_in, ttn_out) / frobenius_norm(ttn_in)`.
