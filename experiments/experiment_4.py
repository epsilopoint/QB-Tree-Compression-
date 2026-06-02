"""
Experiment 4 (Path A, TTN-native) -- CP-synthetic tensor on the Figure-1 tree.

A random rank-K CP tensor with algebraically decaying weights sigma_k = k^{-alpha}
is built, converted once to a bounded-rank input TTN, and compressed by every
method.  Produces the standard 4-panel figure (see experiment_plots.py):
accuracy, sketch topology, QB finish, oversampling.
"""

import os
import sys

import numpy as np


def _repo_root():
    try:
        start = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start = os.getcwd()
    d = start
    for _ in range(6):
        if os.path.exists(os.path.join(d, "ttn_format.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


sys.path.insert(0, _repo_root())
sys.path.insert(0, os.path.join(_repo_root(), "experiments"))

from tree_sketch import build_figure1_tree, all_nodes, subtree_leaves
from ttn_format import TTN
from ttn_norms import frobenius_norm
from synthetic import cp_synthetic
from experiment_plots import run_methods, plot_four_panels, set_scale_mode, METHODS


def run_experiment_4(
    n: int = 20,
    K: int = 20,
    alpha: float = 0.5,
    ranks=(2, 4, 6, 8, 10, 12),
    trials: int = 5,
    R_input: int = 20,
    seed: int = 0,
    save_path: str = "Picture/exp4.png",
    large_scale: bool = False,
    exclude=(),
):
    """CP-synthetic tensor on the Figure-1 tree.  R_input caps the input bond
    rank; the CP edge rank is <= K, so R_input >= K makes the input TTN almost
    exact (the gap is reported).  Target ranks must stay below R_input.

    large_scale=True switches TTN-HMT's dense Gaussian sketch to its n_above-free
    Khatri-Rao form for big n; TTN-SVD and the QB methods are n_above-free
    regardless.  ``exclude`` skips methods by name (keys of METHODS)."""
    assert max(ranks) < R_input, "target ranks must be below the input bond rank"
    set_scale_mode(large_scale)
    exclude = set(exclude)
    methods_run = {k: v for k, v in METHODS.items() if k not in exclude}
    if exclude:
        print(f"Excluding methods: {', '.join(sorted(exclude))}")

    rng = np.random.default_rng(seed)
    sigma = (np.arange(1, K + 1, dtype=np.float64)) ** (-alpha)
    print(f"Generating CP tensor: shape={(n,)*6}, K={K}, alpha={alpha}")
    T = cp_synthetic(shape=(n,) * 6, K=K, sigma=sigma, rng=rng)

    tree = build_figure1_tree()
    N_steps = len(all_nodes(tree)) - 1
    n_leaves = len(subtree_leaves(tree))            # compression steps (= 9 here)
    ttn_in = TTN.from_dense(T, tree, max_rank=R_input)

    gap = float(np.linalg.norm(T - ttn_in.to_dense()) / np.linalg.norm(T))
    print(f"  ||T||_F = {frobenius_norm(ttn_in):.4f}, R_input={R_input}, "
          f"input gap={gap:.2e}, N_steps={N_steps}")

    results = run_methods(ttn_in, ranks, n_leaves, trials=trials, seed=seed,
                          methods=methods_run)
    plot_four_panels(
        results, ranks, N_steps,
        suptitle=(f"Experiment 4 -- CP-synthetic tensor "
                  f"(d=6, n={n}, K={K}, $\\alpha$={alpha}), Figure-1 tree"),
        save_path=save_path,
    )
    return results


if __name__ == "__main__":
    run_experiment_4()
