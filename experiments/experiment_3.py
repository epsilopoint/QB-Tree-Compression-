"""
Experiment 3 (Path A, TTN-native) -- synthetic TTN with prescribed core decay on a
balanced binary tree.  One 4-panel figure per decay regime.

Same generator as Experiment 2 but on a balanced binary tree of d leaves with
larger physical dimension n, stressing the bounded-rank pipeline.
"""

import os
import sys
import time

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

from tree_sketch import build_balanced_binary_tree, all_nodes, subtree_leaves
from ttn_format import TTN
from ttn_norms import frobenius_norm
from synthetic import synthetic_ttn_decay_object
from experiment_plots import run_methods, plot_four_panels, set_scale_mode, METHODS


def _decay_sigma(decay: str, R_input: int) -> np.ndarray:
    i = np.arange(1, R_input + 1, dtype=np.float64)
    if decay == "quadratic":
        return 1.0 / i ** 2
    if decay == "cubic":
        return 1.0 / i ** 3
    if decay == "exponential":
        return 1.0 / 2.0 ** i
    raise ValueError(f"Unknown decay '{decay}'")


def _decay_label(decay: str) -> str:
    return {
        "quadratic":   r"$\sigma_i = 1/i^2$ (quadratic)",
        "cubic":       r"$\sigma_i = 1/i^3$ (cubic)",
        "exponential": r"$\sigma_i = 1/2^i$ (geometric)",
    }[decay]


def run_experiment_3(
    n: int = 100,
    d: int = 4,
    R_input: int = 80,
    decays=("quadratic",),
    ranks=(10, 20, 30, 40, 50),
    trials: int = 5,
    seed: int = 0,
    save_name: str = "experiment_3",
    large_scale: bool = False,
    exclude=(),
):
    """Synthetic TTN with prescribed decay on a balanced binary tree, one figure per decay.

    large_scale=True switches TTN-HMT's dense Gaussian sketch to its n_above-free
    Khatri-Rao form for big n; every other method (including the full-precision
    TTN-SVD) is n_above-free regardless, with no precision floor.

    ``exclude`` is a list of method names (keys of METHODS) to skip.
    """
    set_scale_mode(large_scale)
    exclude = set(exclude)
    methods_run = {k: v for k, v in METHODS.items() if k not in exclude}
    if exclude:
        print(f"Excluding methods: {', '.join(sorted(exclude))}")
    if R_input < max(ranks):
        raise ValueError(f"R_input ({R_input}) must be >= max rank ({max(ranks)}).")
    if n < R_input:
        raise ValueError(f"n ({n}) must be >= R_input ({R_input}).")

    tree = build_balanced_binary_tree(d)
    N_steps = len(all_nodes(tree)) - 1
    n_leaves = len(subtree_leaves(tree))
    print(f"Experiment 3: n={n}, d={d}, R_input={R_input}, N_steps={N_steps}, "
          f"balanced binary tree, ranks={tuple(ranks)}")

    results_by_decay = {}
    for decay in decays:
        print(f"\n===== decay: {decay} =====")
        sigma = _decay_sigma(decay, R_input)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        # Build directly in TTN format (cores only): no dense (n,)**d array.
        ttn_in = synthetic_ttn_decay_object(tree, n=n, R_input=R_input, sigma=sigma, rng=rng)
        print(f"  build {time.time()-t0:.1f}s, ||T||_F = {frobenius_norm(ttn_in):.4e}, "
              f"input gap=0 (exact TTN-rank {R_input})")

        results = run_methods(ttn_in, ranks, n_leaves, trials=trials, seed=seed,
                              methods=methods_run)
        plot_four_panels(
            results, ranks, N_steps,
            suptitle=(f"Experiment 3 -- synthetic TTN, {_decay_label(decay)}, "
                      f"balanced binary tree (d={d}, n={n})"),
            save_path=f"Picture/{save_name}_{decay}.png",
        )
        results_by_decay[decay] = results
    return results_by_decay


if __name__ == "__main__":
    run_experiment_3()
