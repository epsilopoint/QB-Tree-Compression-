"""
Experiment 1 (Path A, TTN-native) -- Yukawa radial tensor on a balanced binary tree.

A d-mode Yukawa/screened-Coulomb radial tensor is built, converted once to a
bounded-rank input TTN, and compressed by every method.  Produces the standard
4-panel figure (see experiment_plots.py).
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

from tree_sketch import build_balanced_binary_tree, all_nodes, subtree_leaves
from ttn_format import TTN
from ttn_norms import frobenius_norm
from synthetic import yukawa_radial_tensor
from experiment_plots import run_methods, plot_four_panels, set_scale_mode, METHODS


def run_experiment_1(
    n: int = 8,
    d: int = 8,
    kappa: float = 1.0,
    radius: float = 1.0,
    ranks=(2, 4, 6, 8, 10, 12),
    trials: int = 5,
    R_input: int = 20,
    seed: int = 0,
    save_path: str = "Picture/exp1.png",
    large_scale: bool = False,
    exclude=(),
):
    """Yukawa radial tensor on a balanced binary tree of d leaves.

    The Yukawa kernel is smooth, so all methods reach ~1e-9 at modest rank.
    large_scale only switches TTN-HMT's dense Gaussian sketch to its
    n_above-free Khatri-Rao form for large n; every other method (including
    the full-precision TTN-SVD) is n_above-free either way.  ``exclude`` skips
    methods by name (keys of METHODS).
    """
    set_scale_mode(large_scale)
    exclude = set(exclude)
    methods_run = {k: v for k, v in METHODS.items() if k not in exclude}
    if exclude:
        print(f"Excluding methods: {', '.join(sorted(exclude))}")
    T = yukawa_radial_tensor(d=d, n=n, kappa=kappa, radius=radius)

    tree = build_balanced_binary_tree(d)
    N_steps = len(all_nodes(tree)) - 1
    n_leaves = len(subtree_leaves(tree))
    ttn_in = TTN.from_dense(T, tree, max_rank=R_input)

    gap = float(np.linalg.norm(T - ttn_in.to_dense()) / np.linalg.norm(T))
    print(f"Yukawa d={d}, n={n}, kappa={kappa}, radius={radius}")
    print(f"  ||T||_F = {frobenius_norm(ttn_in):.4f}, R_input={R_input}, "
          f"input gap={gap:.2e}, N_steps={N_steps}")

    results = run_methods(ttn_in, ranks, n_leaves, trials=trials, seed=seed,
                          methods=methods_run)
    plot_four_panels(
        results, ranks, N_steps,
        suptitle=(f"Experiment 1 -- Yukawa radial tensor "
                  f"(d={d}, n={n}, $\\kappa$={kappa}), balanced binary tree"),
        save_path=save_path,
    )
    return results


if __name__ == "__main__":
    run_experiment_1()
