"""
Experiment 2 (Path A, TTN-native) -- synthetic TTN with prescribed core decay,
Figure-1 tree.  One 4-panel figure per decay regime.

The tensor is generated directly in TTN form with a chosen singular-value decay
sigma_i on every edge, giving full analytic control over the spectrum.  Decays:
quadratic (1/i^2), cubic (1/i^3), geometric (1/2^i).
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


def run_experiment_2(
    n: int = 20,
    R_input: int = 15,
    decays=("quadratic", "cubic", "exponential"),
    ranks=(2, 4, 6, 8, 10, 12, 14),
    trials: int = 5,
    seed: int = 0,
    save_name: str = "experiment_2",
    large_scale: bool = False,
    exclude=(),
):
    """Synthetic TTN with prescribed decay on the Figure-1 tree, one figure per decay.

    Set large_scale=True for Bucci-scale runs (e.g. n=500, R_input=70): TTN-HMT's
    dense Gaussian sketch switches to its n_above-free Khatri-Rao form, so peak
    memory is bounded by bond dimensions instead of n^(d-1).  Every other method,
    including TTN-SVD (full-precision canonical hierarchical SVD), is n_above-free
    regardless, with no precision floor.  Leave False for small n.

    ``exclude`` is a list of method names (keys of METHODS) to skip; rarely
    needed now.  (Historically "QB-Train (PR=r)", the path-graph ttstack sketch,
    OOMed on a branching tree because its contraction stacked R_input^(#leaves)
    axes.  apply_unfolding_TT_struct now routes that contraction through
    contract_network (a flop-minimizing opt_einsum path), so its peak cost is
    bond-bounded -- the same as QB-Tree -- and it runs at scale without exclusion.)
    """
    if R_input < max(ranks):
        raise ValueError(f"R_input ({R_input}) must be >= max rank ({max(ranks)}).")
    if n < R_input:
        raise ValueError(f"n ({n}) must be >= R_input ({R_input}).")

    set_scale_mode(large_scale)
    exclude = set(exclude)
    methods_run = {k: v for k, v in METHODS.items() if k not in exclude}
    if exclude:
        print(f"Excluding methods: {', '.join(sorted(exclude))}")

    tree = build_figure1_tree()
    N_steps = len(all_nodes(tree)) - 1
    n_leaves = len(subtree_leaves(tree))
    print(f"Experiment 2: n={n}, R_input={R_input}, N_steps={N_steps}, "
          f"Figure-1 tree, ranks={tuple(ranks)}")

    results_by_decay = {}
    for decay in decays:
        print(f"\n===== decay: {decay} =====")
        sigma = _decay_sigma(decay, R_input)
        rng = np.random.default_rng(seed)
        # Build the input DIRECTLY in TTN format (cores only) -- never forms the
        # dense (n,)**d array, so Bucci-scale n (e.g. 500) is feasible.  The
        # tensor is exactly TTN-rank R_input on every edge, so the input gap is 0.
        ttn_in = synthetic_ttn_decay_object(tree, n=n, R_input=R_input, sigma=sigma, rng=rng)
        print(f"  ||T||_F = {frobenius_norm(ttn_in):.4f}, input gap=0 (exact TTN-rank {R_input})")

        results = run_methods(ttn_in, ranks, n_leaves, trials=trials, seed=seed,
                              methods=methods_run)
        plot_four_panels(
            results, ranks, N_steps,
            suptitle=(f"Experiment 2 -- synthetic TTN, {_decay_label(decay)}, "
                      f"Figure-1 tree (n={n})"),
            save_path=f"Picture/{save_name}_{decay}.png",
        )
        results_by_decay[decay] = results
    return results_by_decay


if __name__ == "__main__":
    run_experiment_2()
