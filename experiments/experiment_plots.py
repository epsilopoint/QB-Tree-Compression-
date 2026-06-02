"""
Shared method set and 4-panel plotting for the QBTC experiments (Path A).

Every experiment builds its test tensor, converts it ONCE to a bounded-rank TTN,
then calls ``run_methods`` followed by ``plot_four_panels``.  All methods take a
TTN in and return a TTN out; error is measured TTN-natively with
``frobenius_diff_norm`` (no dense n^d arrays are formed by the methods).

Four panels per experiment (all panels: log-y absolute relative error,
TTN-SVD as a black reference, randomized methods shaded with +/-1 sigma):

  (1) Accuracy        TTN-SVD, QB-TreeSVD (PR=1.2r), TTNN(p=3), TTN-HMT(p=3).
  (2) Sketch topology Tree vs Train, both plain-QB at PR=r, + TTN-SVD.
  (3) QB finish       plain-QB vs QB+SVD, both Tree at PR=1.2r, + TTN-SVD.
  (4) Oversampling    QB-TreeSVD at PR = r and PR = 1.2r, + TTN-SVD.

Sketch budget (NEW scheme): the per-copy bond is fixed at R = d (the number of
leaves of the tree), and the number of stacked copies is chosen to reach the
target total sketch dimension:

      R_per_copy = d
      P          = ceil( PR_target / d )      with  PR_target in { r, 1.2r, 2r }

so the realized total sketch dimension is P * d  (>= PR_target).

Encoding: each curve has its own colour (Okabe-Ito colourblind-safe set); all
lines are solid and series are distinguished by colour and marker.  Baselines:
TTN-SVD black, TTNN vermillion, TTN-HMT purple.
"""

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


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

from ttn_format import qbtc_ttn
from ttn_baselines import ttn_svd, ttn_hmt, ttnn
from ttn_norms import frobenius_diff_norm, frobenius_norm


# --------------------------------------------------------------------------- #
#  Sketch budget and method builders                                          #
# --------------------------------------------------------------------------- #
# ----------------------------------------------------------------------
# Scale mode.  The QB methods and the qb_svd finish are n_above-free regardless
# of this switch, and TTN-SVD now uses the full-precision, n-independent
# canonical hierarchical SVD (above_mode="orth"), so it is also n_above-free and
# is the correct quasi-optimal reference at any scale -- no precision floor.
# What the switch still controls:
#   * TTN-HMT: dense Gaussian sketch ("gaussian", forms n_above) vs Khatri-Rao
#     ("kr", n_above-free).  (Its qb_svd finish does not use above_mode.)
# So set_scale_mode(large=True) is only needed to take the *gaussian* TTN-HMT to
# big n; every other curve scales either way.  The "qr" and "gram" TTN-SVD
# back-ends remain available by direct call for comparison (e.g. to exhibit the
# Gram sqrt(eps) floor), but the experiments use "orth".
# ----------------------------------------------------------------------
ABOVE_MODE = "orth"    # TTN-SVD: "orth" (full precision, n-free) | "qr" | "gram"
HMT_SKETCH = "gaussian"  # TTN-HMT sketch: "gaussian" (needs n_above) | "kr" (free)


def set_scale_mode(large: bool) -> None:
    """Switch the n_above-dependent baseline (gaussian TTN-HMT) to large-n mode.

    TTN-SVD uses the full-precision, n-independent canonical hierarchical SVD
    (above_mode="orth") in both modes, so it is the correct quasi-optimal
    reference at any n with no precision floor.  The QB methods and the qb_svd
    (SVD-of-Y) finish are n_above-free regardless of this switch.  Only TTN-HMT's
    sketch family changes: dense Gaussian for small n, Khatri-Rao for large n.
    """
    global ABOVE_MODE, HMT_SKETCH
    ABOVE_MODE = "orth"
    HMT_SKETCH = "kr" if large else "gaussian"


def _budget(r, which, d):
    """Return (P, R_per_copy) under the R = d / P = ceil(PR_target/d) scheme."""
    target = {"r": r, "1.2r": 1.2 * r, "2r": 2 * r}[which]
    R = int(d)
    P = max(1, int(math.ceil(target / float(d))))
    return P, R


def _qb(kind, finish, which):
    """Builder for a QBTC variant (treestack/ttstack sketch x qb/qb_svd finish)."""
    def build(ttn_in, r, seed, d):
        P, R = _budget(r, which, d)
        return qbtc_ttn(ttn_in, r, finish=finish, sketch_kind=kind,
                        P=P, R_per_copy=R, above_mode=ABOVE_MODE, rng_seed=seed)
    return build


# name -> (builder(ttn_in, r, seed, d) -> TTN, is_deterministic)
METHODS = {
    "TTN-SVD":            (lambda t, r, s, d: ttn_svd(t, r, above_mode=ABOVE_MODE), True),
    "TTNN(p=3)":          (lambda t, r, s, d: ttnn(t, r, p=3, rng_seed=s), False),
    "TTN-HMT(p=3)":       (lambda t, r, s, d: ttn_hmt(t, r, oversample=3,
                                                      sketch=HMT_SKETCH,
                                                      above_mode=ABOVE_MODE,
                                                      rng_seed=s), False),
    "QB-Tree (PR=r)":       (_qb("treestack", "qb",     "r"),    False),
    "QB-Train (PR=r)":      (_qb("ttstack",   "qb",     "r"),    False),
    "QB-Tree (PR=1.2r)":    (_qb("treestack", "qb",     "1.2r"), False),
    "QB-TreeSVD (PR=1.2r)": (_qb("treestack", "qb_svd", "1.2r"), False),
    "QB-TreeSVD (PR=r)":    (_qb("treestack", "qb_svd", "r"),    False),
}

# (title, method names, draw quasi-optimal band?)
PANELS = [
    ("(1)  Accuracy",
     ["TTN-SVD", "QB-TreeSVD (PR=1.2r)", "TTNN(p=3)", "TTN-HMT(p=3)"], False),
    ("(2)  Sketch topology  (plain QB, PR = r)",
     ["TTN-SVD", "QB-Tree (PR=r)", "QB-Train (PR=r)"], False),
    ("(3)  QB finish  (Tree, PR = 1.2r)",
     ["TTN-SVD", "QB-Tree (PR=1.2r)", "QB-TreeSVD (PR=1.2r)"], False),
    ("(4)  Oversampling  (QB-TreeSVD)",
     ["TTN-SVD", "QB-TreeSVD (PR=r)", "QB-TreeSVD (PR=1.2r)"], False),
]

# Okabe-Ito colourblind-safe palette; one distinct colour per series, all solid.
STYLE = {
    "TTN-SVD":             dict(color="#000000", marker="^", ls="-"),
    "TTNN(p=3)":           dict(color="#D55E00", marker="o", ls="-"),
    "TTN-HMT(p=3)":        dict(color="#CC79A7", marker="v", ls="-"),
    "QB-Tree (PR=r)":      dict(color="#0072B2", marker="d", ls="-"),
    "QB-Train (PR=r)":     dict(color="#E69F00", marker="s", ls="-"),
    "QB-Tree (PR=1.2r)":   dict(color="#0072B2", marker="d", ls="-"),
    "QB-TreeSVD (PR=1.2r)":dict(color="#009E73", marker="D", ls="-"),
    "QB-TreeSVD (PR=r)":   dict(color="#0072B2", marker="D", ls="-"),
}


# --------------------------------------------------------------------------- #
#  Runner                                                                     #
# --------------------------------------------------------------------------- #
def run_methods(ttn_in, ranks, n_leaves, trials=5, seed=0, methods=None, verbose=True):
    """Compress ``ttn_in`` with every method over ``ranks``.

    ``n_leaves`` (= d) sets the per-copy sketch bond R = d.  Returns
    ``{name: {"mean": np.ndarray, "std": np.ndarray}}`` of relative Frobenius
    errors; deterministic methods use one trial, randomized methods ``trials``.
    """
    methods = methods or METHODS
    norm_in = frobenius_norm(ttn_in)
    results = {name: {"mean": [], "std": []} for name in methods}

    for r in ranks:
        if verbose:
            print(f"  --- rank r = {r} ---")
        for name, (build, deterministic) in methods.items():
            n_tr = 1 if deterministic else trials
            errs = []
            for t in range(n_tr):
                s = seed * 100000 + r * 100 + t
                out = build(ttn_in, r, s, n_leaves)
                errs.append(frobenius_diff_norm(ttn_in, out) / norm_in)
            mu = float(np.mean(errs))
            sd = float(np.std(errs)) if n_tr > 1 else 0.0
            results[name]["mean"].append(mu)
            results[name]["std"].append(sd)
            if verbose:
                tag = "" if deterministic else f" +/- {sd:.1e}"
                print(f"    {name:22s} {mu:.4e}{tag}")

    for name in results:
        results[name]["mean"] = np.asarray(results[name]["mean"], float)
        results[name]["std"] = np.asarray(results[name]["std"], float)
    return results


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_four_panels(results, ranks, N_steps, suptitle, save_path):
    """Render the 2x2 panel figure described in the module docstring."""
    ranks = np.asarray(ranks, float)
    opt = results["TTN-SVD"]["mean"] if "TTN-SVD" in results else None
    FLOOR = 1e-16  # errors below this are machine noise; keep bands from underflowing

    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.20)

    for idx, (title, names, show_band) in enumerate(PANELS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        seen = []

        if show_band:
            ax.fill_between(
                ranks, np.maximum(opt, FLOOR), math.sqrt(N_steps) * np.maximum(opt, FLOOR),
                color="#9aa0a6", alpha=0.16, lw=0,
                label=r"quasi-optimal band $[\sigma_{\mathrm{opt}},\,\sqrt{N}\,\sigma_{\mathrm{opt}}]$",
            )

        for name in names:
            if name not in results:
                continue
            mu = results[name]["mean"]
            sd = results[name]["std"]
            st = STYLE[name]
            ax.plot(ranks, mu, marker=st["marker"], ls=st["ls"], color=st["color"],
                    lw=2.0, ms=6, mec="white", mew=0.6, label=name)
            if sd is not None and np.any(sd > 0):
                lower = np.maximum(mu - sd, np.maximum(0.1 * mu, FLOOR))
                ax.fill_between(ranks, lower, mu + sd, color=st["color"], alpha=0.13, lw=0)
            seen.append(mu[mu > 0])

        if seen:
            allv = np.concatenate(seen)
            lo, hi = max(FLOOR, allv.min() / 3.0), allv.max() * 3.0
            if hi > lo:
                ax.set_ylim(lo, hi)

        ax.set_yscale("log")
        ax.set_xlabel(r"target rank $r$")
        ax.set_xticks(ranks)
        if idx % 2 == 0:
            ax.set_ylabel(r"$\|X-\hat X\|_F\,/\,\|X\|_F$")
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.legend(fontsize=8.5, handlelength=2.8)

    fig.suptitle(suptitle, fontsize=12.5, y=0.97)
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")
    return results
