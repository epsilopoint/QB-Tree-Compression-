"""
Pathological example for Cholesky-Based Compression (CBC) on tensor trains.

Reproduces the two numerical experiments of the CBC quasi-optimality section
(Prop. "Failure without right-canonicality" + the conditioning-dependent
sharpness theorem).  A single explicit pathological tensor is compressed by
three rank-r algorithms and compared against the best rank-r error:

  * CBC      -- Cholesky-Based Compression (sweep-based local truncation),
                cbc_tt.cbc_tt.  Sensitive to the input gauge.
  * TT-SVD   -- deterministic hierarchical SVD, cbc_tt.tt_svd.  Quasi-optimal.
  * QBTC     -- QB Tree Compression specialised to the tensor-train graph,
                a stacked open-leaf TreeStack sketch (P=2, R_sk=8, PR=16=r,
                no oversampling), averaged over several sketch draws.

The tensor (d=6, r=16, n=5, t*=2, a=2) is the track-routed construction of the
"smaller physical dimensions" remark: a single bond index alpha in
{1,...,R_eff=17} is routed across the six modes by two base-n maps iota_L, iota_R,
with the failure localised at cut t*=2.  Its only nonzero entries are the 17
multi-indices (iota_L(alpha), iota_R(alpha)) carrying value a_alpha * b_alpha,
with weights

    a_alpha = a,  b_alpha = eps * q^(alpha-1)   for alpha = 1, ..., r,
    a_{R_eff} = b_{R_eff} = 1,

so r "fake-large" directions of local singular value a hide r small global
entries a*eps, while the single "fake-small" direction carries the dominant
global entry 1.  CBC keeps the fake-large directions and discards the dominant
one, giving error 1 while the best rank-r error is only a*eps -- a failure
ratio 1/(a*eps).  The cores are stored at bond dimension R=20 (R_eff zero-padded)
to model a reported bond dimension exceeding the active rank.

The spread parameter q controls the small spectrum: q=1 ties all r small
singular values at a*eps (Experiment 1); q=1.2 spreads them geometrically over
[a*eps, a*eps*q^(r-1)] (Experiment 2).  The tie at q=1 makes the Frobenius cost
invariant to which degenerate direction is dropped, so QBTC and TT-SVD coincide;
q>1 breaks the degeneracy and separates them.

Two figures are produced, Picture/CBC_q=1.png and Picture/CBC_q=12.png, each a
two-panel plot: (left) absolute Frobenius error vs eps with eps descending,
(right) the quasi-optimality ratio ||X - Xhat|| / ||X - X_best^(r)|| vs eps.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


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

from cbc_tt import cbc_tt, tt_svd, reconstruct_tt, right_canonicalize_tt
from ttn_format import TTN, qbtc_ttn
from tree_sketch import Node


# --------------------------------------------------------------------------- #
#  Construction                                                               #
# --------------------------------------------------------------------------- #
def build_caterpillar(d: int) -> Node:
    """Left-leaning caterpillar tree with d physical leaves.

    The internal edges are exactly the tensor-train cuts {1}, {1,2}, ...,
    {1,...,d-1}, so QBTC on this tree is QBTC specialised to the TT graph
    (N = d - 1 compression steps, matching the sqrt(d-1) quasi-optimality
    constant).
    """
    leaves = [Node(f"L{i}", physical_axis=i) for i in range(d)]
    root = Node("I0")
    cur = root
    for k in range(d - 1):
        if k < d - 2:
            nxt = Node(f"I{k+1}")
            cur.children = [leaves[k], nxt]
        else:
            cur.children = [leaves[k], leaves[k + 1]]
            nxt = None
        for c in cur.children:
            c.parent = cur
        cur = nxt
    return root


def _weights(r: int, R_eff: int, a: float, eps: float, q: float):
    """a_alpha, b_alpha for alpha = 1..R_eff (1-indexed via length R_eff+1)."""
    a_vec = np.zeros(R_eff + 1)
    b_vec = np.zeros(R_eff + 1)
    for al in range(1, r + 1):
        a_vec[al] = a
        b_vec[al] = eps * q ** (al - 1)
    a_vec[R_eff] = 1.0
    b_vec[R_eff] = 1.0
    return a_vec, b_vec


def build_pathological_tt(d: int, r: int, n: int, a: float, eps: float,
                          q: float, R: int, tstar: int = 2):
    """Track-routed pathological TT of the CBC failure remark, padded to bond R.

    Returns a list of d cores [C_1, ..., C_d], C_k of shape (R_{k-1}, n, R_k),
    with the failure localised at cut t* (default 2).  Bond dimension is R on
    every internal cut (R_eff = r+1 active, the rest zero-padded).
    """
    R_eff = r + 1
    assert 1 <= tstar <= d - 1
    assert n ** tstar >= R_eff and n ** (d - tstar) >= R_eff, \
        "feasibility: n^t* and n^(d-t*) must each be >= R_eff"
    a_vec, b_vec = _weights(r, R_eff, a, eps, q)

    def digits(x, ndig):                       # base-n, MSD first, +1 shift
        out = []
        for _ in range(ndig):
            out.append(x % n)
            x //= n
        return [v + 1 for v in out[::-1]]

    iL = {al: digits(al - 1, tstar)     for al in range(1, R_eff + 1)}
    iR = {al: digits(al - 1, d - tstar) for al in range(1, R_eff + 1)}

    cores = []
    # ---- left chain, modes 1..t* (weight a_alpha applied at C_{t*}) ----
    # C_1: (1, n, R)
    C = np.zeros((1, n, R))
    for al in range(1, R_eff + 1):
        C[0, iL[al][0] - 1, al - 1] = 1.0
    cores.append(C)
    # C_2..C_{t*-1}: route iL digits, identity on the bond
    for k in range(2, tstar):
        C = np.zeros((R, n, R))
        for al in range(1, R_eff + 1):
            C[al - 1, iL[al][k - 1] - 1, al - 1] = 1.0
        cores.append(C)
    # C_{t*}: route last left digit, apply a_alpha, expose the cut bond
    C = np.zeros((R, n, R))
    for al in range(1, R_eff + 1):
        C[al - 1, iL[al][tstar - 1] - 1, al - 1] = a_vec[al]
    cores.append(C)
    # ---- right chain, modes t*+1..d (weight b_alpha applied at C_{t*+1}) ----
    # C_{t*+1}: apply b_alpha, route first right digit
    C = np.zeros((R, n, R))
    for al in range(1, R_eff + 1):
        C[al - 1, iR[al][0] - 1, al - 1] = b_vec[al]
    cores.append(C)
    # middle right cores route the interior right digits
    for j in range(1, d - tstar - 1):
        C = np.zeros((R, n, R))
        for al in range(1, R_eff + 1):
            C[al - 1, iR[al][j] - 1, al - 1] = 1.0
        cores.append(C)
    # C_d: route last right digit, close the bond (R_d = 1)
    C = np.zeros((R, n, 1))
    for al in range(1, R_eff + 1):
        C[al - 1, iR[al][d - tstar - 1] - 1, 0] = 1.0
    cores.append(C)
    assert len(cores) == d
    return cores


def best_rank_r_error(r: int, R_eff: int, a: float, eps: float, q: float) -> float:
    """||X - X_best^(r)||_F.  X is a sum of R_eff orthogonal rank-1 terms with
    coefficients a_alpha b_alpha, so the best rank-r approximation (at every cut
    simultaneously) drops the smallest R_eff - r coefficients."""
    a_vec, b_vec = _weights(r, R_eff, a, eps, q)
    coeffs = np.sort(np.abs(a_vec[1:] * b_vec[1:]))[::-1]
    return float(np.sqrt(np.sum(coeffs[r:] ** 2)))


# --------------------------------------------------------------------------- #
#  One instance                                                               #
# --------------------------------------------------------------------------- #
def run_one(d, r, n, a, R, q, eps, n_trials, tstar=2, rng=None):
    """Errors of CBC, TT-SVD and QBTC on one (q, eps) instance."""
    if rng is None:
        rng = np.random.default_rng(0)
    R_eff = r + 1
    cores = build_pathological_tt(d, r, n, a, eps, q, R, tstar)
    X = reconstruct_tt(cores)
    best = best_rank_r_error(r, R_eff, a, eps, q)

    cbc_err = float(np.linalg.norm(X - reconstruct_tt(cbc_tt(cores, target_r=r))))
    ttsvd_err = float(np.linalg.norm(X - reconstruct_tt(tt_svd(cores, target_r=r))))

    # CBC after right-canonicalisation (recovers quasi-optimality; reported, not plotted)
    cbc_rc_err = float(np.linalg.norm(
        X - reconstruct_tt(cbc_tt(right_canonicalize_tt(cores), target_r=r))))

    # QBTC on the TT-graph caterpillar, averaged over sketch draws.
    ttn = TTN.from_dense(X, build_caterpillar(d))
    seeds = rng.integers(0, 2**31 - 1, size=n_trials)
    q_errs = [float(np.linalg.norm(
        X - qbtc_ttn(ttn, r, finish="qb", sketch_kind="treestack",
                     P=2, R_per_copy=8, rng_seed=int(s)).to_dense()))
        for s in seeds]

    return dict(best=best, cbc=cbc_err, ttsvd=ttsvd_err, cbc_rc=cbc_rc_err,
                qbtc_mean=float(np.mean(q_errs)), qbtc_std=float(np.std(q_errs)))


# --------------------------------------------------------------------------- #
#  One experiment (one value of q): sweep eps + two-panel figure              #
# --------------------------------------------------------------------------- #
def run_experiment(q, *, d=6, r=16, n=5, a=2.0, R=20, tstar=2,
                   eps_grid=None, n_trials=20, save_name=None, seed=0,
                   verbose=True):
    if eps_grid is None:
        eps_grid = np.logspace(-6, -2, 36)
    rng = np.random.default_rng(seed)
    R_eff = r + 1

    if verbose:
        print(f"\n===== q = {q}  (d={d}, r={r}, n={n}, t*={tstar}, a={a}, "
              f"R={R}, R_eff={R_eff}) =====")
        for eps in (1e-2, 1e-4, 1e-6):
            res = run_one(d, r, n, a, R, q, eps, n_trials=5, tstar=tstar, rng=rng)
            print(f"  eps={eps:.0e}: best={res['best']:.2e} | "
                  f"CBC={res['cbc']:.2e} (ratio {res['cbc']/res['best']:.1f}, "
                  f"pred {1.0/(a*eps):.1f}) | CBC+rc={res['cbc_rc']:.2e} "
                  f"(ratio {res['cbc_rc']/res['best']:.2f}) | "
                  f"TT-SVD={res['ttsvd']:.2e} ({res['ttsvd']/res['best']:.2f}) | "
                  f"QBTC={res['qbtc_mean']:.2e} ({res['qbtc_mean']/res['best']:.2f})")

    best, cbc, ttsvd = [], [], []
    qmean, qstd = [], []
    for eps in eps_grid:
        res = run_one(d, r, n, a, R, q, eps, n_trials=n_trials, tstar=tstar, rng=rng)
        best.append(res["best"]); cbc.append(res["cbc"]); ttsvd.append(res["ttsvd"])
        qmean.append(res["qbtc_mean"]); qstd.append(res["qbtc_std"])
    best = np.array(best); cbc = np.array(cbc); ttsvd = np.array(ttsvd)
    qmean = np.array(qmean); qstd = np.array(qstd)
    predicted = 1.0 / (a * eps_grid)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    c_cbc, c_svd, c_qb = "crimson", "black", "#1f5fd0"

    # ---- left: absolute Frobenius error ----
    axL.loglog(eps_grid, best, "k--", lw=1.3, alpha=0.55,
               label=r"best rank-$r$  $(=a\varepsilon)$")
    axL.loglog(eps_grid, cbc, "o-", color=c_cbc, ms=4, label="CBC")
    axL.loglog(eps_grid, ttsvd, "^-", color=c_svd, ms=4, label="TT-SVD")
    axL.fill_between(eps_grid, np.maximum(qmean - qstd, 1e-300), qmean + qstd,
                     color=c_qb, alpha=0.20)
    axL.loglog(eps_grid, qmean, "s-", color=c_qb, ms=4, label="QBTC (mean)")
    axL.set_xlabel(r"$\varepsilon$")
    axL.set_ylabel(r"$\|\mathcal{X}-\widehat{\mathcal{X}}\|_F$")
    axL.set_title("Absolute Frobenius error")
    axL.set_xlim(eps_grid.max(), eps_grid.min())          # eps descending
    axL.grid(True, which="both", ls="--", alpha=0.3)
    axL.legend(loc="best", fontsize=9)

    # ---- right: quasi-optimality ratio ----
    axR.loglog(eps_grid, predicted, "k--", lw=1.3, alpha=0.55,
               label=r"predicted $1/(a\varepsilon)$")
    axR.loglog(eps_grid, cbc / best, "o-", color=c_cbc, ms=4, label="CBC")
    axR.loglog(eps_grid, ttsvd / best, "^-", color=c_svd, ms=4, label="TT-SVD")
    axR.fill_between(eps_grid, np.maximum((qmean - qstd) / best, 1e-300),
                     (qmean + qstd) / best, color=c_qb, alpha=0.20)
    axR.loglog(eps_grid, qmean / best, "s-", color=c_qb, ms=4, label="QBTC (mean)")
    axR.axhline(np.sqrt(d - 1), color="grey", ls=":", alpha=0.8,
                label=rf"$\sqrt{{d-1}}={np.sqrt(d-1):.2f}$")
    axR.set_xlabel(r"$\varepsilon$")
    axR.set_ylabel(r"$\|\mathcal{X}-\widehat{\mathcal{X}}\|_F\,/\,"
                   r"\|\mathcal{X}-\mathcal{X}^{(r)}_{\mathrm{best}}\|_F$")
    axR.set_title("Quasi-optimality ratio")
    axR.set_xlim(eps_grid.max(), eps_grid.min())          # eps descending
    axR.grid(True, which="both", ls="--", alpha=0.3)
    axR.legend(loc="best", fontsize=9)

    qlabel = "tied" if abs(q - 1.0) < 1e-12 else "spread"
    fig.suptitle(rf"Prop. 3.2 pathological TT ({qlabel} small singular values, "
                 rf"$q={q}$): $d={d}$, $r={r}$, $n={n}$, $a={a}$", y=1.02)
    fig.tight_layout()

    outdir = "Picture"
    os.makedirs(outdir, exist_ok=True)
    if save_name is None:
        save_name = f"CBC_q={str(q).replace('.', '')}"
    path = os.path.join(outdir, f"{save_name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  saved figure: {path}")
    return path


def main():
    # Experiment 1: tied small singular values (q = 1).
    run_experiment(q=1.0, save_name="CBC_q=1")
    # Experiment 2: spread small singular values (q = 1.2).
    run_experiment(q=1.2, save_name="CBC_q=12")


if __name__ == "__main__":
    main()
