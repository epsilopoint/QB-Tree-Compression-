"""
TTN-native baselines: TTN-SVD, TTN-HMT(p), TTNN(p).
All three take a TTN object as input and return a TTN object as output.

All three accept a TTN object as input (instead of a dense np.ndarray) and
return a TTN object as output. The interface mirrors the dense baselines
in qbtc.py (ttn_svd, ttn_hmt, ttnn) one-for-one — same arguments (other
than the input being a TTN) and the same target_r / oversample / p
semantics.

Internals:
  * ttn_svd   full-precision canonical hierarchical SVD (above_mode="orth"):
    root-orthogonalise the cores (leaves->root QR), then truncate root->leaves
    moving the orthogonality centre, so every truncation is an SVD of a small
    core.  No Gram, no n_above -- backward stable and n-independent.  The
    "qr" and "gram" back-ends are kept for comparison.

  * ttn_hmt   is a thin wrapper around qbtc_ttn(sketch_kind="gaussian",
    P=1, R_per_copy=r+oversample). The Gaussian sketch is the dense Omega
    of HMT; the "qb" finish (oversample=0) and "qb_svd" finish (oversample>0)
    correspond exactly to the dense ttn_hmt's two branches.

  * ttnn      the Bucci-Verzella TTNN algorithm, fully structured: BOTH the
    column sketch (above-leaves) and the row sketch (subtree-leaves) are
    Khatri-Rao products of per-leaf Gaussians, contracted structurally so the
    physical subtree dimension n_subtree(v)=n^(#leaves under v) is never formed.
    Peak work/memory is bond-bounded, independent of n.  The earlier dense-row
    variant (Y_v dense over n_subtree, hence ~n^(#leaves) cost) is kept as
    `ttnn_dense` for reference / cross-checking.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
from typing import Dict, Optional, List
import sys


from tree_sketch import Node, all_nodes, compression_order
from ttn_format import TTN, qbtc_ttn, contract_network


# ------------------------------------------------------------------
# Helper: stable pseudo-inverse (matching qbtc.py's _stable_pinv).
# ------------------------------------------------------------------
def _stable_pinv(R: np.ndarray, eps: float = 10.0) -> np.ndarray:
    U, S, Vh = scipy.linalg.svd(R, full_matrices=False)
    if S.size == 0:
        return np.zeros((R.shape[1], R.shape[0]))
    thresh = eps * np.finfo(S.dtype).eps * S[0]
    S_inv = np.where(S > thresh, 1.0 / S, 0.0)
    return Vh.T * S_inv @ U.T


# ------------------------------------------------------------------
# 1.  TTN-SVD (deterministic, quasi-optimal reference).
# ------------------------------------------------------------------
def _hsvd_orth(ttn_in: TTN, target_r: int) -> TTN:
    """Full-precision, n-independent hierarchical SVD via canonical form.

    Two phases on the cores (no Gram, no n^(d-1) materialisation):

      Phase 1 (leaves -> root): QR each non-root core so that its
        (children+physical) x parent-bond matricisation has orthonormal
        columns.  This brings the network to root-orthogonal canonical form:
        every subtree is an isometry from its parent-bond and the weight sits
        at the root.

      Phase 2 (root -> leaves): a depth-first sweep that keeps the orthogonality
        centre at the current node.  Each edge is truncated exactly once, by an
        SVD of a *small* reshaped core while the rest of the network is
        orthonormal, so the per-edge error equals the discarded singular energy
        and the errors add in quadrature (the orthogonal-error-decomposition /
        quasi-optimality structure).  After truncating an edge the centre is
        pushed into the child; on return it is QR-ed back to the parent so the
        next sibling edge can be cut.

    Because every factorisation is a QR or SVD of a bond-sized matrix (plus a
    single physical mode at a leaf), this is backward stable -- it resolves
    singular directions down to eps*sigma_max, not the sqrt(eps) floor of the
    Gram route -- and its cost is independent of the physical dimension n.
    """
    root = ttn_in.root
    cores: Dict[Node, np.ndarray] = {v: np.array(c, copy=True)
                                     for v, c in ttn_in.cores.items()}

    def caxis(u: Node, j: int) -> int:
        # axis of cores[u] holding the bond to child slot j (axis 0 is the
        # parent-bond for non-root nodes; the root has no parent-bond axis).
        return j if (u is root) else j + 1

    # Phase 1 -- root-orthogonalisation (children before parents).
    for v in compression_order(root):
        C = cores[v]
        p = C.shape[0]
        rest_shape = C.shape[1:]
        rest = int(np.prod(rest_shape)) if rest_shape else 1
        Q, Rm = np.linalg.qr(C.reshape(p, rest).T)          # Q:(rest,k)  Rm:(k,p)
        cores[v] = Q.T.reshape((Q.shape[1],) + rest_shape)
        par = v.parent
        ax = caxis(par, par.children.index(v))
        cores[par] = np.moveaxis(
            np.tensordot(cores[par], Rm, axes=([ax], [1])), -1, ax)

    # Phase 2 -- truncate root -> leaves, moving the orthogonality centre.
    def _truncate(v: Node) -> None:
        for j, c in enumerate(v.children):
            ax = caxis(v, j)
            Cv = cores[v]
            cb = Cv.shape[ax]
            Cv_m = np.moveaxis(Cv, ax, -1)
            oshape = Cv_m.shape[:-1]
            rest = int(np.prod(oshape)) if oshape else 1
            U, S, Wt = np.linalg.svd(Cv_m.reshape(rest, cb), full_matrices=False)
            re = min(target_r, U.shape[1])
            cores[v] = np.moveaxis(U[:, :re].reshape(oshape + (re,)), -1, ax)
            cores[c] = np.tensordot(S[:re, None] * Wt[:re, :], cores[c],
                                    axes=([1], [0]))         # push centre into c
            _truncate(c)
            # move the centre back c -> v (exact QR, no further truncation)
            Cc = cores[c]
            pc = Cc.shape[0]
            rcs = Cc.shape[1:]
            rc = int(np.prod(rcs)) if rcs else 1
            Q2, R2 = np.linalg.qr(Cc.reshape(pc, rc).T)
            cores[c] = Q2.T.reshape((Q2.shape[1],) + rcs)
            cores[v] = np.moveaxis(
                np.tensordot(cores[v], R2, axes=([ax], [1])), -1, ax)

    _truncate(root)
    return TTN(root, cores)


def ttn_svd(ttn_in: TTN, target_r: int, above_mode: str = "orth") -> TTN:
    """Exact hierarchical TTN-SVD on a TTN-format input. The deterministic,
    quasi-optimal reference.

    Three back-ends, selected by ``above_mode``:

      "orth" (default) : full-precision, n-independent hierarchical SVD in
                         canonical form (_hsvd_orth).  Only QR / SVD of
                         bond-sized matrices -- backward stable (resolves
                         singular values down to eps*sigma_max) AND never forms
                         n^(d-1).  Use this everywhere.

    The other two are kept for comparison and operate by a leaves-to-root sweep
    that, at each node v, takes the top-r left singular vectors of the unfolding
    M_v from a small factor B = subtree_V.T @ R_a.T (whose singular values equal
    sigma(M_v)), with R_a.T @ R_a = G = above_V @ above_V.T:

      "qr"   : R_a = qr(above_V.T).  Backward stable but MATERIALISES above_V
               (R_v x n^(d-1)) -- feasible only for small n.
      "gram" : R_a from a symmetric square root of the structured above-Gram
               (compute_above_gram, O(R^2), never forms n_above).  Forming G
               squares the spectrum, so it cannot resolve singular values below
               ~sqrt(eps) ~ 1e-8.  Kept only to exhibit that floor.
    """
    if above_mode == "orth":
        return _hsvd_orth(ttn_in, target_r)

    # Mutate `residual`, build `out_cores` separately (mirror qbtc_ttn).
    residual = TTN(ttn_in.root, dict(ttn_in.cores))
    out_cores: Dict[Node, np.ndarray] = {}

    for v in compression_order(residual.root):
        if v is residual.root:
            continue
        subtree_V, _ = residual._materialize_subtree(v)      # (R_v, n_below) cheap
        if above_mode == "gram":
            G = residual.compute_above_gram(v)               # (R_v, R_v), no n_above
            G = 0.5 * (G + G.T)
            w, Vg = np.linalg.eigh(G)                         # eigenvalues = sigma^2
            w = np.clip(w, 0.0, None)
            R_a = np.sqrt(w)[:, None] * Vg.T                  # R_a.T @ R_a = G
        else:  # "qr"
            above_V, _ = residual._materialize_above(v)       # (R_v, n_above)
            _, R_a = scipy.linalg.qr(above_V.T, mode="economic")  # R_a: (R_v, R_v)
        B = subtree_V.T @ R_a.T                               # (n_below, R_v)
        U, _, _ = scipy.linalg.svd(B, full_matrices=False)
        r_eff = min(target_r, U.shape[1])
        Q_v = U[:, :r_eff]                                   # (n_below, r_eff)

        # Reshape Q_v into the OUTPUT TTN's multi-axis convention.
        if v.is_leaf:
            out_cores[v] = Q_v.T                              # (r_eff, n_v)
        else:
            child_dims = [out_cores[c].shape[0] for c in v.children]
            Q_v_md = Q_v.reshape(*child_dims, r_eff)
            Q_v_md = np.moveaxis(Q_v_md, -1, 0)               # (r_eff, r_c1, ..., r_cm)
            out_cores[v] = Q_v_md

        residual.residual_update(v, Q_v)

    out_cores[residual.root] = residual.cores[residual.root]
    return TTN(residual.root, out_cores)


# ------------------------------------------------------------------
# 2.  TTN-HMT (per-node randomised SVD, Gaussian sketch) — thin wrapper.
# ------------------------------------------------------------------
def ttn_hmt(ttn_in: TTN, target_r: int, oversample: int = 0,
                   sketch: str = "gaussian", above_mode: str = "qr",
                   rng_seed: Optional[int] = None) -> TTN:
    """TTN-HMT on TTN-format input.

    Internals:  qbtc_ttn(sketch_kind=sketch, P=1, R_per_copy=r+oversample)
                with finish="qb" if oversample==0 else "qb_svd".

    sketch="gaussian" : dense per-node Gaussian range finder (materialises
                        n_above; the literal matrix HMT).
    sketch="kr"       : Khatri-Rao structured sketch (no n_above) -- this is
                        the memory-efficient variant Bucci uses at scale.
    above_mode        : forwarded to qbtc_ttn but now IGNORED by the qb_svd
                        finish (which is the n_above-free SVD-of-Y).  So with
                        oversample>0, TTN-HMT is n_above-free as long as
                        sketch="kr"; above_mode no longer affects it.
    For a fully n_above-free TTN-HMT use sketch="kr".
    """
    finish = "qb" if oversample == 0 else "qb_svd"
    return qbtc_ttn(
        ttn_in,
        target_r=target_r,
        finish=finish,
        P=1,
        R_per_copy=target_r + oversample,
        sketch_kind=sketch,
        above_mode=above_mode,
        rng_seed=rng_seed,
    )


# ------------------------------------------------------------------
# 3.  TTNN (Bucci-Verzella Algorithm 4.1) — Path-A version.
# ------------------------------------------------------------------
def _subtree_leaves(node: Node) -> List[Node]:
    """Return all leaves in node's subtree, in DFS left-to-right order."""
    if node.is_leaf:
        return [node]
    out = []
    for c in node.children:
        out.extend(_subtree_leaves(c))
    return out


def ttnn_dense(ttn_in: TTN, target_r: int, p: int = 0,
                rng_seed: Optional[int] = None,
                col_factors: Optional[Dict] = None,
                row_sketch: Optional[Dict] = None) -> TTN:
    """[LEGACY / reference] TTNN with a DENSE row-side Gaussian.

    Superseded by `ttnn` (the fully structured version).  This routine draws a
    dense Gaussian Y_v of shape (n_subtree(v), r+p) per node, so its cost and
    memory scale as n^(#leaves under v) -- e.g. n^3 on the Figure-1 tree -- and
    it becomes very slow / memory-heavy at large n.  Kept only as a reference
    and to cross-check `ttnn` (pass `col_factors`/`row_sketch` to inject the
    same Khatri-Rao sketches and obtain an exact match).

    col_factors : optional {v: {leaf: K_leaf (n_leaf, r)}} to use instead of
                  freshly drawn column Gaussians (the above-leaf sketch X_v).
    row_sketch   : optional {v: Y_v (n_subtree(v), r+p)} to use instead of the
                  freshly drawn dense row Gaussian.

    Step 1 (per-node sketches):
        For each non-root v:
            X_v = Khatri-Rao of per-leaf-Gaussians (above-leaves of v).
                  Path-A computes TX_v = M_v @ X_v structured-ly via
                  apply_unfolding_KR — no n_above materialisation.
            Y_v = dense Gaussian (n_subtree(v), r+p).
            Omega_v = Y_v.T @ TX_v  -> (r+p, r)
            (Z_v, R_v) = QR(Omega_v); M_v_out = Z_v.T @ Y_v.T.

    Step 2 (cores assembly):  identical to dense ttnn — only uses TX_v and
        M_v_out matrices already computed in Step 1.

    Root step:  contract the input TTN with each child's M_v_out projector.
        Done structurally by folding M_v_out into the subtree at each
        child of root, then contracting against cores[root].

    Output: a TTN with bond dimension target_r on every edge.

    Stability note.  TTNN is a two-sided / generalized Nyström scheme: the
    per-node factor is (M X)(Yᵀ M X)⁺ Yᵀ with X of width r and Y of width r+p.
    With FIXED small oversampling (e.g. p=3), the ratio (r+p)/r -> 1 as r grows,
    so the inner matrix Yᵀ M X becomes square and can be ill-conditioned; the
    pseudo-inverse then amplifies, and on slowly-decaying spectra the per-node
    error can EXCEED 1 and grow with r (the projectors are oblique and compound
    across the tree).  This is intrinsic to generalized Nyström, not a defect of
    this port — it matches the dense reference and behaves cleanly on fast-decay
    tensors.  Use proportional oversampling (p ∝ r) to curb the blow-up; even
    then two-sided Nyström is weaker than the one-sided range finders (TTN-HMT)
    and QBTC's sketch+SVD on hard tensors.
    """
    rng = np.random.default_rng(rng_seed)
    root = ttn_in.root
    nodes = all_nodes(root)
    non_root = [n for n in nodes if n is not root]

    # Precompute n_subtree(v) = product of physical dims of leaves under v.
    # In the TTN convention used here, leaf core has shape (R_parent, n_v),
    # so n_v = ttn_in.cores[leaf].shape[1].
    leaf_phys = {l: ttn_in.cores[l].shape[1] for l in nodes if l.is_leaf}
    n_subtree = {l: leaf_phys[l] for l in nodes if l.is_leaf}
    for v in non_root + [root]:
        if not v.is_leaf and v not in n_subtree:
            n_subtree[v] = int(np.prod([leaf_phys[l] for l in _subtree_leaves(v)]))
    n_subtree[root] = int(np.prod([leaf_phys[l] for l in _subtree_leaves(root)]))

    # Step 1: per-node sketches.
    R_mat: Dict[Node, np.ndarray] = {}
    M_proj: Dict[Node, np.ndarray] = {}     # (r, n_subtree(v))
    TX: Dict[Node, np.ndarray] = {}          # (n_subtree(v), r)

    for v in non_root:
        # Above-leaves of v in DFS order — see apply_unfolding_KR.
        # We need K_above[l] of shape (n_l_or_r_l, r) for each above-leaf l.
        # Compute via _materialize_above's internal helper. Easiest: enumerate
        # above-leaves through the existing _materialize_above col_axes order.
        _, col_axes = ttn_in._materialize_above(v, order_only=True)
        # Map col_axes labels back to Nodes (same logic as build_ttstack_omega).
        label_to_node: Dict = {}
        for u in nodes:
            if u.is_leaf:
                label_to_node[u.physical_axis] = u
        # No processed-internals here (we never call residual_update before
        # this loop): all above-leaves are true leaves of the input TTN.
        above_leaves = [label_to_node[L] for L in col_axes]

        # K_above: per-leaf Gaussian (n_l, r) -- or injected via col_factors.
        if col_factors is not None:
            K_above = col_factors[v]
        else:
            K_above = {l: rng.standard_normal((leaf_phys[l], target_r))
                       for l in above_leaves}

        # TX_v = M_v @ X_v in structured form. Shape (n_subtree(v), r).
        TXv, _ = ttn_in.apply_unfolding_KR(v, K_above)

        # Row-side Gaussian Y_v — dense (n_subtree(v), r+p) -- or injected.
        n_D = TXv.shape[0]
        if row_sketch is not None:
            Yv = row_sketch[v]
        else:
            Yv = rng.standard_normal((n_D, target_r + p))

        Omega_v = Yv.T @ TXv                                    # (r+p, r)
        Zv, Rv = scipy.linalg.qr(Omega_v, mode="economic")     # Zv (r+p, r), Rv (r, r)
        Mv_out = Zv.T @ Yv.T                                    # (r, n_D)

        R_mat[v] = Rv
        M_proj[v] = Mv_out
        TX[v] = TXv

    # Step 2: assemble Q-cores for non-root nodes.
    cores_out: Dict[Node, np.ndarray] = {}

    for v in non_root:
        Rv = R_mat[v]
        TXv = TX[v]                                             # (n_subtree(v), r)
        if v.is_leaf:
            B = TXv @ _stable_pinv(Rv)                          # (n_v, r)
            # Leaf TTN core convention is (R_parent, n) — store as B.T.
            cores_out[v] = B.T                                  # (r, n_v)
        else:
            children = list(v.children)
            child_subtree_dims = [n_subtree[c] for c in children]
            TX_resh = TXv.reshape(*child_subtree_dims, target_r)
            from string import ascii_letters
            avail = list(ascii_letters)
            ax_lbl = avail.pop(0)
            child_in_lbls = [avail.pop(0) for _ in children]
            child_out_lbls = [avail.pop(0) for _ in children]
            TX_term = "".join(child_in_lbls) + ax_lbl
            M_terms = ["".join([child_out_lbls[i], child_in_lbls[i]])
                       for i in range(len(children))]
            out_term = "".join(child_out_lbls) + ax_lbl
            spec = TX_term + "," + ",".join(M_terms) + "->" + out_term
            B = contract_network(spec, TX_resh, *[M_proj[c] for c in children])
            # B shape: (r_c1, r_c2, ..., r_cm, r_v).
            m = len(children)
            n_row = int(np.prod(B.shape[:m]))
            B_mat = B.reshape(n_row, target_r) @ _stable_pinv(Rv)
            # B_mat: (n_row, r_v) with n_row = r^m (since every r_ci == target_r).
            # Internal-TTN-core convention: (R_parent, R_c1, ..., R_cm).
            B_resh = B_mat.reshape(*([target_r] * m), target_r)
            #   axes 0..m-1 = R_ci (children-bonds),  axis m = R_parent (= r_v).
            cores_out[v] = np.moveaxis(B_resh, m, 0)
            #   now axis 0 = R_parent, axes 1..m = R_ci.

    # Root core: contract the input TTN with each child's M_proj projector to
    # get the root's small core of shape (r_c1, ..., r_cm).
    children_of_root = list(root.children)

    # For each child c of root, fold M_proj[c] into the subtree at c, returning
    # a (r, R_c_to_root) matrix.  R_c_to_root = ttn_in.cores[c].shape[0] for
    # any non-root c (the parent-bond).
    proj_at_root_child: Dict[Node, np.ndarray] = {}

    def _fold_subtree_with_proj(node: Node,
                                 proj_flat: np.ndarray) -> np.ndarray:
        """Fold a (r, n_subtree(node)) projector into the TTN subtree at `node`,
        returning shape (r, R_node_to_parent)."""
        # proj_flat shape: (r, n_subtree(node)).
        if node.is_leaf:
            # cores[node] shape (R_node, n_node).  Result: proj_flat @ cores[node].T -> (r, R_node).
            return proj_flat @ ttn_in.cores[node].T              # (r, R_node)
        # Internal: reshape proj into per-child subtree dims.
        child_dims = [n_subtree[c] for c in node.children]
        proj_resh = proj_flat.reshape(proj_flat.shape[0], *child_dims)
        # Recursively fold each child's portion.
        # proj_resh has shape (r, d_c1, d_c2, ..., d_cm).
        # For each child c_i, we fold proj_resh[:, d_c1, ..., d_ci, ...] into
        # the subtree at c_i.  This requires reshape/transpose to handle one
        # child at a time.  We'll iteratively reduce dims by contracting.
        cur = proj_resh                                          # (r, d_c1, ..., d_cm)
        # cores[node] shape (R_node, R_c1, ..., R_cm).
        # We'll build up: for each child c_i, fold subtree at c_i giving a
        # (r, R_ci, ..., remaining children dims) tensor, until all child
        # dims are reduced to R_ci.  Then contract with cores[node].
        remaining_children = list(node.children)
        reduced_axes_count = 0
        for i, c in enumerate(node.children):
            # cur shape: (r, R_c1, R_c2, ..., R_ci-1, d_ci, d_ci+1, ..., d_cm)
            # We want to fold subtree at c (dim d_ci, currently at axis 1 + i).
            ax = 1 + i
            # Move ax to position 1 for folding.
            cur_for_c = np.moveaxis(cur, ax, 1)                  # (r, d_ci, ..., others)
            # Flatten "others" so cur_for_c becomes (r, d_ci, M).
            others_shape = cur_for_c.shape[2:]
            M_flat = int(np.prod(others_shape)) if others_shape else 1
            cur_for_c = cur_for_c.reshape(cur_for_c.shape[0], cur_for_c.shape[1], M_flat)
            # Fold: for each m in 0..M_flat-1, fold (r, d_ci) using subtree at c.
            R_c = ttn_in.cores[c].shape[0]
            folded = np.empty((cur_for_c.shape[0], R_c, M_flat))
            for m in range(M_flat):
                folded[:, :, m] = _fold_subtree_with_proj(c, cur_for_c[:, :, m])
            # folded: (r, R_c, M_flat).  Reshape back: (r, R_c, *others_shape).
            folded = folded.reshape((folded.shape[0], R_c) + others_shape)
            # Move R_c back to position ax.
            cur = np.moveaxis(folded, 1, ax)                     # (r, R_c1, ..., d_ci+1, ..., d_cm)
            # Replace d_ci by R_c at position ax (already done).
        # Now cur shape: (r, R_c1, R_c2, ..., R_cm).
        # Contract with cores[node] of shape (R_node, R_c1, ..., R_cm).
        node_core = ttn_in.cores[node]                           # (R_node, R_c1, ..., R_cm)
        # einsum: 'r a b ... , X a b ... -> r X'.
        m = len(node.children)
        from string import ascii_letters
        avail = list(ascii_letters)
        r_lbl = avail.pop(0)
        X_lbl = avail.pop(0)
        ch_lbls = [avail.pop(0) for _ in range(m)]
        cur_term = r_lbl + "".join(ch_lbls)
        core_term = X_lbl + "".join(ch_lbls)
        out_term = r_lbl + X_lbl
        return np.einsum(f"{cur_term},{core_term}->{out_term}", cur, node_core)

    for c in children_of_root:
        proj_at_root_child[c] = _fold_subtree_with_proj(c, M_proj[c])  # (r, R_c)

    # Root core in output TTN: contract cores[root] with each proj_at_root_child[c].
    # cores[root] shape: (R_c1, R_c2, ..., R_cm).
    # Result B_rho shape: (r_c1, r_c2, ..., r_cm).
    cores_in_root = ttn_in.cores[root]                            # (R_c1, ..., R_cm)
    m = len(children_of_root)
    from string import ascii_letters
    avail = list(ascii_letters)
    rc_lbls = [avail.pop(0) for _ in range(m)]
    Rc_lbls = [avail.pop(0) for _ in range(m)]
    root_term = "".join(Rc_lbls)
    proj_terms = [rc_lbls[i] + Rc_lbls[i] for i in range(m)]
    out_term = "".join(rc_lbls)
    spec = root_term + "," + ",".join(proj_terms) + "->" + out_term
    B_rho = contract_network(spec, cores_in_root,
                             *[proj_at_root_child[c] for c in children_of_root])
    cores_out[root] = B_rho

    return TTN(root, cores_out)


# ==================================================================
# Structured TTNN: Khatri-Rao row sketch (bond-bounded, no n_subtree).
# ==================================================================
def _subtree_sketch_KR(ttn_in: TTN, v: Node, H: Dict[Node, np.ndarray]) -> np.ndarray:
    """S_v = subtree_V(v) @ Y_v, where Y_v is the column-wise Khatri-Rao of the
    per-subtree-leaf Gaussians {H[l]} (each (n_l, w)).  Returns (R_v, w), built
    by a bottom-up contraction that never forms the n_subtree dimension."""
    cores = ttn_in.cores
    from string import ascii_letters

    def rec(u: Node) -> np.ndarray:
        if u.is_leaf:
            return cores[u] @ H[u]                       # (R_u, w)
        child_mats = [rec(c) for c in u.children]        # each (R_c, w), shared w
        m = len(u.children)
        av = list(ascii_letters)
        Ru = av.pop(0); w = av.pop(0)
        bl = [av.pop(0) for _ in range(m)]
        core_term = Ru + "".join(bl)                     # cores[u]: (R_u, b1..bm)
        child_terms = [bl[i] + w for i in range(m)]      # child_i: (b_i, w)
        spec = core_term + "," + ",".join(child_terms) + "->" + Ru + w
        return contract_network(spec, cores[u], *child_mats)

    return rec(v)


def _above_sketch_KR(ttn_in: TTN, v: Node, K: Dict[Node, np.ndarray]) -> np.ndarray:
    """A_v = above_V(v) @ X_v, where X_v is the column-wise Khatri-Rao of the
    per-above-leaf Gaussians {K[l]} (each (n_l, r)).  Returns (R_v, r), built
    column-by-column via width-1 overrides of _materialize_above (no n_above).
    This is exactly the (R_v, r) intermediate that apply_unfolding_KR forms
    before its n_subtree-tall subtree multiply."""
    cores = ttn_in.cores
    above = ttn_in.above_leaves(v)
    r = K[above[0]].shape[1]
    R_v = ttn_in.parent_bond_dim(v)
    A = np.zeros((R_v, r), dtype=np.float64)
    for kcol in range(r):
        override = {u: cores[u] @ K[u][:, kcol:kcol + 1] for u in above}  # (R_parent, 1)
        T, _ = ttn_in._materialize_above(v, leaf_override=override)        # (R_v, 1)
        A[:, kcol] = T.reshape(-1)
    return A


def _ttnn_sketch_factors(ttn_in: TTN, target_r: int, p: int,
                          rng: np.random.Generator) -> Dict:
    """Per non-root node v: column factors K^v (above-leaves, width r) and row
    factors H^v (subtree-leaves, width r+p), drawn in a fixed node/leaf order."""
    root = ttn_in.root
    leaf_phys = {l: ttn_in.cores[l].shape[1] for l in all_nodes(root) if l.is_leaf}
    factors: Dict = {}
    for v in all_nodes(root):
        if v is root:
            continue
        K = {l: rng.standard_normal((leaf_phys[l], target_r))
             for l in ttn_in.above_leaves(v)}
        H = {l: rng.standard_normal((leaf_phys[l], target_r + p))
             for l in _subtree_leaves(v)}
        factors[v] = (K, H)
    return factors


def _ttnn_from_factors(ttn_in: TTN, target_r: int, p: int, factors: Dict) -> TTN:
    """Assemble the TTNN output from precomputed Khatri-Rao factors, fully
    structured (no n_subtree(v) object is ever formed for an internal node)."""
    from string import ascii_letters
    root = ttn_in.root
    cores = ttn_in.cores
    non_root = [n for n in all_nodes(root) if n is not root]

    # Per-node sketch: A_v (R_v,r), S_v (R_v,r+p) -> Omega_v, QR; cache R_v_mat,
    # A_v, and the child projector P_v = S_v @ Z_v (R_v, r).
    Rmat: Dict[Node, np.ndarray] = {}
    Amat: Dict[Node, np.ndarray] = {}
    Pmat: Dict[Node, np.ndarray] = {}
    for v in non_root:
        K, H = factors[v]
        A_v = _above_sketch_KR(ttn_in, v, K)               # (R_v, r)
        S_v = _subtree_sketch_KR(ttn_in, v, H)             # (R_v, r+p)
        Omega_v = S_v.T @ A_v                              # (r+p, r)
        Z_v, R_v_mat = scipy.linalg.qr(Omega_v, mode="economic")   # (r+p,r),(r,r)
        Rmat[v] = R_v_mat
        Amat[v] = A_v
        Pmat[v] = S_v @ Z_v                                # (R_v, r)

    cores_out: Dict[Node, np.ndarray] = {}
    for v in non_root:
        R_v_mat = Rmat[v]; A_v = Amat[v]
        if v.is_leaf:
            TX_v = cores[v].T @ A_v                        # (n_v, r)   [n_v small]
            cores_out[v] = (TX_v @ _stable_pinv(R_v_mat)).T            # (r, n_v)
        else:
            m = len(v.children)
            av = list(ascii_letters)
            Rv = av.pop(0); rv = av.pop(0)
            bl = [av.pop(0) for _ in range(m)]
            rcl = [av.pop(0) for _ in range(m)]
            # core_pre[rc..,rv] = sum_{R_v,b..} cores[v][R_v,b..] prod_c P_c[b_c,rc_c] A_v[R_v,rv]
            spec = (Rv + "".join(bl) + ","
                    + ",".join(bl[i] + rcl[i] for i in range(m)) + ","
                    + Rv + rv + "->" + "".join(rcl) + rv)
            core_pre = contract_network(spec, cores[v],
                                        *[Pmat[c] for c in v.children], A_v)
            core_pre = core_pre.reshape(target_r ** m, target_r) @ _stable_pinv(R_v_mat)
            B_resh = core_pre.reshape(*([target_r] * m), target_r)   # 0..m-1=R_ci, m=R_parent
            cores_out[v] = np.moveaxis(B_resh, m, 0)                 # (R_parent, R_c1..R_cm)

    # Root core: contract cores[root] with each child's P_c^T (= the projected
    # subtree at that child).  cores[root] = (R_c1,...,R_cm).
    children = list(root.children)
    m = len(children)
    av = list(ascii_letters)
    Rcl = [av.pop(0) for _ in range(m)]
    rcl = [av.pop(0) for _ in range(m)]
    spec = ("".join(Rcl) + ","
            + ",".join(rcl[i] + Rcl[i] for i in range(m)) + "->" + "".join(rcl))
    cores_out[root] = contract_network(spec, cores[root],
                                       *[Pmat[c].T for c in children])
    return TTN(root, cores_out)


def ttnn(ttn_in: TTN, target_r: int, p: int = 0,
         rng_seed: Optional[int] = None) -> TTN:
    """TTNN (tree tensor network Nystrom) on TTN-format input — fully structured.

    Two-sided generalized Nystrom: per node v the factor is (M_v X)(Yᵀ M_v X)⁺ Yᵀ
    with column sketch X (width r) and row sketch Y (width r+p).  BOTH sketches
    are Khatri-Rao products of per-leaf Gaussians, so the whole pass is contracted
    structurally and never materialises the physical subtree dimension
    n_subtree(v) = n^(#leaves under v).  Peak work/memory is bounded by the bond
    dimensions (like TTN-SVD / QB-Tree), independent of n.

    This replaces the earlier dense-row variant (now `ttnn_dense`), whose dense
    Y_v of shape (n_subtree(v), r+p) made its cost scale as n^(#leaves) -- e.g.
    n^3 on the Figure-1 tree.  For identical Khatri-Rao sketches the two produce
    the same output (verified); against a dense Gaussian row sketch the accuracy
    is statistically equivalent.

    Stability note (unchanged from the dense variant): with FIXED small
    oversampling the inner matrix Yᵀ M_v X becomes near-square and can be
    ill-conditioned, so on slowly-decaying spectra the oblique Nystrom projectors
    can compound across the tree; use proportional oversampling (p ∝ r) to curb
    it.  Two-sided Nystrom is intrinsically weaker than the one-sided range
    finders (TTN-HMT) and QBTC's sketch+SVD on hard tensors.
    """
    rng = np.random.default_rng(rng_seed)
    factors = _ttnn_sketch_factors(ttn_in, target_r, p, rng)
    return _ttnn_from_factors(ttn_in, target_r, p, factors)
