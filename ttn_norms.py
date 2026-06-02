"""
Frobenius norm and inner product on TTN-format tensors via tree contractions
— no dense materialisation at any point.

For two TTNs A and B on the SAME tree topology (same root Node, same physical
dim on every leaf), the inner product

    <A, B> = sum_{all index combinations}  A[idx] * B[idx]

is a single tensor contraction over the tree. At each leaf we form a small
gram

    G_leaf = cores_A[leaf] @ cores_B[leaf].T          # (R_pA, R_pB)

and propagate it upward. At each internal node v with cores_A[v] of shape
(R_pA, R_c1A, ..., R_cmA) and cores_B[v] of shape (R_pB, R_c1B, ..., R_cmB),
the contracted gram at the parent-bond is

    G_v[A, B] = sum_{c_iA, c_iB}  cores_A[v][A, c_1A, ..., c_mA]
                                * cores_B[v][B, c_1B, ..., c_mB]
                                * prod_i G_ci[c_iA, c_iB].

At the root (no parent-bond) the contraction reduces to a scalar.

From <A, B> we get:

    ||A||_F             = sqrt(<A, A>)
    ||A - B||_F^2       = <A, A> - 2 <A, B> + <B, B>          (parallelogram identity)

Cost per node:  O(R_pA · R_pB · prod_i R_ciA · R_ciB) — bounded by bond dims,
NEVER by physical dims n nor by n_above. So <A, B> at paper scale (n = 500)
is essentially free.

Representative spectrum (for OED reference). The spectrum of any binary
unfolding M_v of A is recoverable from R_v × R_v Gram matrices:

    spec(M_v M_v.T)[nonzero]  =  spec( G_below(v) @ G_above(v) )

where G_above(v) = above_V @ above_V.T (already implemented in ttn_format
as compute_above_gram) and G_below(v) = subtree_V @ subtree_V.T is the
analogous Gram on the below-side, computed by walking v's subtree with two
copies of the cores in lockstep.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
from string import ascii_letters
from typing import Optional
import sys

from tree_sketch import Node
from ttn_format import TTN, contract_network


# ---------------------------------------------------------------------
# 1. Inner product <A, B> via tree contraction.
# ---------------------------------------------------------------------
def ttn_inner(A: TTN, B: TTN) -> float:
    """Inner product <A, B> as a scalar.

    A and B must share the same tree topology and the same physical dimension
    on every leaf. Bond dimensions on internal edges may differ (and typically
    do — e.g., A is the input TTN at rank R_input and B is the compressed
    TTN at target_r).
    """
    if A.root is not B.root:
        raise ValueError("ttn_inner requires the same tree (identical root Node)")

    def _node_gram(u: Node) -> np.ndarray:
        """Return G of shape (R_pA, R_pB) at u's parent-bond, or a 0-d scalar
        if u is the root.
        """
        if u.is_leaf:
            # cores_A[u]: (R_pA, n_u);  cores_B[u]: (R_pB, n_u).
            return A.cores[u] @ B.cores[u].T                  # (R_pA, R_pB)

        # Internal — recurse into children, get child grams.
        child_grams = [_node_gram(c) for c in u.children]
        coreA = A.cores[u]
        coreB = B.cores[u]
        m = len(u.children)
        is_root = (u is A.root)

        # Build einsum string with disjoint single-char labels.
        avail = iter(ascii_letters)
        if not is_root:
            pA = next(avail); pB = next(avail)
        ciA = [next(avail) for _ in range(m)]
        ciB = [next(avail) for _ in range(m)]

        coreA_str = ("" if is_root else pA) + "".join(ciA)
        coreB_str = ("" if is_root else pB) + "".join(ciB)
        gram_strs = [ciA[i] + ciB[i] for i in range(m)]
        out_str = "" if is_root else pA + pB

        spec = coreA_str + "," + coreB_str + "," + ",".join(gram_strs) + "->" + out_str
        return contract_network(spec, coreA, coreB, *child_grams)

    return float(_node_gram(A.root))


def frobenius_norm(A: TTN) -> float:
    """||A||_F via TTN contraction. No dense materialisation."""
    val = ttn_inner(A, A)
    # Numerical safety against tiny negatives from rounding.
    return float(np.sqrt(max(val, 0.0)))


def frobenius_diff_norm(A: TTN, B: TTN) -> float:
    """||A - B||_F via TTN orthogonalisation to root canonical form.

    Forms the difference TTN D = A - B, then sweeps leaves-to-root doing a
    QR factorisation of each non-root core and pushing the R-factor into
    the parent's corresponding slot. After the sweep, every non-root core
    is "left-orthogonal" (its matricisation has orthonormal columns
    pointing toward the parent-bond) and ||D||_F = ||cores[root]||_F.

    This is numerically stable: the parallelogram identity formula
    ||A-B||^2 = <A,A> - 2<A,B> + <B,B> loses up to 16 digits to
    cancellation when ||A-B|| << ||A||, ||B||, but the orthogonalised
    norm computes ||D||^2 as a single sum of squares with no
    large-number subtraction — accurate to machine epsilon relative to
    ||A||.

    Cost: one TTN construction (block-diagonal D) plus one QR per
    non-root node and an R-push at each step. Bond dim during the sweep
    is at most R_A + R_B at each edge.
    """
    D = ttn_diff(A, B)
    return _frobenius_norm_via_orthogonalisation(D)


def _frobenius_norm_via_orthogonalisation(ttn: TTN) -> float:
    """Compute ||ttn||_F by gauging the TTN to root canonical form.

    Leaves-to-root QR sweep:
      * At each non-root node v with core shape
            (R_p, *axes_other_than_parent_bond),
        matricise to (prod(other_axes), R_p), QR-factor as Q @ R, replace
        the core with Q (reshaped back), and contract R into the parent's
        slot for v.
      * The TTN is mutated in place — caller should pass a TTN it owns.
    After the sweep, ||ttn|| = ||ttn.cores[root]||_F.
    """
    root = ttn.root
    # Leaves-to-root processing order: any post-order that puts children
    # before their parent. We exclude root since it's the final answer.
    order = _post_order_excluding_root(root)
    for v in order:
        core = ttn.cores[v]
        if v.is_leaf:
            # core: (R_p, n) — already in (parent-bond, other) layout.
            R_p = core.shape[0]
            mat = core.T                                  # (n, R_p)
        else:
            # core: (R_p, R_c1, ..., R_cm) — flatten the children axes
            # into a single "other" axis, parent-bond becomes columns.
            R_p = core.shape[0]
            n_other = int(np.prod(core.shape[1:]))
            mat = core.reshape(R_p, n_other).T            # (n_other, R_p)
        # QR of mat. Q has shape (mat_rows, k); R has shape (k, R_p)
        # where k = min(mat_rows, R_p).
        Q, R = np.linalg.qr(mat, mode='reduced')
        k = R.shape[0]
        # Replace core by Q reshaped back into the same axis layout, with
        # the parent-bond now of size k instead of R_p.
        if v.is_leaf:
            ttn.cores[v] = Q.T                            # (k, n)
        else:
            new_shape = (k,) + core.shape[1:]
            ttn.cores[v] = Q.T.reshape(new_shape)         # (k, R_c1, ..., R_cm)
        # Push R into parent's slot for v. Parent's core has v's parent-bond
        # axis at position child_idx + (0 if parent is root else 1).
        parent = v.parent
        child_idx = parent.children.index(v)
        is_root_parent = (parent is root)
        ax = child_idx + (0 if is_root_parent else 1)
        parent_core = ttn.cores[parent]
        # Move the R_p axis to front, multiply by R, move back.
        pc = np.moveaxis(parent_core, ax, 0)              # (R_p, ...)
        pc_flat = pc.reshape(pc.shape[0], -1)
        new_pc_flat = R @ pc_flat                         # (k, prod(rest))
        new_pc = new_pc_flat.reshape((k,) + pc.shape[1:])
        ttn.cores[parent] = np.moveaxis(new_pc, 0, ax)

    return float(np.linalg.norm(ttn.cores[root]))


def _post_order_excluding_root(root):
    """Post-order traversal — children before parents — excluding root."""
    out = []
    def _rec(u):
        if u.is_leaf:
            out.append(u)
            return
        for c in u.children:
            _rec(c)
        if u is not root:
            out.append(u)
    _rec(root)
    return out


def ttn_diff(A: TTN, B: TTN) -> TTN:
    """Construct a TTN representing A - B (same tree topology required).

    The output bond dim at every internal edge is R_A_edge + R_B_edge.
    Cores are filled block-diagonally:
      * leaf v:       core_D[:R_pA, :] = A[v];  core_D[R_pA:, :] = B[v]
                       (concatenation along the parent-bond axis)
      * non-root v:   core_D is zero except for two diagonal blocks
                       — A's core in the (:R_pA, :R_c1A, ..., :R_cmA)
                       block and B's core in the (R_pA:, R_c1A:, ..., R_cmA:)
                       block.
      * root:         same block-diagonal pattern but in children-bonds only,
                       and B's block is NEGATED to encode the minus sign.

    Negating exactly one core (the root core) gives the difference instead
    of the sum; flipping any single core works, root is just a convenient
    choice.
    """
    if A.root is not B.root:
        raise ValueError("ttn_diff requires the same tree (identical root Node)")

    diff_cores: dict = {}
    root = A.root
    for v in [u for u in _iter_nodes(root)]:
        cA = A.cores[v]
        cB = B.cores[v]
        if v.is_leaf:
            # cA: (R_pA, n);  cB: (R_pB, n).  Concatenate along axis 0.
            diff_cores[v] = np.concatenate([cA, cB], axis=0)
        elif v is root:
            # cA: (R_c1A, ..., R_cmA);  cB: (R_c1B, ..., R_cmB).
            m = len(v.children)
            shape_out = tuple(cA.shape[i] + cB.shape[i] for i in range(m))
            C = np.zeros(shape_out, dtype=cA.dtype)
            slices_A = tuple(slice(0, cA.shape[i]) for i in range(m))
            slices_B = tuple(slice(cA.shape[i], shape_out[i]) for i in range(m))
            C[slices_A] = cA
            C[slices_B] = -cB                 # NEGATE B's root block for A - B
            diff_cores[v] = C
        else:
            # Non-root internal: cA, cB shape (R_p, R_c1, ..., R_cm).
            m = len(v.children)
            sA = cA.shape
            sB = cB.shape
            shape_out = (sA[0] + sB[0],) + tuple(sA[i+1] + sB[i+1] for i in range(m))
            C = np.zeros(shape_out, dtype=cA.dtype)
            slices_A = (slice(0, sA[0]),) + tuple(slice(0, sA[i+1]) for i in range(m))
            slices_B = (slice(sA[0], shape_out[0]),) + \
                       tuple(slice(sA[i+1], shape_out[i+1]) for i in range(m))
            C[slices_A] = cA
            C[slices_B] = cB                  # NOT negated — only root negates
            diff_cores[v] = C

    return TTN(root, diff_cores)


def _iter_nodes(root):
    """All nodes in the tree (root first, then DFS)."""
    stack = [root]
    out = []
    while stack:
        u = stack.pop()
        out.append(u)
        if not u.is_leaf:
            for c in u.children:
                stack.append(c)
    return out


# ---------------------------------------------------------------------
# 2. Below-gram G_below(v) = subtree_V @ subtree_V.T  of shape (R_v, R_v).
# ---------------------------------------------------------------------
def compute_below_gram(ttn: TTN, v: Node) -> np.ndarray:
    """G_below(v) = subtree_V @ subtree_V.T  (R_v × R_v).

    Walks v's subtree contracting two copies of every core in lockstep.
    No dense materialisation. Cost per subtree node: O(R^4) for binary
    children (or O(R^{2(1+m)}) for higher-degree nodes).
    """
    if v is ttn.root:
        raise ValueError("Root has no parent-bond; G_below(root) is a scalar")

    def _expand(u: Node) -> np.ndarray:
        """Return shape (R_pu, R_pu) — gram at u's parent-bond contracted
        through u's whole subtree."""
        if u.is_leaf:
            return ttn.cores[u] @ ttn.cores[u].T              # (R_p, R_p)
        child_grams = [_expand(c) for c in u.children]
        core = ttn.cores[u]                                    # (R_p, R_c1, ..., R_cm)
        m = len(u.children)

        avail = iter(ascii_letters)
        pA = next(avail); pB = next(avail)
        ciA = [next(avail) for _ in range(m)]
        ciB = [next(avail) for _ in range(m)]
        coreA_str = pA + "".join(ciA)
        coreB_str = pB + "".join(ciB)
        gram_strs = [ciA[i] + ciB[i] for i in range(m)]
        spec = (coreA_str + "," + coreB_str + ","
                + ",".join(gram_strs) + "->" + pA + pB)
        return contract_network(spec, core, core, *child_grams)

    return _expand(v)


# ---------------------------------------------------------------------
# 3. Representative singular-spectrum of M_v.
# ---------------------------------------------------------------------
def representative_spectrum(ttn: TTN, v: Node) -> np.ndarray:
    """Singular values of the unfolding M_v of `ttn` at node v.

    Uses the identity: nonzero spec(M_v M_v.T) = spec( G_below(v) @ G_above(v) ).
    Both Grams are R_v × R_v. We symmetrise via Cholesky of G_below to feed
    eigh and recover sigma_i = sqrt(eigvals).

    Returns sigma sorted descending. There are at most R_v nonzero values.
    No n_above or n_below materialisation.
    """
    G_above = ttn.compute_above_gram(v)                       # (R_v, R_v)
    G_below = compute_below_gram(ttn, v)                       # (R_v, R_v)
    G_above = 0.5 * (G_above + G_above.T)
    G_below = 0.5 * (G_below + G_below.T)

    # Spec(G_below @ G_above) = spec(L.T @ G_above @ L) where G_below = L L.T.
    # Cholesky may fail if G_below is rank-deficient or has tiny eigenvalues
    # from rounding; fall back to eigh-based square root in that case.
    try:
        L = np.linalg.cholesky(G_below)                        # G_below = L @ L.T
    except np.linalg.LinAlgError:
        # PSD square root via eigendecomposition: G_below = U diag(d) U.T,
        # then sqrt = U diag(sqrt(d_+)) U.T.
        d, U = scipy.linalg.eigh(G_below)
        d = np.clip(d, 0.0, None)
        L = U * np.sqrt(d)                                     # not lower-triangular but works
    M_sym = L.T @ G_above @ L
    M_sym = 0.5 * (M_sym + M_sym.T)
    eigvals = scipy.linalg.eigvalsh(M_sym)
    eigvals = np.clip(eigvals, 0.0, None)
    sigma = np.sqrt(eigvals)
    return np.sort(sigma)[::-1]                                # descending
