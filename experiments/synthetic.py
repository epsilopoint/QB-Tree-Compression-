"""
Synthetic test tensor with PRESCRIBED singular-value spectrum on every binary unfolding.

We build T = sum_{k=1}^{K} sigma_k * a_k^{(1)} (x) a_k^{(2)} (x) ... (x) a_k^{(d)}
where a_k^{(i)} is the k-th column of an n x K matrix Q^{(i)} obtained by QR of an
n x K iid Gaussian. Each Q^{(i)} has orthonormal columns (Q^{(i)*} Q^{(i)} = I_K).

CONSEQUENCE for any "binary" unfolding T_{(I,J)} where {I, J} partitions the modes:

    T_{(I,J)} = ( bigotimes_{i in I} Q^{(i)} ) * diag(sigma) * ( bigotimes_{j in J} Q^{(j)} )^*

Both Kronecker factors have orthonormal columns (Kronecker preserves orthonormality),
so the displayed expression is an SVD of T_{(I,J)}: its singular values are EXACTLY
sigma_1, ..., sigma_K (followed by zeros). This is true for every partition of the
modes, hence for every binary unfolding induced by an edge of any tensor-tree topology.

In particular, choosing sigma_k = k^{-alpha} produces a tensor whose every binary
unfolding has a k^{-alpha} singular-value tail, and the best rank-r truncation error
(in Frobenius norm) at any edge is sqrt(sum_{k>r} sigma_k^2). This makes the tensor a
controlled, repeatable benchmark for any tree-tensor-network compression method.

The Bucci-Verzella TTNN paper uses tensors of this exact form; see also the analyses
in Grasedyck (HSVD), Kressner-Tobler (HT), and the Cazeaux et al. TTStack paper.
"""

import numpy as np
from typing import Sequence, Optional


def cp_synthetic(
    shape: Sequence[int],
    K: int,
    sigma: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a synthetic CP tensor whose every binary unfolding has singular values
    sigma_1, ..., sigma_K (and then zeros).

    Args:
        shape: tuple of mode dimensions (n_1, ..., n_d). All must be >= K.
        K: number of CP terms (= rank of every binary unfolding).
        sigma: array of length K specifying the singular values. If None, uses k^{-alpha}.
        alpha: power-law exponent if sigma is None. Default 0.5.
        rng: numpy Generator (default new with random seed).

    Returns:
        T: ndarray of shape `shape`.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(shape)
    for n in shape:
        if n < K:
            raise ValueError(f"every dim must be >= K={K}, got {shape}")
    if sigma is None:
        if alpha is None:
            alpha = 0.5
        sigma = (np.arange(1, K + 1, dtype=np.float64)) ** (-alpha)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (K,):
        raise ValueError(f"sigma must have shape ({K},), got {sigma.shape}")

    Qs = []
    for n in shape:
        A = rng.standard_normal((n, K))
        Q, _ = np.linalg.qr(A)  # Q shape (n, K), orthonormal cols
        Qs.append(Q)

    # T[i_1, ..., i_d] = sum_k sigma_k * Q^{(1)}[i_1, k] * ... * Q^{(d)}[i_d, k]
    # Build via einsum: 'i1 k, i2 k, ..., id k, k -> i1 i2 ... id'
    from string import ascii_letters
    avail = list(ascii_letters)
    k_lbl = avail.pop(0)
    mode_lbls = [avail.pop(0) for _ in range(d)]
    Q_terms = ["".join([m, k_lbl]) for m in mode_lbls]
    spec = ",".join(Q_terms + [k_lbl]) + "->" + "".join(mode_lbls)
    # Route through the library helper so this builder uses the same
    # flop-minimizing path (opt_einsum) as the rest of the codebase.  Local
    # import keeps synthetic.py importable on its own.
    from ttn_format import contract_network
    T = contract_network(spec, *Qs, sigma)
    return T


def yukawa_radial_tensor(d: int, n: int, kappa: float = 1.0,
                         domain=(0.0, 1.0), radius: float = None) -> np.ndarray:
    """
    Discretize a radially-symmetric Yukawa-screened "spherical well" potential
    centered in the box:

        f(x_1, ..., x_d) = max(0, radius - ||x - c||) * exp(-kappa * ||x - c||),
        ||y||_2 = sqrt(y_1^2 + ... + y_d^2),
        c = ((a + b) / 2, ..., (a + b) / 2).

    Discretized on the grid

        x_j = a + (j - 0.5) * (b - a) / n,    j = 1, 2, ..., n.

    The factor max(0, radius - r) is a "compactly-supported tent" — it is
    POSITIVE inside the ball ||x - c|| < radius, ZERO outside, and has a KINK
    (continuous but not C^1) on the sphere ||x - c|| = radius. The screening
    factor exp(-kappa * r) is smooth and only modulates the amplitude.

    Crucially, the kink occurs on a (d-1)-dimensional sphere that intersects
    the grid in O(n^{d-1}) points; this makes the spectral decay of every
    binary unfolding GENUINELY ALGEBRAIC. The function belongs to H^{s} with
    s < 3/2 (Sobolev: the gradient has a jump across the sphere; second
    derivatives blow up there as a delta on the (d-1)-dim manifold). Every
    binary unfolding then satisfies
        sigma_k = O(k^{-alpha})
    with alpha around 1.0 to 1.5.

    The function is GENUINELY MULTIVARIATE: although it has the radial form
    g(||x - c||), the kink at radius DOES couple all d modes non-separably
    (max(0, radius - sqrt(sum of x_i^2)) is NOT a sum of low-d functions).
    This contrasts with pair-additive structures sum_{i<j} V(x_i, x_j) which
    give artificially low-rank discretizations.

    Physically:
      * Wavefunction of a particle in a finite spherical well (truncated
        radial profile).
      * Density of a uniform ball in d-dim space (after differentiation).
      * Yukawa-screened version of a confined-region potential, used in
        plasma physics and effective field theory as a smoothed cutoff.
      * Setting kappa = 0 gives the bare tent function (no screening, just kink).

    Args:
        d: spatial dimension; must be >= 2.
        n: grid points per axis; must be EVEN so c does not coincide with a
           grid point.
        kappa: Yukawa screening parameter. kappa = 0 means no screening.
        domain: tuple (a, b) giving the per-axis box.
        radius: radius of the spherical well. Must be a float in
                (0, sqrt(d) * (b-a)/2). Default chooses radius so the kink
                sphere passes roughly through the bulk of the grid:
                    radius = 0.5 * sqrt(d) * (b - a) / 2.

    Returns:
        T: ndarray of shape (n,) * d.
    """
    if d < 2:
        raise ValueError("yukawa_radial_tensor needs d >= 2")
    if n % 2 != 0:
        raise ValueError("n must be even so the center of the box does not "
                         "land on a grid point")
    a, b = domain
    c = 0.5 * (a + b)
    x = a + (np.arange(1, n + 1, dtype=np.float64) - 0.5) * (b - a) / n
    if radius is None:
        # Default: half of the corner-to-center distance sqrt(d) * (b-a)/2.
        # This places the kink-sphere in the bulk of the grid.
        radius = 0.5 * np.sqrt(d) * (b - a) / 2.0

    r2 = np.zeros((n,) * d, dtype=np.float64)
    for i in range(d):
        shape_i = [1] * d
        shape_i[i] = n
        r2 = r2 + (x.reshape(shape_i) - c) ** 2
    r = np.sqrt(r2)
    return np.maximum(0.0, radius - r) * np.exp(-kappa * r)


def best_rank_r_error_at_any_edge(K: int, r: int, sigma: np.ndarray) -> float:
    """
    Frobenius-norm best rank-r approximation error at ANY binary unfolding of the
    synthetic CP tensor with singular values sigma. Identical across all edges.
    """
    if r >= K:
        return 0.0
    return float(np.sqrt(np.sum(sigma[r:] ** 2)))


def synthetic_ttn_with_decay(
    tree,
    n: int,
    R_input: int,
    sigma: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a tensor in TTN format with prescribed singular-value decay sigma on every
    binary edge unfolding, then materialize it as a dense numpy array.

    This mirrors the recipe described in section 8 of the Bucci-Verzella TTNN paper
    (the experiment of Figure 10):

      * Each leaf is a Haar-distributed orthogonal (n, R_input) matrix.
      * Each non-leaf, non-root core has shape (R_input,) * (1 + m_v) where m_v is
        the number of children of v: the parent-bond plus one bond per child. The
        core is a SUPERDIAGONAL tensor whose diagonal entries are sigma[0..R_input-1],
        and zero elsewhere; the diagonal core is then rotated by an independent
        Haar-orthogonal (R_input, R_input) matrix along EACH of its modes.
      * The root core has shape (R_input,) * m_root (no parent-bond), built the
        same way (superdiagonal + Haar rotations on every mode).

    Why this gives exact sigma decay on every edge.  Cutting any single internal edge
    of the TTN partitions the tree into a parent side and a child side; on each side
    the contracted matricization is the product of orthogonal cores (since Kronecker
    products of orthogonal matrices stay orthogonal, and each rotated superdiagonal
    core is orthogonal in the proper matricization). The remaining factor at the cut
    is the rotated superdiagonal core whose off-diagonal block has singular values
    exactly sigma[0..R_input-1]; the orthogonal flanks preserve them. Hence every
    binary unfolding has SVs = sigma.

    Parameters
    ----------
    tree : Node
        Root node of the index tree (e.g., from build_figure1_tree()).
    n : int
        Physical dim per leaf (every leaf has the same n).
    R_input : int
        Internal TTN rank used in the construction (= length of sigma we use; the
        tensor has TTN-rank R_input on every edge by construction).
    sigma : np.ndarray
        Length-R_input array of nonincreasing nonnegative reals giving the prescribed
        per-edge singular values.
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    T : np.ndarray of shape (n,) * d, where d is the number of leaves of `tree`.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if sigma.shape != (R_input,):
        raise ValueError(f"sigma must have shape ({R_input},), got {sigma.shape}")
    if n < R_input:
        # A Haar-orthogonal n x R_input matrix needs orthonormal columns, which
        # requires n >= R_input. (Otherwise economy QR returns an n x n matrix.)
        raise ValueError(
            f"n must be >= R_input so leaf matrices have R_input orthonormal "
            f"columns; got n={n}, R_input={R_input}"
        )

    # Number of leaves and their physical_axis labels.
    leaves = []

    def collect_leaves(node):
        if node.is_leaf:
            leaves.append(node)
        else:
            for c in node.children:
                collect_leaves(c)

    collect_leaves(tree)
    d = len(leaves)
    if d < 2:
        raise ValueError("Tree needs at least 2 leaves")

    def haar_orthogonal(rows: int, cols: int) -> np.ndarray:
        """Random Haar-distributed orthonormal-columns matrix of shape (rows, cols)."""
        if rows < cols:
            raise ValueError(f"haar_orthogonal needs rows >= cols (got {rows}, {cols})")
        # Standard recipe: A iid Gaussian -> economy QR -> Q with sign normalisation.
        A = rng.standard_normal((rows, cols))
        Q, Rmat = np.linalg.qr(A, mode="reduced")  # Q: (rows, cols), Rmat: (cols, cols)
        # Mardia/Stewart sign-normalisation so the distribution is uniform on the
        # Stiefel manifold (without this, Q's distribution depends on sign(R_ii)).
        sign = np.sign(np.diag(Rmat))
        sign[sign == 0] = 1.0
        return Q * sign  # broadcasts column-wise: column j multiplied by sign[j]

    def superdiag_then_rotate(num_modes: int) -> np.ndarray:
        """
        Build a (R_input,) * num_modes tensor: superdiagonal with sigma on the
        diagonal, then multiply (mode-by-mode) by independent Haar-orthogonal
        (R_input, R_input) matrices on every mode. The result preserves the
        singular-value spectrum on every matricization.
        """
        shape = (R_input,) * num_modes
        D = np.zeros(shape, dtype=np.float64)
        idx = (np.arange(R_input),) * num_modes
        D[idx] = sigma
        for axis in range(num_modes):
            U = haar_orthogonal(R_input, R_input)
            # Multiply along `axis`: contract D's axis with U's first axis, restore axis order.
            D = np.tensordot(D, U, axes=([axis], [1]))
            # tensordot moved the contracted axis to the end; roll it back to position `axis`.
            D = np.moveaxis(D, -1, axis)
        return D

    # Build leaf matrices (n, R_input) Haar-orthogonal.
    leaf_mat = {leaf.name: haar_orthogonal(n, R_input) for leaf in leaves}

    # Materialize via post-order traversal: each subtree returns a dense tensor
    # over the physical axes of its leaves, with one extra "open" axis at the
    # parent-bond. We keep a list of physical_axis labels alongside each subtree
    # tensor so we can reorder modes at the end.

    def subtree(node):
        """
        Returns (T_sub, axis_labels) where T_sub is a dense tensor with axes
            [parent_bond, leaf_phys_axis_1, leaf_phys_axis_2, ...]
        and axis_labels is the list of original physical_axis indices of the
        leaves in order. For the root call we pass node=root and no parent-bond
        will be present (handled separately).
        """
        if node.is_leaf:
            # leaf_mat shape (n, R_input); we want axes [parent_bond=R_input, phys=n]
            M = leaf_mat[node.name].T  # shape (R_input, n)
            return M, [node.physical_axis]
        # Non-leaf, non-root: core has shape (R_input,) * (1 + m), axis 0 = parent bond.
        m = len(node.children)
        core = superdiag_then_rotate(num_modes=1 + m)
        # Contract each child's parent-bond axis (axis 0 of child tensor) with the
        # corresponding child-bond axis of the core.
        # We build the result iteratively: start from `core`, contract child 1 into
        # the second axis, then child 2 into the (now first) child-bond axis, etc.
        T_sub = core
        labels = []  # leaf physical-axis labels in the order they end up in T_sub
        for ci, child in enumerate(node.children):
            child_T, child_labels = subtree(child)
            # T_sub axes: [parent, child0_bond, child1_bond, ..., (any leaf axes already glued)]
            # The first child-bond axis to contract is currently at position 1.
            T_sub = np.tensordot(T_sub, child_T, axes=([1], [0]))
            # tensordot leaves axes as: [parent, remaining child-bonds..., already-glued leaf axes..., new leaf axes from child_T]
            labels = labels + child_labels
        return T_sub, labels

    # Special-case root: same as non-leaf, non-root but core has m_root axes (NO parent bond).
    m_root = len(tree.children)
    root_core = superdiag_then_rotate(num_modes=m_root)
    T = root_core
    labels: list = []
    for child in tree.children:
        child_T, child_labels = subtree(child)
        T = np.tensordot(T, child_T, axes=([0], [0]))
        labels = labels + child_labels
    # Now T has axes corresponding to physical_axes in the order `labels`. Permute
    # to natural order 0, 1, ..., d-1.
    perm = np.argsort(labels)
    T = T.transpose(perm)
    return T


def synthetic_ttn_decay_object(
    tree,
    n: int,
    R_input: int,
    sigma: np.ndarray,
    rng: Optional[np.random.Generator] = None,
):
    """
    Build the SAME prescribed-decay tensor as ``synthetic_ttn_with_decay``, but
    return it DIRECTLY as a TTN object (cores only), WITHOUT ever forming the
    dense (n,)**d array.

    This is what makes Bucci-scale mode sizes (e.g. n = 500) feasible: the cores
    are tiny (a leaf is R_input x n, an interior node is R_input**(1+m)); only
    the dense contraction performed by ``synthetic_ttn_with_decay`` exploded as
    n**d. The tensor is exactly TTN-rank R_input on every edge by construction
    (each edge unfolding has singular values exactly sigma), so no compression
    happens at build time and the input gap is 0.

    Cores follow ttn_format's convention:
      * leaf v          -> (R_input, n)              (parent-bond, physical)
      * internal non-root v (m children) -> (R_input,)*(1+m)   (parent-bond, child-bonds)
      * root v (m children)              -> (R_input,)*m       (child-bonds only)
    """
    from ttn_format import TTN  # lazy import; repo root is on sys.path in the experiments

    if rng is None:
        rng = np.random.default_rng()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if sigma.shape != (R_input,):
        raise ValueError(f"sigma must have shape ({R_input},), got {sigma.shape}")
    if n < R_input:
        raise ValueError(
            f"n must be >= R_input so leaf matrices have R_input orthonormal "
            f"columns; got n={n}, R_input={R_input}"
        )

    def haar_orthogonal(rows: int, cols: int) -> np.ndarray:
        if rows < cols:
            raise ValueError(f"haar_orthogonal needs rows >= cols (got {rows}, {cols})")
        A = rng.standard_normal((rows, cols))
        Q, Rmat = np.linalg.qr(A, mode="reduced")
        sign = np.sign(np.diag(Rmat))
        sign[sign == 0] = 1.0
        return Q * sign

    def superdiag_then_rotate(num_modes: int) -> np.ndarray:
        shape = (R_input,) * num_modes
        D = np.zeros(shape, dtype=np.float64)
        idx = (np.arange(R_input),) * num_modes
        D[idx] = sigma
        for axis in range(num_modes):
            U = haar_orthogonal(R_input, R_input)
            D = np.tensordot(D, U, axes=([axis], [1]))
            D = np.moveaxis(D, -1, axis)
        return D

    cores: dict = {}

    def build(node, is_root: bool):
        if node.is_leaf:
            # (n, R_input) Haar with orthonormal columns -> transpose to (R_input, n)
            cores[node] = haar_orthogonal(n, R_input).T
            return
        m = len(node.children)
        cores[node] = superdiag_then_rotate(m if is_root else 1 + m)
        for c in node.children:
            build(c, is_root=False)

    build(tree, is_root=True)
    return TTN(root=tree, cores=cores)


def total_norm(sigma: np.ndarray) -> float:
    """||T||_F = sqrt(sum sigma_k^2) for the synthetic CP tensor."""
    return float(np.sqrt(np.sum(sigma ** 2)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 20
    sigma = (np.arange(1, K + 1, dtype=np.float64)) ** (-0.5)
    T = cp_synthetic(shape=(20,) * 6, K=K, sigma=sigma, rng=rng)
    print(f"T shape:      {T.shape}")
    print(f"||T||_F:      {np.linalg.norm(T):.6f}")
    print(f"expected:     {total_norm(sigma):.6f}")

    # Verify: SVD of an arbitrary binary unfolding has singular values sigma.
    # Take the first 3 modes as rows.
    M = T.reshape(20**3, 20**3)
    s = np.linalg.svd(M, compute_uv=False)
    print(f"\nFirst 5 singular values of (modes 0,1,2 | 3,4,5) unfolding:")
    print(f"  sigma (target):  {sigma[:5]}")
    print(f"  s (computed):    {s[:5]}")
    print(f"  max |s - sigma|: {np.max(np.abs(s[:K] - sigma)):.3e}")
    print(f"  s[K:K+3] (should be ~0):  {s[K:K+3]}")
