"""
TTN data structure for Path A: TTN-aware QBTC.

Convention for core shapes (`ttn.cores[v]`):

  * Leaf v:                    shape (R_parent, n_v)
                               axis 0 = parent-bond, axis 1 = physical mode.
  * Internal non-root v:       shape (R_parent, R_c1, ..., R_cm)
                               axis 0 = parent-bond, axes 1..m = child-bonds
                               in v.children order.
  * Root v:                    shape (R_c1, ..., R_cm)
                               no parent-bond; axes are child-bonds in
                               v.children order.

This convention has axis 0 = parent-bond for every non-root core, which makes
recursive bottom-up contraction in to_dense() trivial.
"""

from __future__ import annotations
from typing import Dict, Optional, List, Tuple, Set
import numpy as np

import sys

from tree_sketch import Node, all_nodes

# opt_einsum gives flop-minimizing contraction paths for many-operand tensor
# networks.  numpy's own optimizer is unsuitable here: "optimal" does not scale
# past ~15 operands, and "greedy" minimizes intermediate *memory*, which on a
# branching tree picks a path that is memory-cheap but flop-catastrophic
# (~R_input^(#leaves) flops -> effectively a hang).  opt_einsum's "auto"/"dp"
# return a bond-bounded path in milliseconds.  We keep a numpy fallback so the
# library still imports without opt_einsum (fine for small / path-like trees).
try:
    import opt_einsum as _oe
    _HAVE_OPT_EINSUM = True
except ImportError:                                   # pragma: no cover
    _HAVE_OPT_EINSUM = False


def _einsum_greedy_flops(*operands):
    """Flop-aware greedy contractor -- the fallback for contract_network when
    opt_einsum is not installed.

    Contracts the network pairwise, at each step choosing the pair whose
    contraction costs the fewest scalar multiplications (the heuristic
    opt_einsum's greedy uses).  This is what keeps cost bounded on a branching
    tree: numpy's own optimize="greedy" instead minimizes intermediate *memory*
    and can pick a path costing ~R^(#leaves) flops (an effective hang) while
    keeping every intermediate small.  The result is identical to np.einsum
    (same contraction, different order).  Accepts both np.einsum signatures.
    """
    from collections import Counter

    # ---- normalize to (array, raw-index-tuple) terms + a raw output tuple ----
    if isinstance(operands[0], str):
        subs = operands[0].replace(" ", "")
        arrays = list(operands[1:])
        ins, arrow, out = subs.partition("->")
        terms = [(arrays[i], tuple(t)) for i, t in enumerate(ins.split(","))]
        out_idx = tuple(out) if arrow else None
    else:
        ops = list(operands)
        out_idx = None
        if len(ops) % 2 == 1:                       # trailing output sublist
            out_idx = tuple(ops[-1]); ops = ops[:-1]
        terms = [(ops[i], tuple(ops[i + 1])) for i in range(0, len(ops), 2)]

    if out_idx is None:                             # implied output: indices used once
        cnt = Counter(i for _, idx in terms for i in set(idx))
        out_idx = tuple(sorted(i for i in cnt if cnt[i] == 1))

    # ---- relabel all indices (chars OR ints) to consecutive ints for sublists ----
    lab = {}
    def _L(i):
        if i not in lab:
            lab[i] = len(lab)
        return lab[i]
    terms = [(arr, tuple(_L(i) for i in idx)) for arr, idx in terms]
    out_idx = tuple(_L(i) for i in out_idx)
    out_set = set(out_idx)

    dim = {}
    for arr, idx in terms:
        for ax, i in enumerate(idx):
            dim[i] = arr.shape[ax]

    # ---- greedy pairwise contraction (minimize per-step flops) ----
    terms = list(terms)
    while len(terms) > 1:
        occ = Counter()
        for _, idx in terms:
            occ.update(set(idx))
        best = None                                 # ((cost, rsize), a, b, res_idx)
        for a in range(len(terms)):
            ia = set(terms[a][1])
            for b in range(a + 1, len(terms)):
                ib = set(terms[b][1])
                if ia.isdisjoint(ib):
                    continue                        # skip outer products
                union = ia | ib
                elim = {i for i in union
                        if occ[i] == (i in ia) + (i in ib) and i not in out_set}
                res = tuple(union - elim)
                cost = 1
                for i in union:
                    cost *= dim[i]
                rsize = 1
                for i in res:
                    rsize *= dim[i]
                if best is None or (cost, rsize) < best[0]:
                    best = ((cost, rsize), a, b, res)
        if best is None:                            # disconnected: outer-product two smallest
            order = sorted(range(len(terms)),
                           key=lambda t: int(np.prod([dim[i] for i in terms[t][1]] or [1])))
            a, b = sorted(order[:2])
            res = tuple(set(terms[a][1]) | set(terms[b][1]))
        else:
            _, a, b, res = best
        arrA, idxA = terms[a]
        arrB, idxB = terms[b]
        r = np.einsum(arrA, list(idxA), arrB, list(idxB), list(res))
        terms.pop(b); terms.pop(a)                  # b > a, remove it first
        terms.append((r, res))

    arr, idx = terms[0]
    if tuple(idx) != out_idx:
        arr = np.einsum(arr, list(idx), list(out_idx))
    return arr


def contract_network(*operands):
    """Contract a multi-operand einsum network with a flop-minimizing path.

    Accepts either np.einsum signature:
      * subscript-string form:  contract_network("ab,bc->ac", A, B, ...)
      * interleaved sublist form: contract_network(A, [0,1], B, [1,2], [0,2])

    Uses opt_einsum's "auto"/"dp" path when available -- flop-optimal-or-near and
    found in milliseconds even for whole-subtree networks of 20+ operands.  When
    opt_einsum is absent it uses a flop-aware greedy fallback
    (_einsum_greedy_flops).  numpy's own "greedy" is deliberately NOT used: it
    minimizes intermediate memory and can pick a flop-catastrophic order
    (~R^(#leaves) flops) on a branching tree.  Every path-choosing contraction in
    the package routes through here, so no tree shape, node degree, or bond-size
    combination can resurrect that blow-up.
    """
    if _HAVE_OPT_EINSUM:
        return _oe.contract(*operands, optimize="auto")
    return _einsum_greedy_flops(*operands)


def _post_order(root: Node) -> List[Node]:
    """Children-before-parent order; root is last."""
    out: List[Node] = []

    def rec(v: Node):
        for c in v.children:
            rec(c)
        out.append(v)
    rec(root)
    return out


def _subtree_leaves(v: Node) -> List[Node]:
    """All leaves in the subtree rooted at v."""
    if v.is_leaf:
        return [v]
    out: List[Node] = []
    for c in v.children:
        out.extend(_subtree_leaves(c))
    return out


class TTN:
    """A tensor in TTN format.

    The cores are stored with the multi-axis convention described in this
    module's docstring. The full tensor T has shape (n_0, n_1, ..., n_{d-1})
    where d is the number of leaves and n_p is the physical_axis-p leaf's
    physical dim.
    """

    def __init__(self, root: Node, cores: Dict[Node, np.ndarray]):
        self.root = root
        # Copy the dict so callers can't mutate internals; keep arrays themselves
        # (arrays are big; we trust the caller not to mutate them in place).
        self.cores: Dict[Node, np.ndarray] = dict(cores)
        # Set of nodes whose subtrees have been processed by residual_update().
        # For a processed node v, cores[v] is an identity matrix and v acts as a
        # "leaf with compressed bond" — i.e., its children's cores are still
        # present (and unchanged after their OWN processing earlier) but they
        # don't need to be touched again. v itself contributes only an identity.
        self.processed: Set[Node] = set()
        self._validate()

    # --------------------------------------------------------------
    # Construction / validation
    # --------------------------------------------------------------

    def _validate(self):
        """Check that all cores are present and have consistent shapes."""
        nodes = all_nodes(self.root)
        missing = [v for v in nodes if v not in self.cores]
        if missing:
            raise ValueError(f"Missing cores: {[v.name for v in missing]}")

        for v in nodes:
            core = self.cores[v]
            if v.is_leaf:
                if core.ndim != 2:
                    raise ValueError(
                        f"Leaf {v.name}: expected 2-axis core (R_parent, n), "
                        f"got shape {core.shape}"
                    )
            elif v is self.root:
                if core.ndim != len(v.children):
                    raise ValueError(
                        f"Root {v.name}: expected {len(v.children)} axes "
                        f"(one per child), got shape {core.shape}"
                    )
            else:
                expected_ndim = 1 + len(v.children)
                if core.ndim != expected_ndim:
                    raise ValueError(
                        f"Internal {v.name}: expected {expected_ndim} axes "
                        f"(parent + {len(v.children)} children), got shape {core.shape}"
                    )

        # Cross-check: parent-bond of v (axis 0) must equal child-bond of parent
        # (axis 1+i where i = v's index in parent.children).
        for v in nodes:
            if v is self.root:
                continue
            parent = v.parent
            i = parent.children.index(v)
            v_parent_bond = self.cores[v].shape[0]
            if parent is self.root:
                parent_child_bond = self.cores[parent].shape[i]
            else:
                parent_child_bond = self.cores[parent].shape[1 + i]
            if v_parent_bond != parent_child_bond:
                raise ValueError(
                    f"Bond mismatch: {v.name} parent-bond={v_parent_bond}, "
                    f"{parent.name} child-bond[{i}]={parent_child_bond}"
                )

    # --------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------

    @property
    def d(self) -> int:
        """Number of leaves = order of the underlying tensor."""
        return len(_subtree_leaves(self.root))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Physical shape of the underlying tensor in physical_axis order."""
        leaves = _subtree_leaves(self.root)
        leaves_by_axis = sorted(leaves, key=lambda v: v.physical_axis)
        return tuple(self.cores[lv].shape[1] for lv in leaves_by_axis)

    def parent_bond_dim(self, v: Node) -> int:
        """R_v: the bond dimension between v and v's parent."""
        if v is self.root:
            raise ValueError("Root has no parent-bond")
        return self.cores[v].shape[0]

    # --------------------------------------------------------------
    # Materialization (to_dense)
    # --------------------------------------------------------------

    def to_dense(self, return_labels: bool = False):
        """Materialize as a dense numpy array of shape self.shape.

        WARNING: only feasible for small examples (n^d <= ~ 10^7 entries on a
        4-GB machine). The point of TTN format is to AVOID materialization.

        Parameters
        ----------
        return_labels : bool
            If False (default), require all axes to correspond to true (un-
            processed) leaves and return the tensor with axes sorted in
            physical_axis order. Errors if any nodes are processed.
            If True, return (T, labels) where `labels` is a list giving the
            label of each axis in T (an int physical_axis for true leaves, or
            a Node object for processed internal nodes). Caller can permute T
            using these labels as needed.
        """
        # Recursively expand each subtree. Each call returns a tensor whose
        # axes are [parent_bond, physical_axes_of_subtree_in_natural_order]
        # plus the leaf-physical-axis labels so we can permute at the end.

        def expand(v: Node):
            """
            Return (T_sub, labels) where:
              T_sub.shape[0]  = R_parent (parent-bond)
              T_sub.shape[1:] = physical / compressed-phys dims of v's subtree
                                in the recursion order.
              labels          = list of axis labels for T_sub.shape[1:].
                                Each label is either an int (physical_axis of
                                a true leaf) or a Node (a processed internal
                                node's compressed-dim axis).
            """
            # Processed: treat as leaf-like (axis 1 of cores[v] is its
            # compressed phys). For true leaves that happen to also be
            # processed, use the int physical_axis label (the axis is still
            # at the same position in the dense tensor, just smaller).
            if v in self.processed:
                if v.is_leaf:
                    return self.cores[v], [v.physical_axis]
                else:
                    return self.cores[v], [v]   # use Node as label

            if v.is_leaf:
                return self.cores[v], [v.physical_axis]

            # Internal, not processed: core shape (R_parent, R_c1, ..., R_cm).
            T_sub = self.cores[v]
            labels = []
            for ci, child in enumerate(v.children):
                child_T, child_labels = expand(child)
                T_sub = np.tensordot(T_sub, child_T, axes=([1], [0]))
                labels = labels + child_labels
            return T_sub, labels

        if self.root.is_leaf:
            raise NotImplementedError("Tree must have at least one internal node")

        # Root has no parent-bond. Treat it the same way (no phantom needed
        # since there's no parent-bond to consume).
        if self.root in self.processed:
            # Edge case: root has been processed (shouldn't happen in QBTC
            # but allow it).
            T = self.cores[self.root]
            labels = [self.root]
        else:
            root_core = self.cores[self.root]
            T = root_core
            labels = []
            for ci, child in enumerate(self.root.children):
                child_T, child_labels = expand(child)
                T = np.tensordot(T, child_T, axes=([0], [0]))
                labels = labels + child_labels

        if return_labels:
            return T, labels

        # Default: require all int labels; sort by physical_axis.
        if not all(isinstance(L, (int, np.integer)) for L in labels):
            raise ValueError(
                "to_dense() requires all axes to be physical_axis ints "
                "(no processed internal nodes). Use to_dense(return_labels=True) "
                "for processed-residual case."
            )
        perm = np.argsort(labels)
        T = T.transpose(perm)
        return T

    # --------------------------------------------------------------
    # Phase 2: unfolding × dense matrix
    # --------------------------------------------------------------

    def _materialize_subtree(self, v: Node) -> Tuple[np.ndarray, List[int]]:
        """
        Partially materialize v's subtree, leaving v's parent-bond axis open.

        Returns (T_sub, leaf_axes_in_order) where:
          * T_sub.shape   = (R_v, n_below)
                             (where n_below = product of physical dims of v's
                             subtree leaves)
          * T_sub axes    = (parent-bond, flattened-physical-axes)
          * leaf_axes_in_order = list of physical_axis values of v's leaves,
                                in the order they were flattened. Caller may
                                permute back to ascending order.
        """
        if v.is_leaf or v in self.processed:
            # cores[v] shape (R_parent, n_or_compressed) — already (parent-bond, phys).
            # Use the Node itself as label for processed internal nodes (they
            # don't have a physical_axis); use physical_axis for true leaves.
            label = v.physical_axis if v.is_leaf else v
            return self.cores[v], [label]

        # Internal v (not processed): cores[v] shape (R_parent, R_c1, ..., R_cm).
        # Recursively materialize each child's subtree, contract its parent-bond
        # axis (axis 0) with the corresponding child-bond axis of v's core.
        T_sub = self.cores[v]                      # (R_parent, R_c1, ..., R_cm)
        all_leaf_axes: List = []
        for ci, child in enumerate(v.children):
            child_T, child_leaf_axes = self._materialize_subtree(child)
            #  child_T:    (R_c, n_below_child)
            #  T_sub:      (R_parent, R_{c_ci}, R_{c_{ci+1}}, ..., R_{c_{m-1}}, n_below_already_glued...)
            # Contract T_sub's child-bond axis (currently at position 1) with
            # child_T's axis 0.
            T_sub = np.tensordot(T_sub, child_T, axes=([1], [0]))
            #  After: (R_parent, remaining child-bonds..., already-glued..., n_below_child)
            all_leaf_axes = all_leaf_axes + child_leaf_axes
        # T_sub shape now: (R_parent, n_below_total) but flattened across multiple axes.
        # Flatten everything after axis 0 into one axis.
        n_below = int(np.prod(T_sub.shape[1:]))
        T_sub = T_sub.reshape(T_sub.shape[0], n_below)
        return T_sub, all_leaf_axes

    def _materialize_above(self, v: Node,
                            leaf_override: Optional[Dict[Node, np.ndarray]] = None,
                            order_only: bool = False
                            ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Partially materialize "everything except v's subtree", leaving v's
        parent-bond axis open.

        Returns (T_above, leaf_axes_in_order) where:
          * T_above.shape = (R_v, n_above)
          * T_above axes  = (parent-bond-of-v, flattened-physical-axes)
          * leaf_axes_in_order = list of physical_axis values for the leaves
                                 outside v's subtree, in flatten order.

        Parameters
        ----------
        v : Node
            Non-root node whose above-tree we materialize.
        leaf_override : dict, optional
            If given, maps a Node (an above-leaf or processed-internal that
            would otherwise be reached as a leaf-like terminal) to a 2D matrix
            of shape (R_parent_of_node, anything) used IN PLACE OF self.cores[node].
            Used by apply_unfolding_KR to substitute each above-leaf's physical
            axis with one Khatri-Rao column at a time.
            The first axis of every override must match the bond dim that
            self.cores[node] has at axis 0 (so the bond contraction works).
        """
        if v is self.root:
            raise ValueError("Root has no 'above' — its bond would be the dummy parent")

        # ----------------------------------------------------------------
        # Fast path: caller only needs the column ORDER (leaf_axes), not the
        # dense array.  Building above_V costs R_v * n_above = R_v * n^(d-1)
        # memory, which is catastrophic on wide trees (a leaf of a 6-leaf tree
        # at n=50 needs ~50 GB).  The leaf ordering is pure tree bookkeeping —
        # identical to the array recursion below but without any tensordot —
        # so we reproduce it by tracking only the running ndim and axis labels.
        # ----------------------------------------------------------------
        if order_only:
            if leaf_override is not None:
                raise ValueError("order_only=True does not support leaf_override")

            def order_stub(u: Node):
                # Returns (ndim, leaf_axes, v_bond_pos) mirroring expand_with_stub,
                # with `ndim` standing in for the array's number of axes.
                if u is v:
                    return 2, [], 1                          # eye(R_v): 2 axes, v-bond at 1
                if u.is_leaf or u in self.processed:
                    label = u.physical_axis if u.is_leaf else u
                    return self.cores[u].ndim, [label], None
                ndim = self.cores[u].ndim                     # (R_parent, R_c1, ..., R_cm)
                leaf_axes: List = []
                v_bond_pos: Optional[int] = None
                for child in u.children:
                    c_ndim, c_leaves, c_vbp = order_stub(child)
                    old_ndim = ndim
                    ndim = old_ndim + c_ndim - 2
                    if v_bond_pos is not None and v_bond_pos > 1:
                        v_bond_pos = v_bond_pos - 1
                    if c_vbp is not None:
                        v_bond_pos = old_ndim + c_vbp - 2
                    leaf_axes = leaf_axes + c_leaves
                return ndim, leaf_axes, v_bond_pos

            # Outer expansion with phantom-parent axis on the root (mirrors below).
            ndim = self.cores[self.root].ndim + 1
            leaf_axes = []
            v_bond_pos = None
            for child in self.root.children:
                c_ndim, c_leaves, c_vbp = order_stub(child)
                old_ndim = ndim
                ndim = old_ndim + c_ndim - 2
                if v_bond_pos is not None and v_bond_pos > 1:
                    v_bond_pos = v_bond_pos - 1
                if c_vbp is not None:
                    v_bond_pos = old_ndim + c_vbp - 2
                leaf_axes = leaf_axes + c_leaves
            # The final phantom-strip + moveaxis of the array path only relocates
            # the v-bond axis, never a leaf axis, so leaf_axes order is final.
            return None, leaf_axes

        # Recursive helper:
        #   expand_with_stub(u) returns (T, leaf_axes, v_bond_pos) where
        #     * T.shape[0]   = u's parent-bond
        #     * T.shape[1:]  = a mix of "remaining axes" — child-bonds we haven't
        #                      yet contracted (during the loop), v's parent-bond
        #                      stub axis if v lives in u's subtree, and physical
        #                      axes of leaves we have already glued in.
        #     * leaf_axes    = list of physical_axis values, one per "leaf" axis
        #                      in T (not including the v-bond axis, if any).
        #     * v_bond_pos   = current axis position of v's parent-bond in T,
        #                      or None if v is not in u's subtree.
        # On RETURN from a fully-processed internal node, T has no remaining
        # child-bonds: it's [R_parent_u, leaves..., (v_bond if present)] in
        # whatever order they ended up after the contractions.
        def expand_with_stub(u: Node) -> Tuple[np.ndarray, List, Optional[int]]:
            # Case 1: u IS v. Return a stub with two R_v axes:
            # axis 0 = u's parent-bond (= R_v), axis 1 = v's parent-bond (= R_v).
            if u is v:
                R_v = self.parent_bond_dim(v)
                stub = np.eye(R_v, dtype=np.float64)
                return stub, [], 1

            # Case 2: u is a leaf or PROCESSED. Core shape (R_parent, n_or_r).
            # For a processed internal node, cores[u] = identity (r, r) and we
            # treat it identically to a leaf: don't recurse into u's children.
            if u.is_leaf or u in self.processed:
                label = u.physical_axis if u.is_leaf else u
                # Use leaf_override if present: this lets the caller substitute
                # in pre-contracted leaf cores (for Khatri-Rao sketching) or
                # other modifications, without mutating self.cores.
                if leaf_override is not None and u in leaf_override:
                    return leaf_override[u], [label], None
                return self.cores[u], [label], None

            # Case 3: u is internal, not processed, != v. Core shape (R_parent, R_c1, ..., R_cm).
            T_u = self.cores[u]
            leaf_axes: List[int] = []
            v_bond_pos: Optional[int] = None

            for ci, child in enumerate(u.children):
                child_T, child_leaves, child_v_bond_pos = expand_with_stub(child)

                # Track positions across the contraction T_u <- tensordot(T_u, child_T, [1], [0]).
                # New T_u axes = (T_u axes minus axis 1) ++ (child_T axes minus axis 0).
                #   * v_bond_pos in T_u (if present): was at position p in old T_u.
                #     p > 0 (axis 0 is parent), and p != 1 (we maintain that
                #     axis 1 is always a child-bond, never a v-bond — see below).
                #     New pos: p     if p < 1 (cannot happen since p > 0),
                #              p - 1 if p > 1.
                #   * v_bond_pos in child_T (if present): was at position c in child_T.
                #     New pos in T_u: (T_u.ndim_old - 1) + (c - 1)
                #                   = T_u.ndim_old + c - 2.
                old_ndim = T_u.ndim

                T_u = np.tensordot(T_u, child_T, axes=([1], [0]))

                if v_bond_pos is not None:
                    if v_bond_pos > 1:
                        v_bond_pos = v_bond_pos - 1
                    # If v_bond_pos == 1 we'd be contracting v_bond, which we never do.

                if child_v_bond_pos is not None:
                    assert v_bond_pos is None, "v can be in only one child subtree"
                    v_bond_pos = old_ndim + child_v_bond_pos - 2

                leaf_axes = leaf_axes + child_leaves

                # Maintain the invariant "axis 1 of T_u is always a child-bond"
                # by NOT moving v_bond to axis 1 mid-loop. After contracting
                # child ci (out of m-1 in 0-indexed), the next child-bond is at
                # position 1 again automatically because we contracted axis 1
                # and the axes of T_u shift left.

            return T_u, leaf_axes, v_bond_pos

        # Outer expansion: root has no parent-bond, so we wrap with a phantom
        # singleton axis 0 to reuse the same contraction logic. After all
        # children are contracted in, we strip the phantom and move v_bond
        # to axis 0.

        T_root = self.cores[self.root]                           # (R_c1, ..., R_cm)
        T = T_root.reshape((1,) + T_root.shape)                   # phantom-parent: (1, R_c1, ..., R_cm)
        leaf_axes: List[int] = []
        v_bond_pos: Optional[int] = None

        for ci, child in enumerate(self.root.children):
            child_T, child_leaves, child_v_bond_pos = expand_with_stub(child)
            old_ndim = T.ndim
            T = np.tensordot(T, child_T, axes=([1], [0]))
            if v_bond_pos is not None and v_bond_pos > 1:
                v_bond_pos = v_bond_pos - 1
            if child_v_bond_pos is not None:
                assert v_bond_pos is None, "v can be in only one root-child subtree"
                v_bond_pos = old_ndim + child_v_bond_pos - 2
            leaf_axes = leaf_axes + child_leaves

        # T axes now: [phantom, leaves+v_bond..., ...]. Strip phantom (axis 0).
        T = T.reshape(T.shape[1:])
        # v_bond_pos was relative to the pre-strip ndim; subtract 1.
        assert v_bond_pos is not None, "v should be somewhere in the tree"
        v_bond_pos = v_bond_pos - 1

        # Move v_bond to axis 0.
        T = np.moveaxis(T, v_bond_pos, 0)
        # Note: leaf_axes ordering is preserved by moveaxis (we only moved one axis,
        # and that axis was NOT a leaf axis).

        # Flatten all non-v_bond axes.
        n_above = int(np.prod(T.shape[1:])) if T.ndim > 1 else 1
        T_above = T.reshape(T.shape[0], n_above)
        return T_above, leaf_axes

    def matricize_at(self, v: Node) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Compute the dense matricization M_v at node v WITHOUT ever forming
        the full dense tensor. Returns (M, row_leaf_axes, col_leaf_axes)
        where:
          * M.shape = (n_below, n_above)
          * row_leaf_axes = physical_axis values of leaves under v (= row index)
          * col_leaf_axes = physical_axis values of leaves outside v's subtree
                            (= col index)

        The factorization M = subtree_V @ above_V^T is exact and uses at most
        rank R_v memory (n_below × R_v on the row side, R_v × n_above on the
        col side). For very small examples (this Phase 2 test), this is fine;
        for production use we'll never form M densely (Phase 5).
        """
        if v is self.root:
            raise ValueError("Cannot matricize at root (no parent-bond)")
        subtree_V, row_axes = self._materialize_subtree(v)   # (R_v, n_below)
        above_V, col_axes = self._materialize_above(v)       # (R_v, n_above)
        M = subtree_V.T @ above_V                             # (n_below, n_above)
        return M, row_axes, col_axes

    def apply_unfolding(self, v: Node, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Compute Y = M_v @ X for node v, without forming M_v explicitly.

        The columns of X correspond to leaves outside v's subtree, in the
        ORDER returned by _materialize_above (NOT necessarily ascending
        physical-axis order). The caller must reshape/permute X so that its
        first n_above rows match this order.

        For Phase 2 test convenience we accept X with rows in any order, as
        long as the caller knows the order; we return the col-leaf-axis
        ordering alongside Y.

        Parameters
        ----------
        v : Node
            Non-root node whose unfolding we want.
        X : np.ndarray of shape (n_above, R_sketch).
            Rows must be in the ordering returned by `matricize_at(v)[2]`
            (i.e. col_leaf_axes order).

        Returns
        -------
        Y : np.ndarray of shape (n_below, R_sketch).
        row_leaf_axes : list of physical_axis values of v's subtree leaves
                        (in the row order of Y).
        """
        if v is self.root:
            raise ValueError("Cannot apply unfolding at root")
        subtree_V, row_axes = self._materialize_subtree(v)   # (R_v, n_below)
        above_V, _ = self._materialize_above(v)              # (R_v, n_above)
        # Y = (subtree_V.T @ above_V) @ X  =  subtree_V.T @ (above_V @ X)
        Y = subtree_V.T @ (above_V @ X)
        return Y, row_axes

    # --------------------------------------------------------------
    # Phase 5: structured Khatri-Rao sketches
    # --------------------------------------------------------------

    def above_leaves(self, v: Node) -> List[Node]:
        """List of leaves outside v's subtree, in physical_axis order.

        The Khatri-Rao sketch needs one factor matrix per such leaf.
        """
        if v is self.root:
            raise ValueError("Root has no above-leaves")
        v_subtree_leaves = set(_subtree_leaves(v))
        all_leaves = [u for u in all_nodes(self.root) if u.is_leaf]
        # NOTE: include processed-internals too? For QBTC's leaves-to-root sweep,
        # at the time we call apply_unfolding_KR(v), v's siblings under the same
        # parent might already be processed. Those processed-internals act like
        # leaves with "physical dim r". We treat them as above-leaves here.
        # Strictly above-NODES (leaves or processed-internals):
        above = []
        for u in all_nodes(self.root):
            if u is v:
                continue
            if u in v_subtree_leaves:
                continue
            # Skip nodes whose subtree contains v.
            if v in _subtree_leaves(u):
                continue
            # Skip nodes that have a child (= internal nodes), UNLESS they're processed
            # (in which case they look like a 2D-leaf to materialize_above's recursion).
            if u.is_leaf or u in self.processed:
                # is u "above" v? It is if u is not in v's subtree AND not v itself.
                # We've already excluded those.
                # We also need to make sure u isn't an ancestor of v (its subtree contains v).
                # For non-root u, check if u's subtree (over the original tree) contains v.
                # But _subtree_leaves(u) gives leaves in u's subtree; if v is a leaf, this
                # check is direct. For non-leaf v we'd need to check more carefully.
                # For now we use _subtree_leaves(u) which gives leaves under u; if v is
                # under u we'd see v's leaves as a subset, so is_v_under_u is equivalent
                # to: v's leaves ⊆ u's leaves. Simpler: walk u's ancestors of v.
                ancestor = v
                is_ancestor_of_v = False
                while ancestor is not None:
                    if ancestor is u:
                        is_ancestor_of_v = True
                        break
                    ancestor = ancestor.parent
                if is_ancestor_of_v:
                    continue
                above.append(u)

        # Order: by physical_axis for true leaves; processed-internals at the end
        # in some deterministic order (use name).
        true_leaves = sorted([u for u in above if u.is_leaf], key=lambda u: u.physical_axis)
        processed_internals = sorted([u for u in above if not u.is_leaf], key=lambda u: u.name)
        return true_leaves + processed_internals

    def apply_unfolding_KR(self, v: Node,
                            K_above: Dict[Node, np.ndarray]
                            ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute Y = M_v @ Omega for node v, where Omega is the Khatri-Rao
        product of small Gaussians attached to each above-leaf.

        For each above-leaf l, K_above[l] is a matrix of shape (n_l, R_sketch)
        (or (r_l, R_sketch) if l is a processed-internal). The dense Omega is
        the column-wise Kronecker product:
            Omega[:, k] = K_l_1[:, k] ⊗ K_l_2[:, k] ⊗ ... ⊗ K_l_kappa[:, k],
        in the order returned by self.above_leaves(v).

        This computes Y = M_v @ Omega WITHOUT materializing Omega densely
        (which would have shape (n_above, R_sketch), the very thing we want
        to avoid).

        Implementation (Approach A: loop over k):
            For each k = 0, ..., R_sketch - 1:
                * Substitute each above-leaf's core with cores[l] @ K_l[:, k:k+1]
                  via the leaf_override of _materialize_above.
                * Materialize the above-tree → vector of shape (R_v,).
                * Multiply by subtree_V.T → column k of Y.
            Returns Y of shape (n_below, R_sketch).

        Approach B (batched k via Hadamard on a shared sketch axis) is left for
        a future optimization; Approach A is correct and simple.

        Parameters
        ----------
        v : Node
            Non-root node whose unfolding we want.
        K_above : dict
            Maps each above-leaf Node u to a 2D matrix K_u of shape
            (cores[u].shape[1], R_sketch). All K_u must share the same
            R_sketch.

        Returns
        -------
        Y : np.ndarray of shape (n_below_v, R_sketch).
        row_leaf_axes : list of physical_axis values for v's subtree leaves.
        """
        if v is self.root:
            raise ValueError("Cannot apply unfolding at root")

        # Validate K_above.
        expected_above = self.above_leaves(v)
        missing = [u for u in expected_above if u not in K_above]
        if missing:
            raise ValueError(
                f"K_above is missing entries for: {[u.name for u in missing]}"
            )
        R_sketch_set = set(K_above[u].shape[1] for u in expected_above)
        if len(R_sketch_set) != 1:
            raise ValueError(
                f"All K_above entries must share R_sketch; got {sorted(R_sketch_set)}"
            )
        R_sketch = R_sketch_set.pop()
        # Check shapes.
        for u in expected_above:
            K_u = K_above[u]
            if K_u.ndim != 2:
                raise ValueError(f"K_above[{u.name}] must be 2D, got shape {K_u.shape}")
            n_u = self.cores[u].shape[1]
            if K_u.shape[0] != n_u:
                raise ValueError(
                    f"K_above[{u.name}].shape[0]={K_u.shape[0]} doesn't match "
                    f"physical dim {n_u} of cores[{u.name}]"
                )

        # Compute subtree_V (no override needed on v's subtree side).
        subtree_V, row_axes = self._materialize_subtree(v)   # (R_v, n_below)
        n_below = subtree_V.shape[1]
        Y = np.zeros((n_below, R_sketch), dtype=np.float64)

        # Loop over k columns. For each k, build leaf_override and materialize.
        for k in range(R_sketch):
            override: Dict[Node, np.ndarray] = {}
            for u in expected_above:
                # cores[u]: (R_parent, n_u). K_u[:, k:k+1]: (n_u, 1).
                # Override: cores[u] @ K_u[:, k:k+1]:  (R_parent, 1).
                override[u] = self.cores[u] @ K_above[u][:, k:k+1]

            T_above_k, _ = self._materialize_above(v, leaf_override=override)
            # T_above_k.shape = (R_v, 1*1*...*1) = (R_v, 1).
            assert T_above_k.shape[1] == 1, f"Expected (R_v, 1), got {T_above_k.shape}"

            # Y[:, k] = subtree_V.T @ T_above_k.flatten() (a length-n_below vector).
            Y[:, k] = (subtree_V.T @ T_above_k).flatten()

        return Y, row_axes

    # --------------------------------------------------------------
    # Phase 5: structured TreeStack sketch (open-leaf, mirror topology)
    # --------------------------------------------------------------

    def _expand_TS(self, u: Node, v: Node,
                    sketch_cores: Dict[Node, np.ndarray]
                    ) -> Tuple[np.ndarray, bool]:
        """
        Walk the above-v sketch tree contracted against the TTN at node u.

        Mirror-topology TreeStack: the sketch tree has the SAME shape as the
        TTN's above-v structure. At every node u that is not in v's subtree
        (or is v itself), we have a sketch tensor G_u = sketch_cores[u].

        Returns
        -------
        T_u : np.ndarray with axes:
            * (R_TTN_u_parent, R_sketch_u_parent)                          if v not in u's subtree
            * (R_TTN_u_parent, R_sketch_u_parent, R_TTN_v_open, R_phi)     if v in u's subtree
        has_v_axes : bool
            True iff v is in u's subtree (in which case T_u carries the two
            extra "open" axes).

        For the recursion to work uniformly for the root (which has no
        parent-bonds), this method handles only NON-ROOT nodes. The root is
        unwrapped by apply_unfolding_TS itself.
        """
        # Case 1: u IS v. Active leaf in the sketch tree.
        if u is v:
            # Sketch core G_v has shape (R_sketch_v_parent, R_phi).
            G_v = sketch_cores[v]
            if G_v.ndim != 2:
                raise ValueError(
                    f"sketch_cores[{v.name}] (active leaf) must be 2D "
                    f"(R_sketch_parent, R_phi), got shape {G_v.shape}"
                )
            R_TTN_v = self.parent_bond_dim(v)
            # T_u[a, b, c, d] = δ(a, c) * G_v[b, d]
            #   axis 0 = R_TTN_u_parent  (= R_TTN_v)
            #   axis 1 = R_sketch_u_parent (= G_v.shape[0])
            #   axis 2 = R_TTN_v_open      (same dim as axis 0)
            #   axis 3 = R_phi
            I = np.eye(R_TTN_v)
            T_u = np.einsum('ac, bd -> abcd', I, G_v)
            return T_u, True

        # Case 2: u is a leaf (true or processed-internal). No recursion below.
        if u.is_leaf or u in self.processed:
            TTN_core = self.cores[u]      # (R_TTN_u_parent, n_u)
            G_u = sketch_cores[u]         # (R_sketch_u_parent, n_u)
            if G_u.ndim != 2 or G_u.shape[1] != TTN_core.shape[1]:
                raise ValueError(
                    f"sketch_cores[{u.name}] must be (R_sketch, {TTN_core.shape[1]}), "
                    f"got shape {G_u.shape}"
                )
            # Contract over the n_u axis (axis 1 of both).
            merged = TTN_core @ G_u.T     # (R_TTN_u_parent, R_sketch_u_parent)
            return merged, False

        # Case 3: u is internal, not v, not processed.
        TTN_core = self.cores[u]    # (R_TTN_u_parent, R_TTN_c1, ..., R_TTN_cm)
        G_u = sketch_cores[u]       # (R_sketch_u_parent, R_sketch_c1, ..., R_sketch_cm)
        m = len(u.children)
        if TTN_core.ndim != 1 + m or G_u.ndim != 1 + m:
            raise ValueError(
                f"Internal node {u.name}: TTN_core shape {TTN_core.shape}, "
                f"G_u shape {G_u.shape}, expected ndim {1+m} for both"
            )

        # Recurse into children.
        children_T = []
        children_has_v = []
        for child in u.children:
            T_c, has_v_c = self._expand_TS(child, v, sketch_cores)
            children_T.append(T_c)
            children_has_v.append(has_v_c)
        has_v_axes = any(children_has_v)
        if sum(children_has_v) > 1:
            raise RuntimeError("v cannot be in two children's subtrees")

        # Build einsum with named indices:
        #   'a' = R_TTN_u_parent,    'A' = R_sketch_u_parent
        #   'b','c','d',... = R_TTN_ci  (lowercase 'b' onward, one per child)
        #   'B','C','D',... = R_sketch_ci
        #   'p', 'P' = R_TTN_v_open, R_phi (only if has_v_axes)
        if m > 24:
            raise NotImplementedError("More than 24 children per internal node not supported.")

        TTN_str = 'a' + ''.join(chr(ord('b') + i) for i in range(m))
        sketch_str = 'A' + ''.join(chr(ord('B') + i) for i in range(m))
        children_strs = []
        for i, has_v_c in enumerate(children_has_v):
            s = chr(ord('b') + i) + chr(ord('B') + i)
            if has_v_c:
                s += 'pP'
            children_strs.append(s)
        out_str = 'aA' + ('pP' if has_v_axes else '')
        einsum_str = f"{TTN_str},{sketch_str}," + ",".join(children_strs) + f"->{out_str}"

        T_u = contract_network(einsum_str, TTN_core, G_u, *children_T)
        return T_u, has_v_axes

    def apply_unfolding_TS(self, v: Node,
                            sketch_cores: Dict[Node, np.ndarray]
                            ) -> Tuple[np.ndarray, List[int]]:
        """
        TreeStack sketch (open-leaf, mirror topology) applied to M_v's column space.

        The sketch tree shares topology with the input TTN. For each TTN node u
        OUTSIDE v's subtree (and for v itself), sketch_cores[u] is the random
        Gaussian tensor at u with shapes:
            * v (active leaf):        (R_sketch_v_parent, R_phi).
            * Above-leaf or processed-internal u: (R_sketch_u_parent, n_u_or_r).
            * Above-internal u (non-root): (R_sketch_u_parent, R_sketch_c1, ..., R_sketch_cm).
            * Root:                   (R_sketch_c1, ..., R_sketch_cm).

        Returns Y = M_v @ Omega^T where Omega is the contracted sketch tree
        (Omega.shape = (n_above, R_phi)), of shape (n_below, R_phi).

        The contraction is leaf-by-leaf (avoiding n_above): at each above-leaf
        we merge TTN core × sketch leaf core over n_u, getting (R_TTN, R_sketch);
        we then walk up, contracting both TTN and sketch cores with the merged
        children. Memory cost per intermediate: O(R_TTN × R_sketch).

        Parameters
        ----------
        v : Node
            Non-root node whose unfolding we want.
        sketch_cores : dict
            Maps each above-tree node (true above-leaf, processed-internal,
            above-internal, root) AND v to its sketch Gaussian tensor.
            Use `build_treestack_cores(self, v, R, rng)` to construct one.

        Returns
        -------
        Y : np.ndarray of shape (n_below_v, R_phi).
        row_leaf_axes : list of physical_axis values for v's subtree leaves.
        """
        if v is self.root:
            raise ValueError("Cannot apply unfolding at root")

        # Validate that sketch_cores has entries for all required nodes.
        required = self._required_sketch_nodes(v)
        missing = [u for u in required if u not in sketch_cores]
        if missing:
            raise ValueError(
                f"sketch_cores is missing entries for: {[u.name for u in missing]}"
            )

        # Materialize subtree_V (cheap; bounded by R_v × n_below).
        subtree_V, row_axes = self._materialize_subtree(v)   # (R_TTN_v, n_below)
        n_below = subtree_V.shape[1]

        # Walk the above-v structure starting from the root.
        # Root is a special case: it has no parent-bond in either TTN or sketch.
        TTN_root = self.cores[self.root]      # (R_TTN_c1, ..., R_TTN_cm)
        G_root = sketch_cores[self.root]      # (R_sketch_c1, ..., R_sketch_cm)
        m = len(self.root.children)
        if TTN_root.ndim != m or G_root.ndim != m:
            raise ValueError(
                f"Root: TTN_core ndim {TTN_root.ndim}, G ndim {G_root.ndim}, "
                f"expected {m}"
            )

        children_T = []
        children_has_v = []
        for child in self.root.children:
            T_c, has_v_c = self._expand_TS(child, v, sketch_cores)
            children_T.append(T_c)
            children_has_v.append(has_v_c)
        has_v_axes = any(children_has_v)
        assert has_v_axes, "v must be in some root-child's subtree"
        if sum(children_has_v) > 1:
            raise RuntimeError("v cannot be in two root-children's subtrees")

        # Einsum for the root: like internal but no parent axes.
        if m > 24:
            raise NotImplementedError("More than 24 children at root not supported.")
        TTN_str = ''.join(chr(ord('b') + i) for i in range(m))
        sketch_str = ''.join(chr(ord('B') + i) for i in range(m))
        children_strs = []
        for i, has_v_c in enumerate(children_has_v):
            s = chr(ord('b') + i) + chr(ord('B') + i)
            if has_v_c:
                s += 'pP'
            children_strs.append(s)
        out_str = 'pP'
        einsum_str = f"{TTN_str},{sketch_str}," + ",".join(children_strs) + f"->{out_str}"

        above_dot_omega = contract_network(einsum_str, TTN_root, G_root, *children_T)
        # above_dot_omega.shape = (R_TTN_v, R_phi)

        # Y = subtree_V.T @ above_dot_omega = (n_below, R_phi).
        Y = subtree_V.T @ above_dot_omega
        return Y, row_axes

    # --------------------------------------------------------------
    # Phase 7: structured qb_svd via the Gram matrix G = above_V @ above_V.T
    # --------------------------------------------------------------

    def _expand_gram(self, u: Node, v: Node) -> Tuple[Optional[np.ndarray], bool]:
        """
        Walk the above-v tree, contracting two copies of the TTN against each
        other. Same recursion shape as `_expand_TS`, but the second factor at
        every node is another copy of the TTN core (with primed indices) rather
        than a sketch core. The leaf-level merge becomes
            leaf_gram[u] = cores[u] @ cores[u].T,
        of shape (R_TTN_u_parent, R_TTN_u_parent_prime).

        Returns
        -------
        gram_u : np.ndarray with axes:
            * (R_TTN_u_parent, R_TTN_u_parent_prime)                           if v not in u's subtree
            * (R_TTN_u_parent, R_TTN_u_parent_prime, R_TTN_v_open, R_TTN_v_open_prime)
                                                                                if v in u's subtree
            * `None` SENTINEL if u IS v itself.  In that case the "gram" is
              implicitly δ(b_v, p) δ(B_v, P) (4-axis identity).  Materializing
              this would be O(R_TTN_v^4) and can blow memory on moderate trees.
              Instead, the caller (v's parent) detects the None and rewrites
              its own einsum to identify b_v↔p and B_v↔P directly, skipping
              the spurious product entirely.
        has_v_axes : bool, True iff v is in u's subtree.
        """
        # Case 1: u IS v.  Skip the 4-axis identity materialization.
        # Caller will treat None as "identify the open axes with TTN_core's
        # child-axis-for-v" via einsum index renaming.
        if u is v:
            return None, True

        # Case 2: u is a leaf or processed-internal. Leaf-level Gram.
        if u.is_leaf or u in self.processed:
            TTN_core = self.cores[u]                     # (R_TTN_p, n_u_or_r)
            leaf_gram = TTN_core @ TTN_core.T            # (R_TTN_p, R_TTN_p)
            return leaf_gram, False

        # Case 3: u is internal, not v, not processed. Two copies of the TTN
        # core meet here.
        TTN_core = self.cores[u]                         # (R_TTN_p, R_TTN_c1, ..., R_TTN_cm)
        m = len(u.children)

        children_grams = []
        children_has_v = []
        for child in u.children:
            gram_c, has_v_c = self._expand_gram(child, v)
            children_grams.append(gram_c)
            children_has_v.append(has_v_c)
        has_v_axes = any(children_has_v)
        if sum(children_has_v) > 1:
            raise RuntimeError("v cannot be in two children's subtrees")

        if m > 24:
            raise NotImplementedError("More than 24 children per internal node not supported.")

        # Index plan:
        #   'a' = R_TTN_u_parent,   'A' = R_TTN_u_parent_prime
        #   'b','c','d',... = R_TTN_ci  ;  'B','C','D',... = R_TTN_ci_prime
        #   'p', 'P' = R_TTN_v_open, R_TTN_v_open_prime  (only if has_v_axes)
        # Children with `gram_c is None` (== this child IS v) cause us to
        # rename their slot in TTN_core / TTN_prime from b_i/B_i to p/P.
        child_letters = [chr(ord('b') + i) for i in range(m)]   # b, c, d, ...
        child_letters_prime = [chr(ord('B') + i) for i in range(m)]
        ttn_chars = ['a']
        ttn_prime_chars = ['A']
        operands_strs = []
        operands_arrays = []
        for i, gram_c in enumerate(children_grams):
            if gram_c is None:
                # Child i IS v.  Rename b_i -> p in TTN_core and B_i -> P in TTN_prime.
                ttn_chars.append('p')
                ttn_prime_chars.append('P')
                # No operand for this child: the implicit identities are
                # absorbed into the index renaming.
            else:
                bi = child_letters[i]
                Bi = child_letters_prime[i]
                ttn_chars.append(bi)
                ttn_prime_chars.append(Bi)
                # Build this child's operand string from gram_c.ndim.
                if gram_c.ndim == 2:
                    operands_strs.append(bi + Bi)
                else:
                    # Has v deeper inside.  Axes are (bi, Bi, p, P).
                    assert gram_c.ndim == 4
                    operands_strs.append(bi + Bi + 'pP')
                operands_arrays.append(gram_c)

        TTN_str = ''.join(ttn_chars)
        TTN_prime_str = ''.join(ttn_prime_chars)
        out_str = 'aA' + ('pP' if has_v_axes else '')
        einsum_parts = [TTN_str, TTN_prime_str] + operands_strs
        einsum_str = ",".join(einsum_parts) + f"->{out_str}"

        gram_u = contract_network(einsum_str, TTN_core, TTN_core, *operands_arrays)
        return gram_u, has_v_axes

    def compute_above_gram(self, v: Node) -> np.ndarray:
        """
        Compute the Gram matrix G = above_V @ above_V.T of shape
        (R_TTN_v, R_TTN_v) WITHOUT materializing above_V or n_above.

        The Gram matrix is exactly the "double tree contraction" of the
        above-v structure: at every above-leaf u, replace the n_u dimension
        with a contracted (cores[u] @ cores[u].T); at every above-internal,
        contract two copies of the TTN core, in lockstep, against the merged
        children's grams. At v we leave R_TTN_v open on both sides.

        Used by the structured qb_svd finish:
            B B.T = A G A.T,   where  A = Q^T @ subtree_V.T  (small).
        Memory cost per intermediate: O(R_TTN^2) — bounded by bond dims, never
        n_above.
        """
        if v is self.root:
            raise ValueError("Root has no above")

        # Walk from the root. Root has no parent-bond.
        TTN_root = self.cores[self.root]    # (R_TTN_c1, ..., R_TTN_cm)
        m = len(self.root.children)

        children_grams = []
        children_has_v = []
        for child in self.root.children:
            gram_c, has_v_c = self._expand_gram(child, v)
            children_grams.append(gram_c)
            children_has_v.append(has_v_c)
        has_v_axes = any(children_has_v)
        assert has_v_axes, "v must be in some root-child's subtree"
        if sum(children_has_v) > 1:
            raise RuntimeError("v cannot be in two root-children's subtrees")

        if m > 24:
            raise NotImplementedError("More than 24 children at root not supported.")

        # Same `None`-sentinel logic as `_expand_gram`'s internal branch:
        # if a root child IS v, skip its (would-be 4-axis identity) operand
        # and rename b_v→p, B_v→P in TTN_root and TTN_root_prime directly.
        child_letters = [chr(ord('b') + i) for i in range(m)]
        child_letters_prime = [chr(ord('B') + i) for i in range(m)]
        ttn_chars = []
        ttn_prime_chars = []
        operands_strs = []
        operands_arrays = []
        for i, gram_c in enumerate(children_grams):
            if gram_c is None:
                ttn_chars.append('p')
                ttn_prime_chars.append('P')
            else:
                bi = child_letters[i]
                Bi = child_letters_prime[i]
                ttn_chars.append(bi)
                ttn_prime_chars.append(Bi)
                if gram_c.ndim == 2:
                    operands_strs.append(bi + Bi)
                else:
                    assert gram_c.ndim == 4
                    operands_strs.append(bi + Bi + 'pP')
                operands_arrays.append(gram_c)

        TTN_str = ''.join(ttn_chars)
        TTN_prime_str = ''.join(ttn_prime_chars)
        out_str = 'pP'
        einsum_parts = [TTN_str, TTN_prime_str] + operands_strs
        einsum_str = ",".join(einsum_parts) + f"->{out_str}"

        G = contract_network(einsum_str, TTN_root, TTN_root, *operands_arrays)
        # G.shape = (R_TTN_v, R_TTN_v)
        return G

    def _required_sketch_nodes(self, v: Node) -> List[Node]:
        """List of nodes that need a sketch_cores entry for TreeStack at v.

        These are: v itself, plus every node OUTSIDE v's subtree (so that
        the recursion has a sketch tensor at every node it visits).
        """
        # v's subtree (= v and its descendants).
        v_subtree = set(_post_order(v))
        out = []
        for u in all_nodes(self.root):
            if u is v:
                out.append(u)
            elif u in v_subtree:
                continue
            else:
                out.append(u)
        return out

    def build_treestack_cores(self, v: Node, R: int,
                                rng: np.random.Generator) -> Dict[Node, np.ndarray]:
        """
        Generate a fresh single-instance TreeStack sketch (mirror topology) for
        use at TTN node v. Each entry of every G_u is i.i.d. N(0, 1/R).

        Returns sketch_cores: dict mapping each required node u (v and all above-v
        nodes) to its random Gaussian tensor with the correct shape per role.
        """
        sigma = 1.0 / np.sqrt(R)
        sketch_cores: Dict[Node, np.ndarray] = {}
        for u in self._required_sketch_nodes(v):
            if u is v:
                # Active leaf: (R_sketch_v_parent, R_phi). Both = R.
                sketch_cores[u] = rng.standard_normal((R, R)) * sigma
            elif u.is_leaf or u in self.processed:
                # Non-active leaf (true or processed-internal): (R, n_u_or_r).
                n_u = self.cores[u].shape[1]
                sketch_cores[u] = rng.standard_normal((R, n_u)) * sigma
            elif u is self.root:
                # Root: (R, R, ..., R) with m_root axes.
                m = len(u.children)
                sketch_cores[u] = rng.standard_normal((R,) * m) * sigma
            else:
                # Non-root internal: (R, R, ..., R) with 1+m axes.
                m = len(u.children)
                sketch_cores[u] = rng.standard_normal((R,) * (1 + m)) * sigma
        return sketch_cores

    def build_ttstack_omega(self, v: Node, R: int,
                              rng: np.random.Generator) -> Tuple[np.ndarray, List]:
        """
        Build a dense Omega of shape (n_above, R) for the TTStack sketch
        (caterpillar topology) on the above-v leaves of `self`.

        Caterpillar structure (Cazeaux-Dupuy-Justiniano):

            active(v)   L_0    L_1    ...    L_{k-2}   L_{k-1}
                \\        |      |              |          |
                 I_1 -- I_2 -- I_3 -- ... -- I_{k-1} --- I_k(root)

        where:
          * active is the active leaf (G shape (R, R), the φ axis is the second axis).
          * L_j is the j-th above-leaf (G shape (R, n_j)).
          * I_j (1 ≤ j < k) is internal (G shape (R, R, R)): parent-bond + 2 children.
          * I_k is the sketch root (G shape (R, R)): 2 children, no parent-bond.

        The leaves L_j are taken in the order returned by self._materialize_above(v)
        (= order of the col_axes of M_v), so that the resulting Omega has rows in
        the order expected by self.apply_unfolding(v, Omega).

        All Gaussians have N(0, 1/R) entries.

        Returns
        -------
        Omega : np.ndarray of shape (n_above, R).
        col_axes : list of physical_axis values (ints) and/or processed-Node markers,
                   matching the col-axis order of self._materialize_above(v).
        """
        if v is self.root:
            raise ValueError("Cannot sketch at root")

        sigma = 1.0 / np.sqrt(R)

        # Identify the above-leaves in _materialize_above's col-axis order.
        _, col_axes = self._materialize_above(v, order_only=True)
        # Map labels back to Nodes:
        label_to_node: Dict = {}
        for u in all_nodes(self.root):
            if u.is_leaf:
                label_to_node[u.physical_axis] = u
            elif u in self.processed:
                label_to_node[u] = u
        above_leaves_in_order = [label_to_node[L] for L in col_axes]
        k = len(above_leaves_in_order)

        if k == 0:
            return np.eye(1), col_axes

        # Random tensors:
        G_active = rng.standard_normal((R, R)) * sigma                        # (parent_to_I1, phi)
        G_leaves = [rng.standard_normal((R, self.cores[u].shape[1])) * sigma  # (parent_to_I_{j+1}, n_j)
                    for u in above_leaves_in_order]
        if k >= 2:
            G_internals = [rng.standard_normal((R, R, R)) * sigma             # (parent_up, child_chain, child_leaf)
                            for _ in range(k - 1)]
            G_root = rng.standard_normal((R, R)) * sigma                      # (child_chain, child_leaf)
        else:
            G_internals = []
            G_root = rng.standard_normal((R, R)) * sigma                      # (child_chain, child_leaf)

        # Build the caterpillar contraction.
        state = G_active   # axes: [chain_bond_up, phi]  shape (R, R)

        if k == 1:
            # I_1 = root with two children: active and L_0.
            # state[i, P] * G_root[i, j] * G_leaves[0][j, n0] -> Pn0
            state = np.einsum('iP, ij, jn -> Pn', state, G_root, G_leaves[0])
            Omega = state.T   # (n_0, R_phi)
        else:
            # k >= 2: process j = 0..k-2 with G_internals[j], then root combines L_{k-1}.
            for j in range(k - 1):
                G_int = G_internals[j]   # (parent_up, child_chain, child_leaf)
                G_leaf = G_leaves[j]     # (parent_to_I_{j+1}, n_j)
                # state axes: (chain_bond, phi, n_0, ..., n_{j-1}) — j existing leaf axes.
                existing_leaf_chars = ''.join(chr(ord('d') + ee) for ee in range(j))
                state_str = 'i' + 'P' + existing_leaf_chars
                G_int_str = 'u' + 'i' + 'c'
                G_leaf_str = 'c' + 'n'
                out_str = 'u' + 'P' + existing_leaf_chars + 'n'
                ein_str = f"{state_str},{G_int_str},{G_leaf_str}->{out_str}"
                state = np.einsum(ein_str, state, G_int, G_leaf)
                # state axes now: (chain_bond_up, phi, n_0, ..., n_j)

            # Final step: root. G_root: (R, R) — (child_chain, child_leaf).
            existing_leaf_chars = ''.join(chr(ord('d') + ee) for ee in range(k - 1))
            state_str = 'i' + 'P' + existing_leaf_chars
            G_root_str = 'i' + 'j'
            G_leaf_str = 'j' + 'n'
            out_str = 'P' + existing_leaf_chars + 'n'
            ein_str = f"{state_str},{G_root_str},{G_leaf_str}->{out_str}"
            state = np.einsum(ein_str, state, G_root, G_leaves[k - 1])
            # state shape: (R_phi, n_0, n_1, ..., n_{k-1})

            n_above = int(np.prod(state.shape[1:]))
            Omega_T = state.reshape(state.shape[0], n_above)   # (R_phi, n_above)
            Omega = Omega_T.T   # (n_above, R_phi)

        return Omega, col_axes

    def apply_unfolding_TT_struct(self, v: Node, R: int,
                                    rng: np.random.Generator
                                    ) -> Tuple[np.ndarray, List]:
        """
        Structured TTStack (caterpillar / TT path-graph) sketch contracted
        against the TTN's above-v tree, WITHOUT materialising n_above.

        The above-network -- the above-v data cores plus the caterpillar sketch
        cores, glued along their shared physical legs -- is assembled as ONE
        einsum whose contraction order is chosen by numpy.  Because the
        caterpillar visits the above-leaves in the tree's DFS order, that order
        crosses each tree edge only a bounded number of times, so a near-optimal
        contraction keeps a frontier of order R_TTN * R^2 instead of the
        R_TTN^(#above-leaves) the legacy leaf-by-leaf accumulation produced.

        Returns the IDENTICAL map to apply_unfolding_TT_struct_legacy for the
        same random draws (the preamble below draws the caterpillar tensors in
        the same order and shapes), but with a far smaller peak intermediate.

        Parameters / Returns: as apply_unfolding_TT_struct_legacy.
        """
        if v is self.root:
            raise ValueError("Cannot apply unfolding at root")

        # --- preamble: draws identical to the legacy routine ---
        _, col_axes = self._materialize_above(v, order_only=True)
        label_to_node: Dict = {}
        for u in all_nodes(self.root):
            if u.is_leaf:
                label_to_node[u.physical_axis] = u
            elif u in self.processed:
                label_to_node[u] = u
        above_leaves_in_order = [label_to_node[L] for L in col_axes]
        k = len(above_leaves_in_order)
        if k == 0:
            raise NotImplementedError(
                "Structured TTStack with no above-leaves not implemented (degenerate)."
            )
        sigma = 1.0 / np.sqrt(R)
        G_active = rng.standard_normal((R, R)) * sigma
        G_leaves = [rng.standard_normal((R, self.cores[u].shape[1])) * sigma
                     for u in above_leaves_in_order]
        if k >= 2:
            G_internals_list = [rng.standard_normal((R, R, R)) * sigma
                                 for _ in range(k - 1)]
        else:
            G_internals_list = []
        G_root_cat = rng.standard_normal((R, R)) * sigma

        # --- assemble the network with integer bond labels (einsum sublist form) ---
        _ctr = [0]
        def _bond():
            _ctr[0] += 1
            return _ctr[0] - 1

        PHI = _bond()                       # sketch output (R_phi)
        VOPEN = _bond()                     # open data bond parent(v)->v (R_TTN_v)
        S = [_bond() for _ in range(k)]     # caterpillar spine bonds s_0..s_{k-1}
        E = [_bond() for _ in range(k)]     # leaf-chain bonds  e_0..e_{k-1}

        leaf_index = {u: j for j, u in enumerate(above_leaves_in_order)}

        # Data-bond id for every above non-root node (and v -> VOPEN), top-down.
        dbid: Dict = {v: VOPEN}
        def _assign(u):
            for child in u.children:
                if child is v:
                    continue                     # already VOPEN
                dbid[child] = _bond()
                if child not in leaf_index:       # above-internal -> recurse
                    _assign(child)
        _assign(self.root)

        operands: List = []

        # caterpillar spine + per-leaf physical fold (cheap: contracts n_j only).
        operands += [G_active, [S[0], PHI]]
        for j, u in enumerate(above_leaves_in_order):
            merged = self.cores[u] @ G_leaves[j].T          # (db_parent, e_j)
            operands += [merged, [dbid[u], E[j]]]
        for j in range(k - 1):
            operands += [G_internals_list[j], [S[j + 1], S[j], E[j]]]   # (d, c, e)
        operands += [G_root_cat, [S[k - 1], E[k - 1]]]                  # (c, e)

        # above-v data cores: root + above-internal nodes.
        def _add_cores(u):
            child_ids = [dbid[child] for child in u.children]
            core = self.cores[u]
            if u is self.root:
                operands.append(core); operands.append(list(child_ids))
            else:
                operands.append(core); operands.append([dbid[u]] + child_ids)
            for child in u.children:
                if child is not v and child not in leaf_index:
                    _add_cores(child)
        _add_cores(self.root)

        operands.append([VOPEN, PHI])       # output sublist -> (R_TTN_v, R_phi)
        # Contract with a flop-minimizing path (see contract_network).  A plain
        # np.einsum(optimize="greedy") here picks a memory-cheap but flop-heavy
        # order on branching trees (e.g. a balanced binary tree), turning this
        # bond-bounded contraction into a multi-hour one.
        above_dot_omega = contract_network(*operands)                # (R_TTN_v, R_phi)

        subtree_V, row_axes = self._materialize_subtree(v)           # (R_TTN_v, n_below)
        Y = subtree_V.T @ above_dot_omega                            # (n_below, R_phi)
        return Y, row_axes

    def apply_unfolding_TT_struct_legacy(self, v: Node, R: int,
                                    rng: np.random.Generator
                                    ) -> Tuple[np.ndarray, List]:
        """
        [LEGACY reference -- superseded by apply_unfolding_TT_struct.  Kept only
        to cross-check the optimal-order rewrite.  This hand-rolled DFS+chain
        walk accumulates one R_TTN axis per open sibling subtree on its frontier,
        giving a peak intermediate of order R_TTN^(#above-leaves) -- e.g.
        R_phi*R_TTN^5 on the 6-leaf Figure-1 tree -- which OOMs at large R_TTN.
        apply_unfolding_TT_struct returns the identical map far more cheaply.]

        Structured TTStack sketch: caterpillar (TT path-graph) sketch over
        the above-v leaves, contracted IN PLACE against the TTN's above-v tree
        without ever materializing n_above or the dense Omega.

        Strategy: interleave a DFS of the TTN above-v tree with the
        caterpillar's chain advancement. Each time the DFS encounters an
        above-leaf, we advance the caterpillar chain by one step (using one
        of the chain's I_j tensors). When the DFS pops up to a TTN internal,
        we contract that internal's TTN core with the accumulated R_TTN_p
        axes from its children. v's stub is encountered somewhere in the DFS
        and contributes the R_TTN_v_open axis without advancing the chain.

        The order in which above-leaves are visited by the DFS exactly
        matches the caterpillar's leaf order (both follow
        `_materialize_above`'s col_axes), so the chain can be advanced
        leaf-by-leaf in caterpillar order with no reordering.

        State invariant during the walk:
            state.shape = (R_chain_head, R_phi, [R_TTN_v_open], *R_TTN_p_LIFO)
        i.e. axis 0 = current chain bond going up, axis 1 = R_phi (always
        present), axis 2 = R_TTN_v_open (present iff v has been visited),
        and the trailing axes are the most-recently-accumulated R_TTN_p
        axes from leaves and popped-up internals (in LIFO/append order).

        After the LAST leaf, the chain is consumed by G_root_cat — axis 0
        (R_chain_head) goes away. After the root TTN core is contracted,
        all R_TTN_p axes go away. The result has shape (R_phi, R_TTN_v_open).

        Parameters
        ----------
        v : Node
            Non-root node whose unfolding we want.
        R : int
            Per-copy sketch dimension (= R_phi = R_chain).
        rng : np.random.Generator

        Returns
        -------
        Y : np.ndarray of shape (n_below_v, R).
        row_leaf_axes : list of physical_axis labels for v's subtree leaves.
        """
        if v is self.root:
            raise ValueError("Cannot apply unfolding at root")

        # Determine above-leaves in caterpillar = DFS order.
        _, col_axes = self._materialize_above(v, order_only=True)
        label_to_node: Dict = {}
        for u in all_nodes(self.root):
            if u.is_leaf:
                label_to_node[u.physical_axis] = u
            elif u in self.processed:
                label_to_node[u] = u
        above_leaves_in_order = [label_to_node[L] for L in col_axes]
        k = len(above_leaves_in_order)

        if k == 0:
            raise NotImplementedError(
                "Structured TTStack with no above-leaves not implemented (degenerate)."
            )

        sigma = 1.0 / np.sqrt(R)

        # Random caterpillar tensors.
        G_active = rng.standard_normal((R, R)) * sigma          # (chain_to_I_1, R_phi)
        G_leaves = [rng.standard_normal((R, self.cores[u].shape[1])) * sigma
                     for u in above_leaves_in_order]            # each (chain_into_I_{j+1}, n_j)
        # G_internals_list[j] = caterpillar internal I_{j+1} for j=0..k-2 (none if k==1).
        # G_root_cat = G_root for the caterpillar (always present when k>=1).
        if k >= 2:
            G_internals_list = [rng.standard_normal((R, R, R)) * sigma for _ in range(k - 1)]
        else:
            G_internals_list = []
        G_root_cat = rng.standard_normal((R, R)) * sigma         # (chain_in, leaf_chain_in)

        # Mutable state and leaf counter (closure).
        state = G_active        # initial: (R_chain_head, R_phi). 0 R_TTN_p axes, no V_o yet.
        leaf_idx_holder = [0]   # mutable int (closure trick)
        v_visited = [False]

        def _ttn_axes_chars(N: int, exclude_v_o: bool, used_singles: str) -> List[str]:
            """Return single-char labels for the trailing R_TTN axes of state.

            Number of TTN axes = N - 2 if no V_o, else N - 3. They use chars
            outside `used_singles`. We always pick from a-z avoiding clashes.
            """
            n_ttn = N - 2 - (1 if not exclude_v_o and v_visited[0] else 0)
            if n_ttn <= 0:
                return []
            chars = []
            cand = 'abdefghijklmnoqrstuvwxyz'   # avoid 'c','p','t','V','O' used elsewhere
            for ch in cand:
                if ch in used_singles:
                    continue
                chars.append(ch)
                if len(chars) == n_ttn:
                    return chars
            raise NotImplementedError("Too many R_TTN axes for single-char einsum.")

        def expand(u: Node) -> None:
            nonlocal state
            # Case 1: u IS v. Append R_TTN_v_open axis (at fixed position 2 from
            # the front, i.e. just after R_phi if chain present, or just after
            # R_phi if chain consumed), AND R_TTN_p_v at the end (LIFO).
            # Use identity to link them.
            if u is v:
                R_TTN_v = self.parent_bond_dim(v)
                I = np.eye(R_TTN_v)
                N = state.ndim
                # CHAIN PRESENT iff at least one non-v leaf hasn't been
                # processed yet AND state still carries the chain axis at 0.
                # Concretely: leaf_idx_holder[0] < k means more leaves remain
                # to advance the chain, so state still has chain at axis 0.
                # If leaf_idx_holder[0] == k, the last non-v leaf has run via
                # G_root_cat and consumed the chain — state shape starts with
                # (R_phi, ...) without R_chain.
                chain_present = leaf_idx_holder[0] < k
                if chain_present:
                    # state shape: (R_chain, R_phi, *R_TTN_p_LIFO).
                    ttn_chars = _ttn_axes_chars(N, exclude_v_o=True, used_singles='cpoV')
                    state_str = 'cp' + ''.join(ttn_chars)
                    I_str = 'oV'
                    out_str = 'cpo' + ''.join(ttn_chars) + 'V'
                else:
                    # state shape: (R_phi, *R_TTN_p_LIFO).  No 'c'.
                    n_ttn = N - 1
                    cand = 'abdefghijklmnoqrstuvwxyz'
                    used = set('poV')
                    chars = [c for c in cand if c not in used][:n_ttn]
                    if len(chars) < n_ttn:
                        raise NotImplementedError(
                            "Too many R_TTN axes for single-char einsum (chain-consumed v stub)."
                        )
                    ttn_chars = chars
                    state_str = 'p' + ''.join(ttn_chars)
                    I_str = 'oV'
                    out_str = 'po' + ''.join(ttn_chars) + 'V'
                state = np.einsum(f'{state_str},{I_str}->{out_str}', state, I)
                v_visited[0] = True
                return

            # Case 2: u is an above-leaf (true or processed-internal). Advance chain.
            if u.is_leaf or u in self.processed:
                j = leaf_idx_holder[0]
                leaf_idx_holder[0] += 1
                G_leaf = G_leaves[j]                              # (R_chain_in, n_u)
                merged_leaf = self.cores[u] @ G_leaf.T            # (R_TTN_p_u, R_chain_in)

                N = state.ndim
                ttn_chars = _ttn_axes_chars(N, exclude_v_o=False, used_singles='cpdteoV')
                state_str = 'cp' + ('o' if v_visited[0] else '') + ''.join(ttn_chars)

                if j < k - 1:
                    # Internal caterpillar node I_{j+1}: shape (R_chain_up, R_chain_in, R_chain_in_leaf).
                    G_int = G_internals_list[j]
                    G_str = 'dce'
                    merged_str = 'te'
                    out_str = 'dp' + ('o' if v_visited[0] else '') + ''.join(ttn_chars) + 't'
                    state = np.einsum(
                        f'{state_str},{G_str},{merged_str}->{out_str}',
                        state, G_int, merged_leaf, optimize=True
                    )
                else:  # j == k - 1: last leaf. Use G_root_cat (no R_chain_up after).
                    G_str = 'ce'
                    merged_str = 'te'
                    out_str = 'p' + ('o' if v_visited[0] else '') + ''.join(ttn_chars) + 't'
                    state = np.einsum(
                        f'{state_str},{G_str},{merged_str}->{out_str}',
                        state, G_root_cat, merged_leaf, optimize=True
                    )
                return

            # Case 3: internal u (not v, not processed). Recurse children, then contract.
            m = len(u.children)
            for child in u.children:
                expand(child)

            # state's last m axes are the children's R_TTN_p axes. Contract with TTN_core[u].
            TTN_core = self.cores[u]                          # (R_TTN_p_u, c1, ..., cm)
            N = state.ndim
            state_axes_to_contract = list(range(N - m, N))
            core_axes_to_contract = list(range(1, m + 1))
            state = np.tensordot(state, TTN_core,
                                  axes=(state_axes_to_contract, core_axes_to_contract))
            # tensordot appends TTN_core's remaining axes (= axis 0 = R_TTN_p_u) at the end.
            # State.ndim is now (N - m) + 1 = N - m + 1.

        # Walk root's children (root has no parent-bond).
        for child in self.root.children:
            expand(child)

        # Final root TTN-core contraction: root has no parent-bond, so its core has
        # m_root axes (one per child). Contract state's last m_root R_TTN_p axes
        # with all m_root axes of TTN_core[root].
        TTN_root = self.cores[self.root]
        m_root = len(self.root.children)
        N = state.ndim
        state = np.tensordot(state, TTN_root,
                              axes=(list(range(N - m_root, N)),
                                     list(range(m_root))))
        # State should now be (R_phi, R_TTN_v_open). Sanity-check:
        if state.ndim != 2:
            raise RuntimeError(
                f"Final state.ndim = {state.ndim}, expected 2 "
                f"(R_phi, R_TTN_v_open). state.shape={state.shape}"
            )

        # state is (R_phi, R_TTN_v_open). We want above_dot_omega = (R_TTN_v, R_phi).
        above_dot_omega = state.T

        subtree_V, row_axes = self._materialize_subtree(v)    # (R_TTN_v, n_below)
        Y = subtree_V.T @ above_dot_omega                     # (n_below, R_phi)
        return Y, row_axes

    def _materialize_omega(self, v: Node,
                            sketch_cores: Dict[Node, np.ndarray]
                            ) -> Tuple[np.ndarray, List]:
        """
        Materialize the dense Omega (TreeStack mirror) from sketch_cores.

        This is a verification/reference helper, NOT used in the structured
        contraction. It contracts the sketch tree fully, returning Omega of
        shape (n_above, R_phi) with rows in the order returned by
        self._materialize_above(v) (so it can be used directly with
        self.apply_unfolding(v, Omega) for cross-checks).

        The recursion mirrors _materialize_above's logic but uses sketch_cores
        at every node and treats v as the active leaf with G_v = sketch_cores[v]
        as the (R_parent, R_phi) "stub". The R_phi axis is treated as a
        leaf-like tail axis at v's position; we then move it to row position 0
        before reshape/transpose so that Omega has rows = above-leaves' phys
        flattened in DFS order, cols = R_phi.
        """
        if v is self.root:
            raise ValueError("Root has no above")

        def expand(u: Node) -> Tuple[np.ndarray, List, Optional[int]]:
            """Returns (T_u, leaf_axes, phi_tail_pos).

            T_u axes:
              * axis 0 = R_parent_bond_in_sketch
              * axis 1+ = "tail" of leaf-phys axes and possibly phi.
            leaf_axes: list of physical_axis labels (or Node for processed) for
                each tail axis EXCEPT phi.
            phi_tail_pos: position of phi WITHIN the tail (so axis 1 + phi_tail_pos
                in T_u), or None if v is not in u's subtree.
            """
            if u is v:
                # G_v: (R_parent, R_phi). Tail = [R_phi] at tail_pos 0.
                return sketch_cores[v], [], 0
            if u.is_leaf or u in self.processed:
                # G_u: (R_parent, n_u). Tail = [n_u] at tail_pos 0.
                label = u.physical_axis if u.is_leaf else u
                return sketch_cores[u], [label], None
            # Internal: G_u (R_parent, R_c1, ..., R_cm).
            T_u = sketch_cores[u]                # (R_parent, R_c1, ..., R_cm)
            leaf_axes: List = []
            phi_tail_pos: Optional[int] = None

            for ci, child in enumerate(u.children):
                child_T, child_leaves, child_phi = expand(child)
                T_u_old_ndim = T_u.ndim          # = (1 + remaining_child_bonds + tail_so_far)
                T_u = np.tensordot(T_u, child_T, axes=([1], [0]))
                # Old axes: [parent (0), R_c{ci} (1), R_c{ci+1}..R_c{m-1} (2..), tail_so_far (...)].
                # Removed axis 1. Appended child_T's axes 1+ (= child's tail).
                # New axes: [parent (0), R_c{ci+1}..R_c{m-1} (1..), tail_so_far (..., shifted left), child_tail (..., appended)].
                # So tail-positions of pre-existing entries decrease by 1; child's tail-positions become old_tail_count - 1 + child_pos.
                old_tail_count = T_u_old_ndim - 1
                if phi_tail_pos is not None:
                    phi_tail_pos -= 1   # was at tail_pos >= 1 (after axis 1 = R_c{ci} removed, shifts left by 1)
                if child_phi is not None:
                    if phi_tail_pos is not None:
                        raise RuntimeError("phi in two subtrees")
                    phi_tail_pos = (old_tail_count - 1) + child_phi
                leaf_axes = leaf_axes + child_leaves

            return T_u, leaf_axes, phi_tail_pos

        # Root: G_root has shape (R_c1, ..., R_cm), no parent-bond.
        # Wrap with phantom parent of size 1.
        G_root = sketch_cores[self.root]
        T = G_root.reshape((1,) + G_root.shape)
        leaf_axes: List = []
        phi_tail_pos: Optional[int] = None

        for ci, child in enumerate(self.root.children):
            child_T, child_leaves, child_phi = expand(child)
            T_old_ndim = T.ndim
            T = np.tensordot(T, child_T, axes=([1], [0]))
            old_tail_count = T_old_ndim - 1
            if phi_tail_pos is not None:
                phi_tail_pos -= 1
            if child_phi is not None:
                if phi_tail_pos is not None:
                    raise RuntimeError("phi in two subtrees")
                phi_tail_pos = (old_tail_count - 1) + child_phi
            leaf_axes = leaf_axes + child_leaves

        # Strip phantom axis 0 (size 1).
        T = T.reshape(T.shape[1:])
        if phi_tail_pos is None:
            raise RuntimeError("phi was not encountered — internal bug")
        # Move phi (currently at axis phi_tail_pos in T, since axis 0 was phantom-stripped)
        # to row position 0.
        T = np.moveaxis(T, phi_tail_pos, 0)
        # Now T axes: [R_phi, leaves_in_DFS_order...]
        n_above = int(np.prod(T.shape[1:]))
        Omega_T = T.reshape(T.shape[0], n_above)   # (R_phi, n_above)
        return Omega_T.T, leaf_axes               # (n_above, R_phi)


    def residual_update(self, v: Node, Q_v: np.ndarray) -> None:
        """
        In-place update of the residual TTN for QBTC at node v.

        Conceptually represents:    T_residual  =  T  contracted-at-v  with  Q_v^T

        For a LEAF v:
          * Original cores[v]:   shape (R_v_old, n_v).
          * Q_v:                  shape (n_v, r).
          * Result: the underlying tensor's v-physical axis (dim n_v) is replaced
                    by a compressed axis (dim r), with Q_v^T applied.

        For an INTERNAL v (whose children are already processed):
          * subtree_V_old(v) is materialized from current cores; shape
                              (R_v_old, n_below_compressed).
          * Q_v:               shape (n_below_compressed, r).
          * Result: v's "n_below" virtual axis is compressed to r.

        Implementation:
          1. subtree_V_old = self._materialize_subtree(v)    # (R_v_old, n_below)
          2. B_v = Q_v.T @ subtree_V_old.T                    # (r, R_v_old)
          3. Contract B_v's R_v_old axis into v's parent's core along v's
             child-bond axis. v's child-bond dim shrinks from R_v_old to r.
          4. Set cores[v] = identity(r, r) and add v to self.processed.

        After this update, v's slot in the residual is "trivial" (an identity
        bond of dim r), and v's parent has absorbed the action of Q_v^T.
        Subsequent residual_update calls on other nodes work uniformly: they
        re-materialize subtrees (which now use the updated cores[parent]) and
        propagate further compressions upward.

        Parameters
        ----------
        v : Node
            The node whose subtree gets compressed. Must NOT be the root, and
            must NOT already be in self.processed.
        Q_v : np.ndarray of shape (n_below_v, r)
            Orthonormal columns recommended (this is what QBTC computes via
            QR), but any matrix with the right shape works for testing.

        Returns
        -------
        None (mutates self in place).
        """
        if v is self.root:
            raise ValueError("Cannot residual_update at the root")
        if v in self.processed:
            raise ValueError(f"{v.name} has already been processed")
        if Q_v.ndim != 2:
            raise ValueError(f"Q_v must be 2D, got shape {Q_v.shape}")

        # Step 1. Materialize subtree at v (using current residual cores).
        subtree_V_old, _ = self._materialize_subtree(v)     # (R_v_old, n_below)
        R_v_old = subtree_V_old.shape[0]
        n_below = subtree_V_old.shape[1]
        if Q_v.shape[0] != n_below:
            raise ValueError(
                f"Q_v.shape[0]={Q_v.shape[0]} doesn't match n_below={n_below} "
                f"of {v.name}'s current subtree"
            )
        r_new = Q_v.shape[1]

        # Step 2. Compute the absorbed coefficient B_v.
        # B_v[r, R] = sum_{a} Q_v[a, r] * subtree_V_old[R, a]
        # Equivalent to: subtree_V_old @ Q_v, then transpose. Or just:
        B_v = Q_v.T @ subtree_V_old.T                        # (r, R_v_old)

        # Step 3. Contract B_v into v's parent's core.
        parent = v.parent
        # v's index in parent.children determines which child-bond axis of
        # parent's core to contract.
        i_v = parent.children.index(v)
        # Parent's child-bond axis for v:
        #   If parent is root:   axis i_v        (root has no parent-bond)
        #   Else (internal):     axis 1 + i_v    (axis 0 is parent's parent-bond)
        if parent is self.root:
            v_axis_in_parent = i_v
        else:
            v_axis_in_parent = 1 + i_v

        parent_core = self.cores[parent]
        # Contract parent_core's v-child-bond axis with B_v's axis 1 (R_v_old).
        # tensordot(parent_core, B_v, axes=([v_axis], [1])):
        #   Result axes = [parent_core's other axes in original order] ++ [B_v's other axes]
        #               = [parent_core minus v_axis] ++ [r-axis from B_v]
        # We want the r-axis to land at position v_axis_in_parent (replacing the contracted axis).
        new_parent_core = np.tensordot(parent_core, B_v, axes=([v_axis_in_parent], [1]))
        # The B_v's axis 0 (= r) is now the LAST axis of new_parent_core.
        # Move it back to v_axis_in_parent.
        new_parent_core = np.moveaxis(new_parent_core, -1, v_axis_in_parent)
        self.cores[parent] = new_parent_core

        # Step 4. Set cores[v] to identity (r, r) and mark as processed.
        # For LEAF v: this is a (r, r) identity in (R_parent_new=r, "n_phys"=r) form.
        # For INTERNAL v: same — but we'd still keep v.children's cores around (they
        #                 stay as identities from prior leaf processing).
        # We override cores[v] to a 2D identity regardless of v's original ndim.
        # Subsequent _materialize_subtree(parent_of_v) will recurse into v but
        # treat it as a leaf-like 2D core (axis 0 = parent-bond, axis 1 = "phys").
        self.cores[v] = np.eye(r_new, dtype=np.float64)
        self.processed.add(v)

    # --------------------------------------------------------------
    # Decomposition (from_dense)
    # --------------------------------------------------------------

    @staticmethod
    def from_dense(T: np.ndarray, root: Node,
                    max_rank: Optional[int] = None,
                    rtol: float = 0.0) -> "TTN":
        """
        Decompose a dense tensor into TTN format via leaves-to-root SVD.

        Each compression step is an exact SVD; rank truncation is governed by
        max_rank (hard cap) and rtol (drop singular values < rtol * sigma_max).
        With both defaults (None / 0.0), every bond is taken at full rank.

        This is the deterministic baseline we'll compare TTN-aware QBTC against.

        Parameters
        ----------
        T : np.ndarray of shape `physical_axes_of_root_subtree_in_axis_order`.
        root : Node
            The tree topology.
        max_rank : int, optional
            If given, cap every internal bond at this rank.
        rtol : float
            Drop bond singular values < rtol * sigma_max.

        Returns
        -------
        TTN
        """
        # Working state: a tensor with one axis per leaf in physical_axis order.
        # We process leaves first; at each step, we SVD the unfolding at the
        # current node, store the U-factor as that node's core, and contract
        # the (S V^T) part into the parent (replacing the child's "axis" with
        # the new compressed bond).

        nodes = all_nodes(root)
        leaves = _subtree_leaves(root)
        leaves_by_axis = sorted(leaves, key=lambda v: v.physical_axis)
        if T.shape != tuple(T.shape):
            raise ValueError("T must be a numpy array with explicit shape")

        # Map each Node to a "current axis" in the working tensor `state`. As
        # we compress nodes, axes get replaced by virtual rank-axes; we track
        # the axis position of each unprocessed node here.
        state = T.copy()

        # axis_of[v]: current axis index of v in `state`, or None if v has been
        # absorbed already.
        axis_of: Dict[Node, Optional[int]] = {}
        for lv in leaves_by_axis:
            axis_of[lv] = lv.physical_axis
        for v in nodes:
            if not v.is_leaf:
                axis_of[v] = None  # not in the working tensor yet

        cores: Dict[Node, np.ndarray] = {}

        def truncate_rank(s: np.ndarray) -> int:
            """Determine effective rank from singular values."""
            if len(s) == 0:
                return 0
            r = len(s)
            if rtol > 0:
                r = min(r, int(np.sum(s > rtol * s[0])))
            if max_rank is not None:
                r = min(r, max_rank)
            return max(1, r)

        def process_node(v: Node):
            """
            For node v, identify its 'incoming axes' in `state` (the leaves'
            axes if v is a leaf, or its children's currently-virtual-bond axes
            if v is internal); SVD the unfolding (those axes vs the rest);
            store U as cores[v]; replace those axes in state with the new
            compressed-bond axis (= the SV*Vt absorbed into the rest of state).
            """
            nonlocal state
            if v is root:
                # Root: just store what's left. No more SVD.
                # state should have axes corresponding exactly to root's
                # children's compressed bonds, in v.children order.
                in_axes = [axis_of[c] for c in v.children]
                # Permute state so those axes come first, then squeeze any extras.
                others = [a for a in range(state.ndim) if a not in in_axes]
                if others:
                    raise ValueError(
                        f"Root processing: unexpected leftover axes {others}"
                    )
                state = state.transpose(in_axes)
                cores[v] = state
                return

            # Determine which axes of `state` v is contributing.
            if v.is_leaf:
                in_axes = [axis_of[v]]
            else:
                in_axes = [axis_of[c] for c in v.children]

            # Move those axes to the front, others to the back.
            others = [a for a in range(state.ndim) if a not in in_axes]
            perm = list(in_axes) + others
            state_perm = state.transpose(perm)
            n_in = int(np.prod([state.shape[a] for a in in_axes]))
            n_out = int(np.prod([state.shape[a] for a in others]))
            M = state_perm.reshape(n_in, n_out)

            # SVD: U is (n_in, r), S is (r,), Vt is (r, n_out).
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            r_eff = truncate_rank(S)
            U = U[:, :r_eff]                       # (n_in, r_eff)
            SVt = (S[:r_eff, None] * Vt[:r_eff, :])  # (r_eff, n_out)

            # Store v's core. Reshape U to (in_dims..., r_eff), then move
            # the rank axis to the front to match our "axis 0 = parent-bond"
            # convention.
            in_dims = [state.shape[a] for a in in_axes]
            U_resh = U.reshape(*in_dims, r_eff)             # (in_dims..., r_eff)
            # Convention: axis 0 = parent-bond (= r_eff axis), then in_dims.
            U_resh = np.moveaxis(U_resh, -1, 0)              # (r_eff, *in_dims)
            cores[v] = U_resh

            # Replace v's contribution to `state`: SVt has shape (r_eff, n_out).
            # Reshape to (r_eff, *out_dims).
            out_dims = [state.shape[a] for a in others]
            new_state = SVt.reshape(r_eff, *out_dims)        # (r_eff, *out_dims)

            # Update axis_of:
            # The 'others' axes go to positions 1, 2, ..., in `new_state`. We
            # need to figure out which original axes they correspond to and
            # update `axis_of` accordingly.
            # Map: original axis a (in `state`) → new axis in `new_state`.
            # In `new_state`, axis 0 = r_eff (= v's bond to parent), axes 1..k =
            # the 'others' axes in their original ORDER (since `others` is
            # sorted ascending and `transpose(perm).reshape` preserves order).
            new_axis_for_old: Dict[int, int] = {a: 1 + i for i, a in enumerate(others)}
            # All nodes not yet processed and whose axis_of is one of `others`
            # get their axis_of remapped:
            for u in nodes:
                ax = axis_of.get(u)
                if ax is not None and ax in new_axis_for_old:
                    axis_of[u] = new_axis_for_old[ax]
            # And v itself gets axis 0 in the new state (its bond to parent).
            axis_of[v] = 0
            state = new_state

        for v in _post_order(root):
            process_node(v)

        return TTN(root, cores)


# ============================================================
# Phase 4: TTN-aware QBTC end-to-end
# ============================================================

def qbtc_ttn(ttn_input: "TTN", target_r: int,
             finish: str = "qb_svd", P: int = 2,
             R_per_copy: Optional[int] = None,
             sketch_kind: str = "gaussian",
             above_mode: str = "qr",
             rng_seed: Optional[int] = None) -> "TTN":
    """
    TTN-aware QBTC: takes a TTN as input and produces a compressed TTN of
    target rank `target_r` at every internal edge.

    Parameters
    ----------
    ttn_input : TTN
        Input TTN. Must NOT have any processed nodes.
    target_r : int
        Compressed rank at every output edge.
    finish : "qb", "qb_svd", "qb_svd_exact", or "qb_cbc"
        * "qb":          Plain QR truncation — Q_v = Q_full[:, :r]. One pass.
        * "qb_svd":      SVD-of-Y finish (the default). Q_v = top-r left
                         singular vectors of the range sketch Y itself,
                         obtained as Q_full @ U_R[:, :r] where U_R diagonalises
                         the small QR factor of Y (Y = Q_full R_qr). By the
                         identity Y = Q_full (B Omega) with B = Q_full.T @ M_v,
                         this is exactly a ONE-PASS randomized SVD of the
                         projected core B through the sketch Omega. Cheap
                         (no n_above, no Gram, no R^4) and full precision
                         (no spectrum squaring -> no ~1e-8 floor). At
                         sketch_dim = r it coincides with "qb"; with
                         oversampling it improves on "qb", recovering most
                         (not all) of the exact-SVD gain. above_mode is
                         IGNORED by this finish.
        * "qb_svd_exact": Former qb_svd, RETAINED for reference (not used by
                         the experiments). Aligns Q_v with the EXACT top-r left
                         singulars of the full unfolding M_v via the R_v x R_v
                         above-Gram. Only finish with the sharp (1+eps) per-node
                         guarantee, but it pays the double-tree Gram cost:
                         O(R_input^4) intermediates (above_mode="gram",
                         n_above-free) or a thin QR of above_V (above_mode="qr",
                         forms n_above).
        * "qb_cbc":      Cholesky-Based-Compression-style finish — Q_v aligned
                         with top-r left singulars of just A = Q_full.T @
                         subtree_V.T (the local core projected by Q). Skips
                         the above-Gram entirely; one tiny SVD per node.
    P : int
        Number of stacked sketch copies.
    R_per_copy : int, optional
        Sketch dim per copy. Default = target_r (so total sketch dim = P * target_r).
    sketch_kind : "gaussian", "kr", "treestack", or "ttstack"
        * "gaussian":   dense Gaussian Omega of shape (n_above, sketch_dim).
                        Same family as ttn_hmt. Materializes n_above for the
                        sketch step itself.
        * "kr":         Khatri-Rao Omega — one Gaussian factor per above-leaf,
                        column-wise Kronecker-producted. Sketch step does NOT
                        materialize n_above.
        * "treestack":  Open-leaf TreeStack with mirror topology. P stacked
                        copies, each computed via a fully-structured leaf-by-
                        leaf contraction. Memory bounded by R_TTN × R_sketch.
        * "ttstack":    Open-leaf TreeStack with caterpillar (TT path-graph)
                        topology, contracted by apply_unfolding_TT_struct in
                        optimal order. Fully structured (never forms n_above);
                        peak intermediate ~R_TTN^3, bond-bounded like treestack.

    The default "qb_svd" finish keeps the isometry property
    (Q_v.T @ Q_v = I_r), so the orthogonal error decomposition still holds
    exactly; only its per-node bound weakens from the exact (1+eps) to the
    plain-QB (rank-revealing) bound, since it does not realise the best
    rank-r subspace inside the sketched range. Use "qb_svd_exact" if the sharp
    (1+eps) constant is required and the R_input^4 Gram cost is affordable.
    rng_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    TTN
        Compressed TTN with the same tree topology.
    """
    if ttn_input.processed:
        raise ValueError("ttn_input must have no processed nodes")
    if sketch_kind not in ("gaussian", "kr", "treestack", "ttstack"):
        raise ValueError(f"Unknown sketch_kind: {sketch_kind!r}; expected "
                         f"'gaussian', 'kr', 'treestack', or 'ttstack'")
    if R_per_copy is None:
        R_per_copy = target_r
    sketch_dim = P * R_per_copy
    rng = np.random.default_rng(rng_seed)

    # Build a residual TTN by copying the input cores. We mutate this as we
    # sweep — residual_update absorbs each Q_v into its parent.
    residual = TTN(ttn_input.root, dict(ttn_input.cores))

    out_cores: Dict[Node, np.ndarray] = {}

    for v in _post_order(ttn_input.root):
        if v is ttn_input.root:
            continue

        # 1. Materialize subtree_V (always cheap — bounded by R_v × n_below).
        subtree_V, _ = residual._materialize_subtree(v)   # (R_v_old, n_below)
        n_below = subtree_V.shape[1]

        # 2. Sketch — branch on sketch_kind.
        if sketch_kind == "gaussian":
            # Materialize above_V (n_above × R_v).
            above_V, _ = residual._materialize_above(v)        # (R_v_old, n_above)
            n_above = above_V.shape[1]
            Omega = rng.standard_normal((n_above, sketch_dim))
            Y = subtree_V.T @ (above_V @ Omega)               # (n_below, sketch_dim)
        elif sketch_kind == "kr":
            # Khatri-Rao: one Gaussian factor per above-leaf.
            above_nodes = residual.above_leaves(v)
            K_above = {}
            for u in above_nodes:
                n_u = residual.cores[u].shape[1]
                K_above[u] = rng.standard_normal((n_u, sketch_dim)) / np.sqrt(n_u)
            Y, _ = residual.apply_unfolding_KR(v, K_above)    # (n_below, sketch_dim)
        elif sketch_kind == "treestack":
            # P independent stacked TreeStack copies.
            Y_cols = []
            for _ in range(P):
                sketch_cores = residual.build_treestack_cores(v, R_per_copy, rng)
                Y_p, _ = residual.apply_unfolding_TS(v, sketch_cores)   # (n_below, R_per_copy)
                Y_cols.append(Y_p)
            Y = np.concatenate(Y_cols, axis=1) / np.sqrt(P)
        elif sketch_kind == "ttstack":
            # P independent stacked TTStack copies (caterpillar sketch),
            # each via the fully-structured DFS+chain interleaved contraction.
            # Memory bounded by R_chain × R_phi × R_TTN^max_concurrent.
            Y_cols = []
            for _ in range(P):
                Y_p, _ = residual.apply_unfolding_TT_struct(v, R_per_copy, rng)
                Y_cols.append(Y_p)
            Y = np.concatenate(Y_cols, axis=1) / np.sqrt(P)

        # 3. QR.  Keep R_qr — the SVD-of-Y finish reuses it.
        Q_full, R_qr = np.linalg.qr(Y)                    # (n_below, k), (k, sketch_dim)

        # 4. Finish.
        if finish == "qb":
            r_eff = min(target_r, Q_full.shape[1])
            Q_v = Q_full[:, :r_eff]
        elif finish == "qb_svd":
            # SVD-of-Y finish (default). Take the top-r LEFT singular vectors
            # of the range sketch Y itself. The economy QR gives
            # Y = Q_full @ R_qr with Q_full orthonormal, so svd(Y) lifts
            # svd(R_qr):  U_Y = Q_full @ U_R, hence Q_v = Q_full @ U_R[:, :r].
            # Equivalently (Y = Q_full (B Omega), B = Q_full.T @ M_v) this is a
            # ONE-PASS randomized SVD of the projected core B through Omega.
            # No above_V, no Gram, no R^4, full precision; above_mode unused.
            U_R, sv, _ = np.linalg.svd(R_qr, full_matrices=False)
            # Keep the SAME rank plain-QB keeps; do NOT shrink to the numerical
            # rank of Y.  Below the floating-point floor (fast-decaying spectra at
            # high target rank) the trailing singular directions of Y are
            # rounding-level.  Shrinking to the numerical rank would commit a
            # lower-rank core than requested and leave a LARGER residual than plain
            # QB, even though the two finishes are mathematically identical at
            # PR = r (same column space col(Y)).  Keeping target_r columns makes
            # QB+SVD never worse than plain QB, as the theory predicts, all the way
            # down to machine precision.
            r_eff = min(target_r, Q_full.shape[1])
            if r_eff >= Q_full.shape[1]:
                # No subspace selection happens (e.g. PR = r, no oversampling):
                # the SVD rotation spans all of col(Y) and is an inert gauge choice
                # that leaves col(Q_v) = col(Y) unchanged.  Use the stable QR basis
                # directly, so QB+SVD coincides EXACTLY with plain QB (no rotation
                # rounding) rather than sitting a few ulp above it at the floor.
                Q_v = Q_full[:, :r_eff]
            else:
                Q_v = Q_full @ U_R[:, :r_eff]
        elif finish == "qb_svd_exact":
            # RETAINED for reference; NOT used by the experiments. Exact top-r
            # left singular vectors of the FULL unfolding B = Q_full.T @ M_v =
            # A @ above_V, recovered from C = A @ R_a.T (sketch_dim x R_v) with
            # A = Q_full.T @ subtree_V.T and R_a.T @ R_a = G := above_V above_V.T
            # (so C C.T = A G A.T = B B.T). Only finish with the sharp (1+eps)
            # per-node guarantee, but it pays the double-tree above-Gram cost
            # (O(R_input^4) intermediates).
            #   above_mode="gram" : G = compute_above_gram(v), n_above-free,
            #                       symmetric sqrt via eigh (squares spectrum,
            #                       ~1e-8 floor).
            #   above_mode="qr"   : R_a = qr(above_V.T), backward stable, but
            #                       MATERIALISES above_V (R_v x n_above).
            A = Q_full.T @ subtree_V.T                    # (sketch_dim, R_v)
            if above_mode == "gram":
                G = residual.compute_above_gram(v)        # (R_v, R_v), no n_above
                G = 0.5 * (G + G.T)
                w, Vg = np.linalg.eigh(G)                 # eigenvalues = sigma(above_V)^2
                w = np.clip(w, 0.0, None)
                R_a = np.sqrt(w)[:, None] * Vg.T          # (R_v, R_v): R_a.T @ R_a = G
            else:  # "qr"
                above_V, _ = residual._materialize_above(v)   # (R_v, n_above)
                R_a = np.linalg.qr(above_V.T, mode="r")       # (R_v, R_v), no squaring
            C = A @ R_a.T                                 # (sketch_dim, R_v)
            U_C, sv, _ = np.linalg.svd(C, full_matrices=False)
            top = sv[0] if sv.size > 0 else 0.0
            # rcond-style relative threshold on sigma (NOT sigma^2).
            thresh = top * np.finfo(C.dtype).eps * max(C.shape)
            r_keep = int(np.count_nonzero(sv > thresh))
            r_eff = min(target_r, r_keep)
            Q_v = Q_full @ U_C[:, :r_eff]
        elif finish == "qb_cbc":
            # Cholesky-Based-Compression-style finish: SVD of just
            #   A = Q_full.T @ subtree_V.T          (sketch_dim, R_v)
            # i.e. the LOCAL core at v projected by Q (in the residual,
            # subtree_V *is* the local core reshaped, since v's children
            # are identity-stub'd after their own compression).
            #
            # Compared to qb_svd this skips compute_above_gram entirely:
            # the SVD aligns Q_v with the dominant directions of just the
            # local core, NOT the full unfolding M_v. Result is one tiny
            # SVD of a (sketch_dim × R_v) matrix per node — no R^4 Gram,
            # no n_above contraction, no second QR (Q already has
            # orthonormal columns and U_A's columns are orthonormal, so
            # Q_v = Q_full @ U_A[:, :r] is automatically orthonormal).
            A = Q_full.T @ subtree_V.T               # (sketch_dim, R_v)
            U_A, _, _ = np.linalg.svd(A, full_matrices=False)
            r_eff = min(target_r, U_A.shape[1])
            Q_v = Q_full @ U_A[:, :r_eff]
        else:
            raise ValueError(f"Unknown finish: {finish!r}")

        # 5. Reshape Q_v into the OUTPUT TTN's multi-axis convention.
        if v.is_leaf:
            out_cores[v] = Q_v.T                          # (r_eff, n_v)
        else:
            child_dims = [out_cores[c].shape[0] for c in v.children]
            Q_v_md = Q_v.reshape(*child_dims, Q_v.shape[1])     # (r_c1, ..., r_cm, r_eff)
            Q_v_md = np.moveaxis(Q_v_md, -1, 0)                  # (r_eff, r_c1, ..., r_cm)
            out_cores[v] = Q_v_md

        # 6. Residual update: contract Q_v^T into v's parent.
        residual.residual_update(v, Q_v)

    # Root: take whatever remains in the residual.
    out_cores[ttn_input.root] = residual.cores[ttn_input.root]

    return TTN(ttn_input.root, out_cores)


# ============================================================
# NOTE: development self-tests that cross-checked this module against the dense
# `qbtc` reference were removed with the dense path. Correctness of the TTN-native
# pipeline is now exercised by experiments/sanity_check_hilbert_6d.py and the
# other experiments/ drivers.
# ============================================================
