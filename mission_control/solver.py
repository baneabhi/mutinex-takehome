"""Minimum-cost assignment, backed by SciPy.

Rows are slots, columns are candidates. An ineligible pair is simply an edge the
caller does not supply. :func:`assign_pairs` gives each row a distinct column,
or ``None`` where the row could not be filled.

Three entry points, narrowest first:

==========================  =================================================
:func:`assign_pairs`        the primitive — eligible pairs as flat arrays
:func:`assign_masked`       dense cost array plus a boolean eligibility mask
:func:`assign`              list of lists, ``None`` meaning ineligible
==========================  =================================================

:mod:`mission_control.matching` uses the first; the others exist for tests and
small problems.


The "unfilled" column
---------------------

Both of SciPy's routines raise when the rows cannot all be matched —
``optimize.linear_sum_assignment`` gives ``cost matrix is infeasible``,
``sparse.csgraph.min_weight_full_bipartite_matching`` gives ``no full matching
exists``. §7.5 wants the opposite: *infeasible slots are reported ``unfilled``
with a reason, because a partial team is actionable and an exception is not.*

So either way we supply an escape:

    **one extra column per row, reachable only by that row, priced above any
    real assignment.**

A row matched to its own column is a slot nobody could fill. A full matching
therefore always exists and the exception becomes unreachable. Priced above the
total of every real cost put together, it also makes the solver **fill as many
rows as possible first, and only then minimise cost** — a cheaper team that
leaves a slot empty is never preferred, which is exactly §7.6's point.

Infeasibility is therefore *not* what decides between the two routines.


Why sparse
----------

Eligibility here is sparse: a requirement matches a fraction of the roster, so
most ``(slot, crew)`` pairs do not exist. Two consequences.

**A dense matrix has no absent cell.** Every impossible pair still carries a
number the solver considers — 23M of 40M cells at 2000 slots against 20000 crew
— so its work grows with the **roster** rather than with how many people are
actually eligible. Holding eligible pairs fixed at 120000 and growing the roster
with people who qualify for nothing, 300 -> 64000: dense 1 ms -> 74 ms, sparse
15 ms -> 25 ms.

**Sparse is what makes the pruning possible**, and that is where most of the
speed comes from. Each slot only needs its ``R`` cheapest candidates, ``R``
being the number of slots (:mod:`mission_control.matching` applies this; §7.5
has the proof). In a dense matrix you cannot drop a cell — the solver still
walks rows x columns — so the reduction has no dense equivalent.

On identical input, 2000 slots against 20000 crew with 16.8M eligible pairs:

======================================  ========  ==========
dense + a large finite cost per hole      41 s     320 MB
sparse, all eligible pairs                20 s
sparse + the R-cheapest pruning          4.2 s     4M edges
======================================  ========  ==========

**Below a few thousand crew, dense is faster** — building the sparse structure
costs more than the cells it saves. This is a scale-driven choice, not an
unconditional one.


What this costs us
------------------

**Tie-breaking is no longer ours.** Where several assignments are equally
optimal, which one comes back is SciPy's choice — deterministic for a given
input and SciPy version, but not contractual across versions. Ties are common
here, because a requirement with ``count: 2`` produces two identical rows. So
§7.5's determinism claim is really "identical inputs and an identical SciPy
version give identical teams". Callers needing a stable *order* must impose it;
:mod:`mission_control.matching` does, by sorting candidates by member id.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

Cost = int
Matrix = Sequence[Sequence[Cost | None]]


def assign_pairs(
    row_indices: Sequence[int] | np.ndarray,
    col_indices: Sequence[int] | np.ndarray,
    costs: Sequence[Cost] | np.ndarray,
    *,
    row_count: int,
    column_count: int,
) -> list[int | None]:
    """Assign each row to a distinct column, minimising the total.

    The three arrays are parallel and list **only the eligible pairs**; every
    pair must appear at most once. Returns one entry per row: the column index,
    or ``None`` if that row could not be filled.

    Columns with no edges are simply never matched, so a caller can leave gaps
    in its column numbering — which is what lets
    :mod:`mission_control.matching` exclude already-placed candidates without
    renumbering anything.
    """
    if row_count == 0:
        return []

    rows = np.asarray(row_indices, dtype=np.int64)
    cols = np.asarray(col_indices, dtype=np.int64)
    # Shifted by one because SciPy's sparse routines treat an explicit zero as
    # an *absent* edge, and rank-as-cost (§7.5) makes the best candidate's cost
    # exactly zero. A constant shift cannot reorder two assignments of the same
    # size, and assignments of different sizes are separated by `unfilled` below.
    weights = np.asarray(costs, dtype=np.float64) + 1.0

    unfilled = (
        float(row_count) * (float(weights.max()) + 1.0) + 1.0 if weights.size else 1.0
    )
    """Strictly greater than the total of every real cost: at most ``row_count``
    edges, each at most ``weights.max()``. So one unfilled slot always outweighs
    any saving elsewhere, and fewer unfilled slots always wins."""

    own_column = column_count + np.arange(row_count, dtype=np.int64)

    biadjacency = csr_matrix(
        (
            np.concatenate([weights, np.full(row_count, unfilled)]),
            (
                np.concatenate([rows, np.arange(row_count, dtype=np.int64)]),
                np.concatenate([cols, own_column]),
            ),
        ),
        shape=(row_count, column_count + row_count),
    )

    matched_rows, matched_cols = min_weight_full_bipartite_matching(biadjacency)

    chosen: list[int | None] = [None] * row_count
    for row, col in zip(matched_rows.tolist(), matched_cols.tolist()):
        if col < column_count:  # otherwise it took its own column: unfilled
            chosen[row] = col
    return chosen


def assign_masked(costs: np.ndarray, eligible: np.ndarray) -> list[int | None]:
    """:func:`assign_pairs` over a dense cost array and a boolean mask of the
    same shape. Cost values where ``eligible`` is ``False`` are never read."""
    row_count, column_count = costs.shape
    rows, cols = np.nonzero(eligible)
    return assign_pairs(
        rows, cols, costs[rows, cols], row_count=row_count, column_count=column_count
    )


def assign(cost: Matrix) -> list[int | None]:
    """:func:`assign_pairs` over a list of lists, where ``None`` means
    ineligible.

    The readable form, for small problems and for tests. Converting cell by cell
    in Python is fine at that size; the other two are what to reach for when it
    is not.
    """
    row_count = len(cost)
    if row_count == 0:
        return []
    column_count = len(cost[0])

    rows: list[int] = []
    cols: list[int] = []
    weights: list[Cost] = []
    for row, cells in enumerate(cost):
        for col, cell in enumerate(cells):
            if cell is not None:
                rows.append(row)
                cols.append(col)
                weights.append(cell)

    return assign_pairs(
        rows, cols, weights, row_count=row_count, column_count=column_count
    )


def greedy(cost: Matrix) -> list[int | None]:
    """Fill each row in turn with its cheapest remaining eligible column.

    Not used in production, and deliberately not backed by SciPy. It exists so
    §7.6's test can demonstrate what the global solver buys: greedy strands rows
    whose only eligible column was taken by an earlier row, producing a mission
    that cannot be activated. No row ordering saves it — filling the scarce slot
    first works on any single example, but the symmetric counterexample is one
    row away.
    """
    taken: set[int] = set()
    chosen: list[int | None] = []

    for row in cost:
        best_col: int | None = None
        best_cost: Cost | None = None
        for col, cell in enumerate(row):
            if cell is None or col in taken:
                continue
            if best_cost is None or cell < best_cost:
                best_col, best_cost = col, cell
        if best_col is not None:
            taken.add(best_col)
        chosen.append(best_col)

    return chosen


def total(cost: Matrix, assignment: Sequence[int | None]) -> Cost:
    """Total cost of the filled rows.

    Unfilled rows contribute nothing, so compare :func:`filled` first — a
    cheaper total over fewer rows is worse, not better.
    """
    return sum(
        cost[row][col]  # type: ignore[misc]
        for row, col in enumerate(assignment)
        if col is not None and cost[row][col] is not None
    )


def filled(assignment: Sequence[int | None]) -> int:
    return sum(1 for col in assignment if col is not None)
