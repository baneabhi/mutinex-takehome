"""A second, independent implementation of minimum-cost assignment.

**Test fixture, not production code.** It lives here rather than in the package
because nothing ships it — ``mission_control.solver`` delegates to SciPy.

It exists because SciPy cannot check itself. The production solver used to be
this code, verified against ``scipy.optimize.linear_sum_assignment``; now that
SciPy *is* the production solver that comparison is circular. Keeping the
hand-written version as an oracle preserves the check at sizes brute force
cannot reach — brute force is factorial and gives up around 6x6, while this
agrees with the real solver on thousands of candidates.

It also pins the part of ``solver.py`` that is genuinely ours and could be
wrong: the sentinel sizing, the padding, and the interpretation at the boundary.
A bug in any of those shows up as a disagreement here.

The algorithm is the Hungarian algorithm by successive shortest augmenting
paths. Rows join the matching one at a time; each new row is added by finding
the cheapest chain of reassignments. Potentials keep every reduced cost
non-negative and every matched pair at exactly zero, which is what lets the
chain search be a plain Dijkstra — displacing a row from a column is a negative
move in raw costs, and Dijkstra cannot take negative edges.
"""

from __future__ import annotations

from collections.abc import Sequence

Cost = int
Matrix = Sequence[Sequence[Cost | None]]

UNREACHED = float("inf")


def _cheapest_chain(
    cost: list[list[Cost]],
    start_row: int,
    row_holding_col: list[int | None],
    row_potential: list[Cost],
    col_potential: list[Cost],
) -> tuple[int | None, list[Cost | float], list[int | None], list[bool]]:
    """Dijkstra from ``start_row``: a row reaches any column at its reduced
    cost, and a column reaches the row holding it free of charge. Stops at the
    first unheld column it settles."""
    column_count = len(col_potential)
    distance: list[Cost | float] = [UNREACHED] * column_count
    arrived_from_row: list[int | None] = [None] * column_count
    settled = [False] * column_count

    current_row = start_row
    distance_to_current_row: Cost | float = 0

    while True:
        for col in range(column_count):
            if settled[col]:
                continue
            reduced = (
                cost[current_row][col] - row_potential[current_row] - col_potential[col]
            )
            if distance_to_current_row + reduced < distance[col]:
                distance[col] = distance_to_current_row + reduced
                arrived_from_row[col] = current_row

        nearest = None
        for col in range(column_count):
            if not settled[col] and (nearest is None or distance[col] < distance[nearest]):
                nearest = col
        if nearest is None or distance[nearest] == UNREACHED:
            return None, distance, arrived_from_row, settled
        settled[nearest] = True

        if row_holding_col[nearest] is None:
            return nearest, distance, arrived_from_row, settled

        current_row = row_holding_col[nearest]
        distance_to_current_row = distance[nearest]


def _match_rows_one_by_one(
    cost: list[list[Cost]], row_count: int, column_count: int
) -> list[int | None]:
    row_potential: list[Cost] = [0] * row_count
    col_potential: list[Cost] = [0] * column_count
    row_holding_col: list[int | None] = [None] * column_count
    col_held_by_row: list[int | None] = [None] * row_count

    for start_row in range(row_count):
        free_col, distance, arrived_from_row, settled = _cheapest_chain(
            cost, start_row, row_holding_col, row_potential, col_potential
        )
        if free_col is None:
            continue

        # Restore the invariant, reading the *old* matching, before augmenting.
        chain_cost = distance[free_col]
        row_potential[start_row] += chain_cost  # type: ignore[assignment]
        for col, is_settled in enumerate(settled):
            if not is_settled:
                continue
            slack = chain_cost - distance[col]
            col_potential[col] -= slack  # type: ignore[assignment]
            holder = row_holding_col[col]
            if holder is not None:
                row_potential[holder] += slack  # type: ignore[assignment]

        # Walk the chain back, moving every row on it one column along.
        col = free_col
        while True:
            row = arrived_from_row[col]
            assert row is not None
            displaced_col = col_held_by_row[row]
            row_holding_col[col] = row
            col_held_by_row[row] = col
            if row == start_row:
                break
            col = displaced_col  # type: ignore[assignment]

    return col_held_by_row


def assign(cost: Matrix) -> list[int | None]:
    """Same contract as ``mission_control.solver.assign``, independently
    implemented — including the sentinel and the padding, so the comparison
    covers those too."""
    row_count = len(cost)
    if row_count == 0:
        return []
    given_columns = len(cost[0])

    real_costs = [cell for row in cost for cell in row if cell is not None]
    sentinel = row_count * ((max(real_costs) if real_costs else 0) + 1) + 1

    column_count = max(given_columns, row_count)
    dense = [
        [
            cost[row][col]
            if col < given_columns and cost[row][col] is not None
            else sentinel
            for col in range(column_count)
        ]
        for row in range(row_count)
    ]

    matched = _match_rows_one_by_one(dense, row_count, column_count)
    return [
        col
        if col is not None and col < given_columns and cost[row][col] is not None
        else None
        for row, col in enumerate(matched)
    ]
