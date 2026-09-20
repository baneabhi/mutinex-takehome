"""The assignment algorithm (§7.5, §10).

Property tests rather than fixtures, because the thing being claimed is a
property: the global solver never does worse than greedy, and it agrees with
brute force wherever brute force is affordable.
"""

from __future__ import annotations

import itertools
import random

import pytest

from mission_control.solver import assign, filled, greedy, total

from . import reference_solver


def quality(cost, assignment):
    """Fill first, then cost. A cheaper total over fewer slots is worse — a
    partial team may not be able to fly at all (§7.6)."""
    return (-filled(assignment), total(cost, assignment))


def brute_force(cost):
    rows, cols = len(cost), len(cost[0])
    width = max(cols, rows)
    padded = [[(cost[i][j] if j < cols else None) for j in range(width)] for i in range(rows)]
    best = None
    for choice in itertools.permutations(range(width), rows):
        candidate = [j if padded[i][j] is not None else None for i, j in enumerate(choice)]
        score = quality(padded, candidate)
        if best is None or score < best:
            best = score
    return best


def random_matrix(rng, max_rows=5, max_cols=5, density=0.5):
    rows, cols = rng.randint(1, max_rows), rng.randint(1, max_cols)
    return [
        [rng.randint(0, 9) if rng.random() < density else None for _ in range(cols)]
        for _ in range(rows)
    ]


def test_an_empty_problem_is_an_empty_answer():
    assert assign([]) == []


def test_every_assignment_is_a_valid_matching():
    """No column used twice, and no row placed on an ineligible pair."""
    rng = random.Random(1)
    for _ in range(2000):
        cost = random_matrix(rng)
        result = assign(cost)

        used = [j for j in result if j is not None]
        assert len(used) == len(set(used)), (cost, result)
        for i, j in enumerate(result):
            assert j is None or cost[i][j] is not None, (cost, result)


def test_the_solver_agrees_with_brute_force():
    rng = random.Random(2)
    for _ in range(1500):
        cost = random_matrix(rng)
        assert quality(cost, assign(cost)) == brute_force(cost), cost


def test_the_solver_agrees_with_an_independent_implementation():
    """The oracle, for sizes brute force cannot reach.

    The production solver delegates to SciPy, so SciPy cannot check it. This
    compares against ``tests/reference_solver.py`` — a hand-written Hungarian
    implementation kept for exactly this purpose — which also covers the parts
    that are ours rather than SciPy's: the sentinel sizing, the padding, and the
    interpretation at the boundary.
    """
    rng = random.Random(99)
    for _ in range(500):
        cost = random_matrix(
            rng, max_rows=10, max_cols=14, density=rng.choice([0.2, 0.5, 0.9, 1.0])
        )
        assert quality(cost, assign(cost)) == quality(cost, reference_solver.assign(cost)), cost


def test_it_fills_the_maximum_possible_number_of_rows():
    """The property the sentinel exists to guarantee, checked against a
    different algorithm entirely.

    ``maximum_bipartite_matching`` (Hopcroft-Karp) answers only the feasibility
    half — how many rows *can* be matched at all, ignoring cost. The sentinel is
    sized so that filling more rows always beats filling them cheaply, so the
    two numbers must agree. If the sentinel were ever too small, the solver
    would quietly trade a filled slot for a cheaper total and this would catch
    it.
    """
    numpy = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    csgraph = pytest.importorskip("scipy.sparse.csgraph")

    def largest_possible(cost):
        eligible = numpy.array(
            [[cell is not None for cell in row] for row in cost], dtype=numpy.int8
        )
        matched = csgraph.maximum_bipartite_matching(
            sparse.csr_matrix(eligible), perm_type="column"
        )
        return int((matched != -1).sum())

    rng = random.Random(7)
    for _ in range(600):
        cost = random_matrix(rng, max_rows=8, max_cols=10, density=rng.choice([0.2, 0.5, 0.8]))
        assert filled(assign(cost)) == largest_possible(cost), cost


def test_the_sentinel_survives_costs_far_larger_than_the_matrix():
    """The sentinel is sized from the largest real cost, so a single huge cost
    must not make "leave a slot empty" look cheaper than filling it."""
    cost = [
        [1, None],
        [10**9, None],
    ]
    # Row 1's only eligible column is 0, which row 0 also wants and is cheaper
    # on. One of them must go unfilled — but exactly one, not both.
    assert filled(assign(cost)) == 1

    cost = [
        [1, 10**9],
        [10**9, None],
    ]
    # Now both can be filled, and only by paying the huge cost on one of them.
    assert filled(assign(cost)) == 2


def test_the_global_total_is_never_worse_than_greedy():
    """§10's stated property. Holds for cost *and* for how many slots got
    filled, which is the part that decides whether a mission can fly."""
    rng = random.Random(3)
    for _ in range(2000):
        cost = random_matrix(rng, max_rows=6, max_cols=6)
        assert quality(cost, assign(cost)) <= quality(cost, greedy(cost))


def test_greedy_is_sometimes_strictly_worse():
    """Otherwise the previous test would pass with ``assign = greedy``."""
    rng = random.Random(4)
    strictly_better = 0
    for _ in range(2000):
        cost = random_matrix(rng, max_rows=6, max_cols=6, density=0.35)
        if quality(cost, assign(cost)) < quality(cost, greedy(cost)):
            strictly_better += 1
    assert strictly_better > 100, strictly_better


def test_greedy_strands_a_slot_the_global_solver_fills():
    """§7.6's example, as a matrix.

    Rows are [Pilot, Flight Surgeon]; columns are [Anya, Boris, Chen]. Anya is
    eligible for both, Boris for Pilot only, Chen for neither.
    """
    cost = [
        [0, 1, None],  # Pilot: Anya rank 0, Boris rank 1
        [0, None, None],  # Flight Surgeon: Anya only
    ]
    assert greedy(cost) == [0, None], "greedy takes Anya for Pilot and strands the Surgeon"
    assert assign(cost) == [1, 0], "the global optimum fills both"
    assert filled(greedy(cost)) == 1
    assert filled(assign(cost)) == 2


def test_the_result_is_about_feasibility_not_ranking():
    """The same fixture with **uniform costs** (§10).

    This pins that §7.6 is a feasibility argument that depends on eligibility,
    not on ranking — which is why §7.4 can have no preference model without
    weakening it.
    """
    uniform = [
        [0, 0, None],
        [0, None, None],
    ]
    assert filled(greedy(uniform)) == 1
    assert filled(assign(uniform)) == 2


def test_a_forbidden_pair_is_never_chosen_even_when_it_is_the_only_option():
    assert assign([[None, None]]) == [None]
    assert assign([[None], [None]]) == [None, None]


def test_more_slots_than_candidates_fills_what_it_can():
    """Padding is on the candidate side and must be ineligible, not zero-cost —
    a zero-cost dummy would be cheaper than a real person."""
    cost = [[3], [5], [7]]
    result = assign(cost)
    assert filled(result) == 1
    assert result.count(None) == 2


def test_a_zero_cost_candidate_is_still_preferred_over_no_candidate():
    """The specific failure a zero-cost dummy column would cause."""
    cost = [[0], [0]]
    result = assign(cost)
    assert filled(result) == 1


def test_minimising_the_total_is_not_every_slots_first_choice():
    """Intended: the solver may take rank 2 on one slot to avoid rank 8 on
    another (§7.5)."""
    cost = [
        [0, 2],
        [0, 8],
    ]
    assert assign(cost) == [1, 0]
    assert total(cost, assign(cost)) == 2


def test_the_solver_is_deterministic():
    rng = random.Random(5)
    for _ in range(200):
        cost = random_matrix(rng, max_rows=6, max_cols=8)
        assert assign(cost) == assign(cost)
        assert assign(cost) == assign([list(row) for row in cost])


@pytest.mark.parametrize("size", [1, 2, 5, 10, 25])
def test_a_fully_eligible_square_problem_is_a_perfect_matching(size):
    rng = random.Random(6 + size)
    cost = [[rng.randint(0, 100) for _ in range(size)] for _ in range(size)]
    result = assign(cost)
    assert filled(result) == size
    assert sorted(result) == list(range(size))


def test_it_is_fast_enough_at_realistic_scale():
    """Hundreds of crew against tens of slots is milliseconds (§7.5)."""
    import time

    rng = random.Random(7)
    cost = [[rng.randint(0, 200) for _ in range(400)] for _ in range(40)]
    started = time.perf_counter()
    result = assign(cost)
    elapsed = time.perf_counter() - started

    assert filled(result) == 40
    assert elapsed < 2.0, f"took {elapsed:.2f}s"
