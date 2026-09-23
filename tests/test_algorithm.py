"""Kern gegen Handrechnung + Bibliotheksgegenprobe (pymoo): nicht-dominierte Sortierung und Crowding-Distance stimmen exakt mit
pymoo überein; Order Crossover/Tausch-Mutation wie in genetic-algorithm-demo; NSGA-II findet auf sehr kleinen Instanzen
nachweislich die Mehrheit der Brute-Force-Pareto-Front."""

from itertools import permutations

import numpy as np
import pytest
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance as pymoo_crowding_distance
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort as pymoo_sort

import nsga2_algorithm as A


# --- Handrechnung: 6 Punkte, 2 Ziele, Fronten und Crowding-Distance von Hand bestimmt ------------------------------------------------------

HAND_POINTS = np.array([
    [1.0, 5.0],   # 0 - Front 1
    [2.0, 3.0],   # 1 - Front 1
    [4.0, 1.0],   # 2 - Front 1
    [2.0, 5.0],   # 3 - Front 2 (dominiert von 0 und 1)
    [3.0, 3.0],   # 4 - Front 2 (dominiert von 1)
    [5.0, 2.0],   # 5 - Front 2 (dominiert von 2)
])


def test_fast_non_dominated_sort_matches_hand_calculation():
    fronts, rank = A.fast_non_dominated_sort(HAND_POINTS)
    assert sorted(fronts[0].tolist()) == [0, 1, 2]
    assert sorted(fronts[1].tolist()) == [3, 4, 5]
    assert rank.tolist() == [0, 0, 0, 1, 1, 1]


def test_crowding_distance_matches_hand_calculation():
    front0 = HAND_POINTS[[0, 1, 2]]     # dieselbe Reihenfolge wie im Front-Array
    cd = A.crowding_distance(front0)
    assert cd[0] == np.inf and cd[2] == np.inf     # Randindividuen je Ziel
    assert cd[1] == pytest.approx(1.0)              # ((4-1)/3 + (5-1)/4) / 2 Ziele = (1.0 + 1.0) / 2


def test_dominates_matches_hand_examples():
    assert A.dominates((1, 5), (2, 5))
    assert A.dominates((2, 3), (2, 5))
    assert not A.dominates((2, 5), (3, 3))
    assert not A.dominates((3, 3), (2, 5))
    assert not A.dominates((1, 1), (1, 1))    # ein Punkt dominiert sich selbst nicht (keine strikte Verbesserung)


# --- Bibliotheksgegenprobe: pymoo ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_fast_non_dominated_sort_matches_pymoo(seed):
    rng = np.random.default_rng(seed)
    n_obj = int(rng.integers(2, 4))
    F = rng.random((25, n_obj)) * 100
    fronts_mine, _ = A.fast_non_dominated_sort(F)
    fronts_pymoo = pymoo_sort(F)
    assert [sorted(f.tolist()) for f in fronts_mine] == [sorted(f) for f in fronts_pymoo]


@pytest.mark.parametrize("seed", range(20))
def test_crowding_distance_matches_pymoo(seed):
    rng = np.random.default_rng(seed)
    n_obj = int(rng.integers(2, 4))
    F = rng.random((25, n_obj)) * 100
    fronts, _ = A.fast_non_dominated_sort(F)
    for front in fronts:
        if len(front) < 3:
            continue
        cd_mine = A.crowding_distance(F[front])
        cd_pymoo = pymoo_crowding_distance(F[front])
        np.testing.assert_allclose(np.sort(cd_mine), np.sort(cd_pymoo), rtol=1e-9)


# --- Operatoren (wortgleich zu genetic-algorithm-demo) ----------------------------------------------------------------------------------


def test_order_crossover_matches_hand_calculation():
    class _FixedRNG:
        def integers(self, lo, hi, size=None):
            return np.array([3, 6])
    p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    p2 = np.array([5, 4, 6, 9, 2, 1, 7, 8, 3])
    child = A.order_crossover(p1, p2, _FixedRNG())
    assert child.tolist() == [9, 2, 1, 4, 5, 6, 7, 8, 3]


@pytest.mark.parametrize("seed", range(20))
def test_order_crossover_always_produces_valid_permutation(seed):
    rng = np.random.default_rng(seed)
    n = 12
    p1, p2 = rng.permutation(n), rng.permutation(n)
    child = A.order_crossover(p1, p2, rng)
    assert sorted(child.tolist()) == list(range(n))


def test_swap_mutation_probability_zero_never_changes():
    rng = np.random.default_rng(0)
    ind = np.arange(10)
    for _ in range(20):
        assert np.array_equal(A.swap_mutation(ind, 0.0, rng), ind)


# --- Selektion / Überlebensauswahl --------------------------------------------------------------------------------------------------------


def test_crowded_compare_hand_cases():
    assert A.crowded_compare(0, 1.0, 1, 5.0) is True         # niedrigerer Rang gewinnt immer
    assert A.crowded_compare(1, 5.0, 0, 1.0) is False
    assert A.crowded_compare(0, 5.0, 0, 1.0) is True          # bei Rang-Gleichstand höhere Crowding-Distance
    assert A.crowded_compare(0, 1.0, 0, 1.0) is False         # exakt gleich -> a ist nicht besser als b


def test_crowded_tournament_picks_the_unique_best_far_more_than_chance():
    # Kandidaten werden MIT Zurücklegen gezogen (rng.integers) - auch bei k = Populationsgröße ist die Aufnahme des
    # Besten nicht garantiert (P = 1 - ((n-1)/n)^k), aber deutlich wahrscheinlicher als 1/n reiner Zufall.
    rank = np.array([1, 1, 0, 1, 1])
    crowding = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    selected = A.crowded_tournament_select(rank, crowding, k=len(rank), n_select=2000, rng=rng)
    assert (selected == 2).mean() > 0.5          # Zufallsniveau wäre 0.2; erwartet ~0.67


def test_crowded_tournament_tie_break_by_crowding_far_more_than_chance():
    rank = np.array([0, 0, 0])
    crowding = np.array([1.0, np.inf, 1.0])
    rng = np.random.default_rng(0)
    selected = A.crowded_tournament_select(rank, crowding, k=3, n_select=2000, rng=rng)
    assert (selected == 1).mean() > 0.5          # Zufallsniveau wäre 1/3; erwartet ~0.70


def test_select_survivors_fills_fronts_and_truncates_by_crowding():
    objectives = HAND_POINTS
    idx, rank_out, crowding_out = A.select_survivors(objectives, pop_size=4)
    assert set(idx.tolist()) >= {0, 1, 2}          # Front 1 (3 Punkte) passt vollständig hinein
    assert len(idx) == 4
    assert np.all(rank_out[np.isin(idx, [0, 1, 2])] == 0)
    fourth = idx[~np.isin(idx, [0, 1, 2])][0]
    assert fourth in (3, 4, 5)                      # ein Punkt aus Front 2 ergänzt


def test_select_survivors_keeps_exact_pop_size_across_many_sizes():
    rng = np.random.default_rng(1)
    objectives = rng.random((40, 3)) * 100
    for pop_size in (1, 5, 20, 40):
        idx, rank_out, crowding_out = A.select_survivors(objectives, pop_size)
        assert len(idx) == pop_size == len(set(idx.tolist()))


# --- NSGA-II-Lauf auf einer sehr kleinen Instanz: nachweislich die Mehrheit der Brute-Force-Front -------------------------------------------


def _brute_force_front(n_nodes, fn):
    tours = np.array([(0,) + p for p in permutations(range(1, n_nodes))])
    obj = fn(tours)
    dom = A.dominance_matrix(obj)
    return obj[~dom.any(axis=0)]


def test_nsga2_finds_most_of_the_brute_force_front_on_a_tiny_instance():
    rng = np.random.default_rng(42)
    xy = rng.random((6, 2)) * 100.0
    D = A.dist_matrix(xy)
    raw = rng.uniform(0.6, 3.4, size=(6, 6))
    factor = (raw + raw.T) / 2.0
    np.fill_diagonal(factor, 0.0)

    def fn(pop):
        dist = A.tour_length_batch(pop, D)
        co2 = A.tour_edge_cost_batch(pop, D, factor)
        return np.stack([dist, co2], axis=1)

    true_front = _brute_force_front(6, fn)
    hits = []
    for seed in range(10):
        r = A.run_nsga2(fn, 6, pop_size=30, generations=60, cx_prob=0.9, mut_prob=0.2, tournament_k=2, seed=seed)
        found = r.front1
        reached = sum(np.any(np.all(np.abs(found - p) < 1e-6, axis=1)) for p in true_front)
        hits.append(reached / len(true_front))
    assert np.median(hits) >= 0.8
