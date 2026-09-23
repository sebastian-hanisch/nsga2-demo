"""Vehikel (Reproduzierbarkeit, bitidentisch zur genetic-algorithm-demo für xy/CO2) und Auswertung (Objektive, Brute-Force-Front,
Abdeckung, Hypervolumen, Sweep, beide Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import nsga2_algorithm as A
import nsga2_constants as C
import nsga2_evaluation as E
import nsga2_scenario as S


def test_generate_perm_is_reproducible_and_shaped():
    a = S.generate_perm(20, cluster_share=30, seed=7)
    b = S.generate_perm(20, cluster_share=30, seed=7)
    assert np.array_equal(a.xy, b.xy) and np.array_equal(a.co2_factor_matrix, b.co2_factor_matrix) and np.array_equal(a.time_factor_matrix, b.time_factor_matrix)
    assert a.xy.shape == (21, 2) and a.co2_factor_matrix.shape == (21, 21) and a.time_factor_matrix.shape == (21, 21)
    assert not np.array_equal(a.co2_factor_matrix, a.time_factor_matrix)      # unabhängig gezogen


def test_co2_and_time_factor_matrices_are_symmetric_zero_diagonal_in_range():
    inst = S.generate_perm(30, 0, seed=1)
    for m, lo, hi in ((inst.co2_factor_matrix, C.CO2_FACTOR_LO, C.CO2_FACTOR_HI), (inst.time_factor_matrix, C.TIME_FACTOR_LO, C.TIME_FACTOR_HI)):
        assert np.array_equal(m, m.T)
        assert np.all(np.diag(m) == 0.0)
        off_diag = m[~np.eye(len(m), dtype=bool)]
        assert off_diag.min() >= lo and off_diag.max() <= hi


def test_generate_perm_matches_genetic_algorithm_demo_bit_for_bit_on_the_comparison_instance():
    """Die kleine Vergleichsinstanz (n=8, Seed 19) muss xy und co2_factor_matrix bitidentisch zur genetic-algorithm-demo liefern,
    damit `C.GA_WEIGHTED_SUM_REACHED`/`GA_WEIGHTED_SUM_FRONT_SIZE` dieselbe Instanz zitieren. Reproduziert ga_scenario.generate_perm
    hier lokal (kein Cross-Repo-Import, wie überall im Portfolio), um Abweichungen sofort zu bemerken."""
    def ga_generate_perm(n, cluster_share, seed):
        rng = np.random.default_rng(seed)
        n_grouped = int(round(n * cluster_share / 100))
        uniform = rng.random((n - n_grouped, 2)) * C.AREA
        centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
        which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
        grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
        depot = np.array([[C.AREA / 2, C.AREA / 2]])
        xy = np.vstack([depot, uniform, grouped])
        n_nodes = n + 1
        raw = rng.uniform(C.CO2_FACTOR_LO, C.CO2_FACTOR_HI, size=(n_nodes, n_nodes))
        co2 = (raw + raw.T) / 2.0
        np.fill_diagonal(co2, 0.0)
        return xy, co2

    xy_ref, co2_ref = ga_generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    inst = S.generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    assert np.array_equal(inst.xy, xy_ref)
    assert np.array_equal(inst.co2_factor_matrix, co2_ref)


def test_comparison_instance_front_size_matches_genetic_algorithm_demos_measurement():
    """Regressionstest für einen echten Fund: die erste Fassung zählte Touren statt eindeutiger Zielwerte und meldete
    9 statt der von der genetic-algorithm-demo gemessenen 8 Frontpunkte (zwei verschiedene Touren erreichen denselben
    (Distanz, CO2)-Wert)."""
    s = E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, n_obj=2)
    fn, n_nodes = E.objective_fn(s)
    _, front = E.brute_force_front(n_nodes, fn)
    assert len(front) == C.GA_WEIGHTED_SUM_FRONT_SIZE == 8


def test_objective_fn_shapes_and_matches_manual_computation():
    s = E.Settings(n=5, n_obj=2, seed=1, pop=10, gens=5)
    fn, n_nodes = E.objective_fn(s)
    pop = np.array([np.arange(n_nodes)])
    obj = fn(pop)
    assert obj.shape == (1, 2)
    inst, D = E.instance(s.n, s.cluster_share, s.seed)
    dist = A.tour_length_batch(pop, D)
    co2 = A.tour_edge_cost_batch(pop, D, inst.co2_factor_matrix)
    assert obj[0, 0] == pytest.approx(dist[0]) and obj[0, 1] == pytest.approx(co2[0])

    s3 = E.Settings(n=5, n_obj=3, seed=1, pop=10, gens=5)
    fn3, _ = E.objective_fn(s3)
    obj3 = fn3(pop)
    assert obj3.shape == (1, 3)
    assert obj3[0, 0] == pytest.approx(dist[0]) and obj3[0, 1] == pytest.approx(co2[0])


def test_brute_force_front_is_non_dominated_complete_and_deduplicated_at_small_n():
    s = E.Settings(n=6, n_obj=2, seed=3)
    fn, n_nodes = E.objective_fn(s)
    all_obj, front = E.brute_force_front(n_nodes, fn)
    assert len(all_obj) == 720          # 6! Touren
    assert len(np.unique(front, axis=0)) == len(front)      # keine doppelten Zielwerte in der Front
    dom = A.dominance_matrix(front)
    assert not dom.any()                # keine Front-Punkte dominieren sich gegenseitig


def test_front_coverage_counts_exact_matches_only():
    true_front = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0]])
    found = np.array([[1.0, 5.0], [2.0, 3.0], [9.0, 9.0]])
    reached, total = E.front_coverage(true_front, found)
    assert reached == 2 and total == 3


def test_hypervolume_2d_matches_hand_calculation():
    front = np.array([[1.0, 5.0], [3.0, 2.0]])
    hv = E.hypervolume_2d(front, reference=(5.0, 6.0))
    assert hv == pytest.approx(10.0)    # siehe Handrechnung in der README/im Plan: 2*1 + 2*4


def test_hypervolume_2d_empty_front_is_zero():
    assert E.hypervolume_2d(np.empty((0, 2)), reference=(1.0, 1.0)) == 0.0


def test_run_config_and_sweep_smoke():
    rows = E.sweep("gens", base=E.Settings(n=15, n_obj=2, pop=20), values=(10, 30))
    assert len(rows) == 2
    assert all(r["hypervolume"] >= 0 for r in rows)


def test_comparison_experiment_smoke_small():
    report = E.comparison_experiment(n=6, seed=3, pop=20, gens=30, seeds=(1, 2))
    assert report["front_size"] >= 1
    assert 0 <= report["reached_median"] <= report["front_size"]
    assert report["ga_weighted_sum_reached"] == C.GA_WEIGHTED_SUM_REACHED


def test_objective_count_experiment_smoke_small():
    rows = E.objective_count_experiment(n=6, seed=3, pop=20, gens=30, seeds=(1, 2))
    assert {r["n_obj"] for r in rows} == set(C.N_OBJ_OPTIONS)
    for r in rows:
        assert 0.0 <= r["coverage_share"] <= 1.0
