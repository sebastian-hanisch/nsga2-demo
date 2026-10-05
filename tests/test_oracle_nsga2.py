"""Orakel-Tests (unabhängiger Rechenweg): nicht-dominierte Sortierung gegen wiederholtes Abschälen der Fronten, Crowding-
Distance per Schleife und von Hand, Überlebensauswahl und Turnier gegen die Definition, 2D-Hypervolumen gegen eine
Rechteckvereinigung (und pymoo, falls vorhanden), Brute-Force-Front gegen eine Aufzählung mit Rundungs-Deduplikation
(Regression: gespiegelte Touren erzeugten bei drei Zielen einen Scheinpunkt)."""

import itertools

import numpy as np
import pytest

import nsga2_algorithm as A
import nsga2_evaluation as E


def _dom(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _peel(F):
    n, rank, rem, r = len(F), [-1] * len(F), set(range(len(F))), 0
    while rem:
        cur = [i for i in rem if not any(_dom(F[j], F[i]) for j in rem if j != i)]
        for i in cur:
            rank[i] = r
        rem -= set(cur)
        r += 1
    return rank


def _ref_cd(fr):
    m, n_obj = len(fr), len(fr[0])
    if m <= 2:
        return [float("inf")] * m
    d = [0.0] * m
    for k in range(n_obj):
        order = sorted(range(m), key=lambda i: fr[i][k])
        lo, hi = fr[order[0]][k], fr[order[-1]][k]
        d[order[0]] = d[order[-1]] = float("inf")
        if hi - lo < 1e-9:
            continue
        for p in range(1, m - 1):
            d[order[p]] += (fr[order[p + 1]][k] - fr[order[p - 1]][k]) / (hi - lo)
    return [x / n_obj for x in d]


def test_sort_and_masks_match_front_peeling():
    rng = np.random.default_rng(3)
    for k in range(80):
        n, m = int(rng.integers(1, 25)), int(rng.integers(2, 5))
        F = rng.integers(0, 6, size=(n, m)).astype(float) if k % 2 else rng.random((n, m))
        fronts, rank = A.fast_non_dominated_sort(F)
        ref = _peel(F.tolist())
        assert rank.tolist() == ref
        assert [sorted(f.tolist()) for f in fronts] == [sorted(i for i in range(n) if ref[i] == r) for r in range(max(ref) + 1)]
        assert A.non_dominated_mask(F).tolist() == [r == 0 for r in ref]


def test_crowding_distance_matches_loop_and_hand_examples():
    cd = A.crowding_distance(np.array([[0., 8.], [1., 7.], [2., 4.], [4., 0.]]))
    assert np.isinf(cd[0]) and np.isinf(cd[3])
    assert cd[1] == pytest.approx(0.5) and cd[2] == pytest.approx(0.8125)     # (0,5 + 0,5)/2 bzw. (0,75 + 0,875)/2
    rng = np.random.default_rng(4)
    for _ in range(60):
        fr = rng.random((int(rng.integers(1, 12)), int(rng.integers(2, 5)))) * 100
        assert np.allclose(np.nan_to_num(A.crowding_distance(fr), posinf=1e18), np.nan_to_num(_ref_cd(fr.tolist()), posinf=1e18))


def test_survivors_and_tournament_match_definition():
    rng = np.random.default_rng(5)
    for _ in range(60):
        n, m = int(rng.integers(4, 30)), int(rng.integers(2, 4))
        p = int(rng.integers(1, n))
        F = rng.random((n, m)) * 50
        idx, rk, _ = A.select_survivors(F, p)
        ref = _peel(F.tolist())
        sel, cut = set(idx.tolist()), max(ref[i] for i in idx)
        assert len(sel) == p and rk.tolist() == [ref[i] for i in idx]
        assert all((i in sel) == (ref[i] < cut) for i in range(n) if ref[i] != cut)
        front = [i for i in range(n) if ref[i] == cut]
        cdf = dict(zip(front, _ref_cd([F[i].tolist() for i in front])))
        outs = [cdf[i] for i in front if i not in sel]
        if outs:
            assert min(cdf[i] for i in front if i in sel) >= max(outs) - 1e-12
    for s in range(30):
        n, kk = int(rng.integers(3, 15)), int(rng.integers(2, 6))
        rank, cr = rng.integers(0, 3, size=n), rng.random(n)
        got = A.crowded_tournament_select(rank, cr, kk, 8, np.random.default_rng(s))
        r2 = np.random.default_rng(s)
        for g in got:
            cand = r2.integers(0, n, size=kk)
            best = cand[0]
            for c in cand[1:]:
                if (rank[c], -cr[c]) < (rank[best], -cr[best]):
                    best = c
            assert g == best


def _hv_union(pts, ref):
    pts = [p for p in pts if p[0] < ref[0] and p[1] < ref[1]]
    xs = sorted({p[0] for p in pts} | {ref[0]})
    tot = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        ys = [p[1] for p in pts if p[0] <= a]
        if ys:
            tot += (b - a) * (ref[1] - min(ys))
    return tot


def test_hypervolume_2d_matches_rectangle_union():
    assert E.hypervolume_2d(np.array([[1., 3.], [2., 2.], [3., 1.]]), (4., 4.)) == 6.0        # 1*1 + 1*2 + 1*3
    assert E.hypervolume_2d(np.empty((0, 2)), (4., 4.)) == 0.0
    rng = np.random.default_rng(6)
    for k in range(80):
        pts = rng.random((int(rng.integers(1, 15)), 2)) * 100
        if k % 3 == 0:
            pts = np.round(pts / 10) * 10                       # Gleichstände in einem Ziel
        nd = pts[A.non_dominated_mask(pts)]
        assert E.hypervolume_2d(nd, (110., 110.)) == pytest.approx(_hv_union(pts.tolist(), (110., 110.)))


def test_hypervolume_2d_matches_pymoo_if_available():
    hv = pytest.importorskip("pymoo.indicators.hv")
    rng = np.random.default_rng(7)
    for _ in range(20):
        pts = rng.random((int(rng.integers(1, 12)), 2)) * 100
        nd = pts[A.non_dominated_mask(pts)]
        assert E.hypervolume_2d(nd, (110., 110.)) == pytest.approx(hv.HV(ref_point=np.array([110., 110.]))(nd))


@pytest.mark.parametrize("n,seed,n_obj", [(5, 2, 3), (5, 2, 2), (6, 3, 3), (6, 1, 2)])
def test_brute_force_front_matches_enumeration(n, seed, n_obj):
    """n=5, Seed 2, 3 Ziele ist die Regression: ohne Rundungs-Deduplikation hatte die Front 5 statt 4 Punkte."""
    st = E.Settings(n=n, seed=seed, n_obj=n_obj)
    fn, n_nodes = E.objective_fn(st)
    inst, D = E.instance(st.n, st.cluster_share, st.seed)
    factors = [None, inst.co2_factor_matrix] + ([inst.time_factor_matrix] if n_obj == 3 else [])

    def objs(t):
        edges = [(t[i], t[(i + 1) % n_nodes]) for i in range(n_nodes)]
        return tuple(round(float(sum(D[a, b] * (1.0 if f is None else f[a, b]) for a, b in edges)), 6) for f in factors)

    pts = {objs((0,) + p) for p in itertools.permutations(range(1, n_nodes))}
    ref = {p for p in pts if not any(_dom(q, p) for q in pts)}
    _, front = E.brute_force_front(n_nodes, fn)
    assert {tuple(np.round(r, 6)) for r in front} == ref and len(front) == len(ref)


def test_standardfall_front1_counts_duplicate_individuals():
    """Die README-Aussage "Front 1 = alle 60 Individuen" zählt Individuen mit identischer Tour einzeln (gegenseitig dominieren
    sich gleiche Zielwerte nicht): verschiedene Touren gibt es deutlich weniger."""
    import nsga2_constants as C
    p = C.PRESETS["Standardfall (2 Ziele)"]
    r = E.run(E.Settings(n=p["n"], cluster_share=p["ballung"], n_obj=p["n_obj"], seed=p["seed"], pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], k=p["k"], run_seed=p["run_seed"]))
    assert len(r.front1) == p["pop"]
    assert len({tuple(t) for t in r.final_population.tolist()}) < p["pop"] / 3
