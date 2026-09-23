"""Auswertung der NSGA-II-Demo: ein Lauf, Sweep über Populationsgröße/Generationen (2D-Hypervolumen als Kennzahl), und zwei
Experimente - Abdeckung der Brute-Force-Pareto-Front im Vergleich zur gewichteten Summe der genetic-algorithm-demo (schließt deren
eigenen Befund ab), und Abdeckung bei 2 vs. 3 Zielen (bereitet NSGA-IIIs Motivation vor)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import nsga2_algorithm as A
import nsga2_constants as C
import nsga2_scenario as S

BRUTE_FORCE_MAX_N = 9      # (n_nodes - 1)! Touren; bei 9 Knoten sind das 40 320


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    n_obj: int = C.DEFAULT_N_OBJ
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    cx: float = C.DEFAULT_CX
    mut: float = C.DEFAULT_MUT
    k: int = C.DEFAULT_K
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate_perm(n, cluster_share, seed)
    return inst, A.dist_matrix(inst.xy)


def objective_fn(settings):
    """(Zielfunktion, Knotenzahl) für `settings`. Zielfunktion: Population (P, n_nodes) -> Objektive (P, n_obj), niedriger ist besser."""
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)

    def fn(pop):
        dist = A.tour_length_batch(pop, D)
        co2 = A.tour_edge_cost_batch(pop, D, inst.co2_factor_matrix)
        if settings.n_obj == 2:
            return np.stack([dist, co2], axis=1)
        time = A.tour_edge_cost_batch(pop, D, inst.time_factor_matrix)
        return np.stack([dist, co2, time], axis=1)
    return fn, inst.n_nodes


def run(settings, keep_history=False):
    fn, n_nodes = objective_fn(settings)
    return A.run_nsga2(fn, n_nodes, settings.pop, settings.gens, settings.cx, settings.mut, settings.k, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object


def analyse(settings, keep_history=True):
    return Analysis(settings, run(settings, keep_history=keep_history))


# --- Brute-Force-Referenz (kleine Instanzen) -----------------------------------------------------------------------------------------------


def brute_force_front(n_nodes, fn):
    """Alle (n_nodes - 1)! Touren (Knoten 0 fest an erster Stelle). Gibt (alle Objektive - mit Duplikaten, mehrere Touren können
    denselben Zielwert haben, (Distanz, CO2) hängt nicht an der Tour-Identität -, wahre Pareto-Front als EINDEUTIGE nicht-dominierte
    Zielwerte) zurück. Dedupliziert vor der Sortierung, sonst zählt ein zweimal erreichter Frontwert doppelt (anders als bei der
    genetic-algorithm-demo, deren Front ebenfalls über eindeutige Werte zählt). Nutzt `non_dominated_mask` (O(N * F)), nicht die
    volle (N, N)-Dominanzmatrix - bei 9 Knoten sind das 40 320 Touren, eine volle Matrix (1.6 Milliarden Paare) wäre zu
    langsam/speicherhungrig."""
    tours = np.array([(0,) + p for p in permutations(range(1, n_nodes))], dtype=np.int64)
    obj = fn(tours)
    unique_obj = np.unique(obj, axis=0)
    nd_mask = A.non_dominated_mask(unique_obj)
    return obj, unique_obj[nd_mask]


def front_coverage(true_front_obj, found_obj, tol=1e-6):
    """Wie viele Punkte von `true_front_obj` (bis auf Fließkomma-Toleranz) in `found_obj` vorkommen."""
    reached = 0
    for point in true_front_obj:
        if np.any(np.all(np.abs(found_obj - point) < tol, axis=1)):
            reached += 1
    return reached, len(true_front_obj)


# --- Experiment 1: schließt genetic-algorithm-demos Cliffhanger ----------------------------------------------------------------------------


def comparison_experiment(n=None, seed=None, pop=None, gens=None, seeds=None):
    n = C.COMPARISON_N if n is None else n
    seed = C.COMPARISON_VEHICLE_SEED if seed is None else seed
    pop = C.COMPARISON_POP if pop is None else pop
    gens = C.COMPARISON_GENS if gens is None else gens
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds

    s = Settings(n=n, seed=seed, n_obj=2, pop=pop, gens=gens)
    fn, n_nodes = objective_fn(s)
    _, true_front = brute_force_front(n_nodes, fn)

    reached_counts = []
    for run_seed in seeds:
        r = run(replace(s, run_seed=run_seed), keep_history=False)
        reached, _ = front_coverage(true_front, r.front1)
        reached_counts.append(reached)
    return {
        "front_size": len(true_front),
        "reached_median": float(np.median(reached_counts)),
        "reached_best": int(np.max(reached_counts)),
        "reached_worst": int(np.min(reached_counts)),
        "reached_all": reached_counts,
        "ga_weighted_sum_reached": C.GA_WEIGHTED_SUM_REACHED,
        "ga_weighted_sum_front_size": C.GA_WEIGHTED_SUM_FRONT_SIZE,
    }


# --- Experiment 2: Abdeckung bei 2 vs. 3 Zielen (bereitet NSGA-III vor) --------------------------------------------------------------------


def objective_count_experiment(n=None, seed=None, pop=None, gens=None, seeds=None):
    n = C.COMPARISON_N if n is None else n
    seed = C.COMPARISON_VEHICLE_SEED if seed is None else seed
    pop = C.OBJCOUNT_POP if pop is None else pop
    gens = C.OBJCOUNT_GENS if gens is None else gens
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds

    rows = []
    for n_obj in C.N_OBJ_OPTIONS:
        s = Settings(n=n, seed=seed, n_obj=n_obj, pop=pop, gens=gens)
        fn, n_nodes = objective_fn(s)
        _, true_front = brute_force_front(n_nodes, fn)
        reached_counts = []
        for run_seed in seeds:
            r = run(replace(s, run_seed=run_seed), keep_history=False)
            reached, _ = front_coverage(true_front, r.front1)
            reached_counts.append(reached)
        rows.append({
            "n_obj": n_obj, "front_size": len(true_front),
            "reached_median": float(np.median(reached_counts)),
            "coverage_share": float(np.median(reached_counts)) / len(true_front),
        })
    return rows


# --- Sweep: Populationsgröße/Generationen vs. 2D-Hypervolumen -----------------------------------------------------------------------------


@lru_cache(maxsize=64)
def hypervolume_reference(n, cluster_share, seed, sample=2000, rng_seed=0):
    """Referenzpunkt für das 2D-Hypervolumen dieser Instanz: 1.1 * schlechtester Wert je Ziel über eine Zufallsstichprobe von Touren."""
    inst, D = instance(n, cluster_share, seed)
    rng = np.random.default_rng(rng_seed)
    tours = np.array([rng.permutation(inst.n_nodes) for _ in range(sample)])
    dist = A.tour_length_batch(tours, D)
    co2 = A.tour_edge_cost_batch(tours, D, inst.co2_factor_matrix)
    return (float(dist.max()) * C.HYPERVOLUME_REFERENCE_MARGIN, float(co2.max()) * C.HYPERVOLUME_REFERENCE_MARGIN)


def hypervolume_2d(front_obj, reference):
    """Exaktes dominiertes 2D-Volumen (Fläche) einer nicht-dominierten Front gegen einen schlechteren Referenzpunkt."""
    if len(front_obj) == 0:
        return 0.0
    pts = front_obj[np.argsort(front_obj[:, 0])]
    hv = 0.0
    for i in range(len(pts)):
        x, y = pts[i]
        next_x = pts[i + 1, 0] if i + 1 < len(pts) else reference[0]
        width, height = next_x - x, reference[1] - y
        if width > 0 and height > 0:
            hv += width * height
    return float(hv)


def run_config(base, seeds=None, **changes):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **changes)
    ref = hypervolume_reference(s0.n, s0.cluster_share, s0.seed)
    hvs, front_sizes = [], []
    for run_seed in seeds:
        r = run(replace(s0, run_seed=run_seed), keep_history=False)
        hvs.append(hypervolume_2d(r.front1, ref))
        front_sizes.append(len(r.front1))
    return {"hypervolume": float(np.mean(hvs)), "front_size": float(np.mean(front_sizes))}


def sweep(param, base=None, values=None):
    base = Settings(n_obj=2) if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]
