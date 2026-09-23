"""NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002), numpy von Grund auf. Order Crossover, Tausch-Mutation und die
Distanz-/Kantenkosten-Hilfen sind wortgleich aus genetic-algorithm-demo/ga_algorithm.py kopiert; neu ist die Mehrziel-Selektion
(schnelle nicht-dominierte Sortierung + Crowding-Distance) anstelle einer Turnierselektion auf einem einzelnen Fitnesswert.

Die nicht-dominierte Sortierung ist über eine volle (N, N)-Dominanzmatrix vektorisiert (numpy-Broadcast), nicht über ein
Python-Doppel-Loop mit einem `dominates()`-Aufruf je Paar - bei Populationsgröße 200 über 400 Generationen wäre Letzteres zu
langsam (N² Python-Funktionsaufrufe je Generation)."""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


# --- Distanz / Kantenkosten (wie genetic-algorithm-demo) -----------------------------------------------------------------------------------


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def tour_length_batch(pop, D):
    nxt = np.roll(pop, -1, axis=1)
    return D[pop, nxt].sum(axis=1)


def tour_edge_cost_batch(pop, D, factor_matrix):
    """Verallgemeinert tour_co2_batch aus genetic-algorithm-demo: Summe aus Distanz * Kantenfaktor - für CO2 UND Fahrzeit nutzbar."""
    nxt = np.roll(pop, -1, axis=1)
    return (D[pop, nxt] * factor_matrix[pop, nxt]).sum(axis=1)


def tour_edges(tour):
    t = np.asarray(tour)
    a, b = t, np.roll(t, -1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}


def edge_share(tour, reference):
    return len(tour_edges(tour) & tour_edges(reference)) / len(tour)


# --- Population: Erzeugen, Crossover, Mutation (wie genetic-algorithm-demo) ------------------------------------------------------------------


def init_population(pop_size, n_nodes, rng):
    return np.array([rng.permutation(n_nodes) for _ in range(pop_size)], dtype=np.int64)


def order_crossover(p1, p2, rng):
    n = len(p1)
    i, j = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[i:j + 1] = p1[i:j + 1]
    taken = set(child[i:j + 1].tolist())
    fill = [g for g in p2.tolist() if g not in taken]
    pos = [k for k in range(n) if not (i <= k <= j)]
    for k, g in zip(pos, fill):
        child[k] = g
    return child


def swap_mutation(ind, p_mut, rng):
    if rng.random() >= p_mut:
        return ind.copy()
    out = ind.copy()
    i, j = rng.integers(0, len(ind), size=2)
    out[i], out[j] = out[j], out[i]
    return out


# --- Mehrziel-Sortierung: schnelle nicht-dominierte Sortierung + Crowding-Distance -------------------------------------------------------


def dominates(a, b):
    """a dominiert b (Minimierung): a <= b in jedem Ziel, a < b in mindestens einem. Einzelpaar-Hilfsfunktion (Tests/Handrechnung);
    die Sortierung selbst nutzt die vektorisierte Dominanzmatrix unten."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def dominance_matrix(objectives):
    """(N, N)-Boolmatrix, vektorisiert: dom[i, j] = i dominiert j. Für die NSGA-II-Population (einige hundert Individuen)
    gedacht - bei sehr großem N (z. B. einer Brute-Force-Enumeration aller Touren) `non_dominated_mask` unten verwenden,
    O(N²) wäre dort zu langsam/speicherhungrig."""
    F = np.asarray(objectives, dtype=float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    return dom


def non_dominated_mask(objectives):
    """Boolmaske der nicht-dominierten Punkte, ohne die volle (N, N)-Matrix zu bilden: nach dem ersten Ziel sortiert,
    gegen eine laufend aktualisierte (meist kleine) Front gefiltert - O(N * F) statt O(N²), F = Frontgröße. Geeignet für
    sehr große N (z. B. alle (n-1)! Touren einer Brute-Force-Enumeration), wo die volle Matrix nicht mehr in den
    Speicher passt bzw. zu langsam wäre."""
    F = np.asarray(objectives, dtype=float)
    order = np.argsort(F[:, 0])
    frontier_idx = []
    frontier_obj = np.empty((0, F.shape[1]))
    for idx in order:
        p = F[idx]
        if len(frontier_obj) and np.any(np.all(frontier_obj <= p, axis=1) & np.any(frontier_obj < p, axis=1)):
            continue                                            # von einem bestehenden Frontpunkt dominiert
        if len(frontier_obj):
            dominated_by_p = np.all(p <= frontier_obj, axis=1) & np.any(p < frontier_obj, axis=1)
            keep = ~dominated_by_p
            frontier_idx = [fi for fi, k in zip(frontier_idx, keep.tolist()) if k]
            frontier_obj = frontier_obj[keep]
        frontier_idx.append(int(idx))
        frontier_obj = np.vstack([frontier_obj, p])
    mask = np.zeros(len(F), dtype=bool)
    mask[frontier_idx] = True
    return mask


def fast_non_dominated_sort(objectives):
    """objectives: (N, M), niedriger ist besser. Gibt (fronts, rank) zurück - fronts: Liste von Index-Arrays (Front 0 = nicht
    dominiert), rank: (N,) Array. Klassischer Algorithmus (Deb et al. 2002), die N x N-Dominanzprüfung ist vektorisiert."""
    n = len(objectives)
    dom = dominance_matrix(objectives)
    domination_count = dom.sum(axis=0).astype(np.int64)     # n_p je Individuum p
    rank = np.full(n, -1, dtype=np.int64)
    current = np.where(domination_count == 0)[0]
    rank[current] = 0
    fronts = [current]
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            dominated_by_p = np.where(dom[p])[0]
            domination_count[dominated_by_p] -= 1
            newly_free = dominated_by_p[domination_count[dominated_by_p] == 0]
            rank[newly_free] = i + 1
            next_front.extend(newly_free.tolist())
        i += 1
        fronts.append(np.array(next_front, dtype=np.int64))
    fronts.pop()
    return fronts, rank


def crowding_distance(front_objectives):
    """Crowding-Distance je Individuum EINER Front (m, M)-Array, durch die Zielzahl M geteilt (wie pymoo) - eine reine
    Reskalierung, ändert keinen Vergleich (crowded_compare, Turnier, Abschneiden), macht die Zahlen aber direkt mit der
    Bibliotheksgegenprobe vergleichbar. Randindividuen je Ziel = unendlich."""
    m, n_obj = front_objectives.shape
    if m <= 2:
        return np.full(m, np.inf)
    dist = np.zeros(m)
    for k in range(n_obj):
        order = np.argsort(front_objectives[:, k])
        vals = front_objectives[order, k]
        spread = vals[-1] - vals[0]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if spread < EPS:
            continue
        dist[order[1:-1]] += (vals[2:] - vals[:-2]) / spread
    return dist / n_obj


def crowding_distance_all(objectives, fronts):
    """Crowding-Distance für ALLE Individuen, frontweise berechnet - (N,) Array in der ursprünglichen Reihenfolge."""
    n = len(objectives)
    dist = np.zeros(n)
    for front in fronts:
        if len(front) == 0:
            continue
        dist[front] = crowding_distance(objectives[front])
    return dist


def crowded_compare(rank_a, cd_a, rank_b, cd_b):
    """True, wenn a besser ist als b (niedrigerer Rang, bei Gleichstand höhere Crowding-Distance)."""
    if rank_a != rank_b:
        return rank_a < rank_b
    return cd_a > cd_b


def crowded_tournament_select(rank, crowding, k, n_select, rng):
    """n_select Eltern-Indizes: je k zufällige Kandidaten antreten lassen, den nach crowded_compare besten wählen."""
    n = len(rank)
    out = np.empty(n_select, dtype=np.int64)
    for s in range(n_select):
        cand = rng.integers(0, n, size=k)
        best = cand[0]
        for c in cand[1:]:
            if crowded_compare(rank[c], crowding[c], rank[best], crowding[best]):
                best = c
        out[s] = best
    return out


def select_survivors(objectives, pop_size):
    """Fronten der Reihe nach auffüllen, letzte passende Front nach Crowding-Distance abschneiden.
    Gibt (Indizes der Überlebenden, ihr Rang, ihre Crowding-Distance) zurück."""
    fronts, rank_all = fast_non_dominated_sort(objectives)
    survivors = []
    crowding_out = np.zeros(len(objectives))
    for front in fronts:
        if len(survivors) + len(front) <= pop_size:
            if len(front) > 0:
                cd = crowding_distance(objectives[front])
                crowding_out[front] = cd
            survivors.extend(front.tolist())
            if len(survivors) == pop_size:
                break
        else:
            remaining = pop_size - len(survivors)
            cd = crowding_distance(objectives[front])
            order = np.argsort(-cd)
            chosen = front[order[:remaining]]
            crowding_out[chosen] = cd[order[:remaining]]
            survivors.extend(chosen.tolist())
            break
    idx = np.array(survivors, dtype=np.int64)
    return idx, rank_all[idx], crowding_out[idx]


# --- NSGA-II-Hauptschleife -----------------------------------------------------------------------------------------------------------------


@dataclass
class Generation:
    population: np.ndarray
    objectives: np.ndarray
    rank: np.ndarray
    crowding: np.ndarray


@dataclass
class NSGA2Result:
    final_population: np.ndarray
    final_objectives: np.ndarray
    final_rank: np.ndarray
    final_crowding: np.ndarray
    front1_history: list = field(default_factory=list)      # Front-1-Objektive je Generation (klein, immer gespeichert)
    generations: list = field(default_factory=list)          # volle Generation-Objekte, nur bei keep_history=True

    @property
    def front1(self):
        return self.final_objectives[self.final_rank == 0]

    @property
    def front1_population(self):
        return self.final_population[self.final_rank == 0]


def run_nsga2(objective_fn, n_nodes, pop_size, generations, cx_prob, mut_prob, tournament_k, seed, keep_history=False):
    """objective_fn(population) -> (pop_size, n_obj), niedriger ist besser je Ziel."""
    rng = np.random.default_rng(seed)
    pop = init_population(pop_size, n_nodes, rng)
    obj = objective_fn(pop)
    fronts, rank = fast_non_dominated_sort(obj)
    crowding = crowding_distance_all(obj, fronts)
    front1_history = [obj[rank == 0].copy()]
    gens = [Generation(pop.copy(), obj.copy(), rank.copy(), crowding.copy())] if keep_history else []

    for _ in range(generations):
        parents = crowded_tournament_select(rank, crowding, tournament_k, pop_size, rng)
        children = []
        for i in range(pop_size):
            p1 = pop[parents[i]]
            p2 = pop[parents[rng.integers(0, pop_size)]]
            child = order_crossover(p1, p2, rng) if rng.random() < cx_prob else p1.copy()
            child = swap_mutation(child, mut_prob, rng)
            children.append(child)
        offspring = np.array(children, dtype=np.int64)
        obj_offspring = objective_fn(offspring)

        combined_pop = np.concatenate([pop, offspring], axis=0)
        combined_obj = np.concatenate([obj, obj_offspring], axis=0)
        idx, rank, crowding = select_survivors(combined_obj, pop_size)
        pop = combined_pop[idx]
        obj = combined_obj[idx]

        front1_history.append(obj[rank == 0].copy())
        if keep_history:
            gens.append(Generation(pop.copy(), obj.copy(), rank.copy(), crowding.copy()))

    return NSGA2Result(pop, obj, rank, crowding, front1_history, gens)
