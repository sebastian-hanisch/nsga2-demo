"""Vehikel der NSGA-II-Demo: dieselbe Lieferroute wie genetic-algorithm-demo (Depot + n Kundenstopps, 100 x 100-km-Gebiet,
euklidische Entfernungen) - `xy`- und `co2_factor_matrix`-Erzeugung wortgleich aus `ga_scenario.generate_perm` kopiert, damit die
kleine Vergleichsinstanz (n=8, Seed 19) bitidentisch zur Pareto-Front-Instanz der genetic-algorithm-demo ist. Neu: ein drittes,
unabhängiges Kantenmerkmal Fahrzeit (derselbe Konstruktionstrick wie der CO2-Faktor - manche Straßenabschnitte sind kurz aber
langsam, andere lang aber schnell, unabhängig von Distanz und CO2)."""

from dataclasses import dataclass

import numpy as np

import nsga2_constants as C


@dataclass(frozen=True)
class PermInstance:
    xy: np.ndarray                    # (n + 1, 2); Zeile 0 = Depot
    co2_factor_matrix: np.ndarray     # (n + 1, n + 1); symmetrisch
    time_factor_matrix: np.ndarray    # (n + 1, n + 1); symmetrisch, unabhängig vom CO2-Faktor
    n: int
    cluster_share: int
    seed: int

    @property
    def n_nodes(self):
        return self.n + 1


def generate_perm(n, cluster_share=0, seed=0):
    rng = np.random.default_rng(seed)
    n_grouped = int(round(n * cluster_share / 100))
    uniform = rng.random((n - n_grouped, 2)) * C.AREA
    centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
    which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
    grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
    depot = np.array([[C.AREA / 2, C.AREA / 2]])
    xy = np.vstack([depot, uniform, grouped])
    n_nodes = n + 1
    raw_co2 = rng.uniform(C.CO2_FACTOR_LO, C.CO2_FACTOR_HI, size=(n_nodes, n_nodes))
    co2_factor_matrix = (raw_co2 + raw_co2.T) / 2.0
    np.fill_diagonal(co2_factor_matrix, 0.0)
    raw_time = rng.uniform(C.TIME_FACTOR_LO, C.TIME_FACTOR_HI, size=(n_nodes, n_nodes))
    time_factor_matrix = (raw_time + raw_time.T) / 2.0
    np.fill_diagonal(time_factor_matrix, 0.0)
    return PermInstance(xy, co2_factor_matrix, time_factor_matrix, n, int(cluster_share), int(seed))
