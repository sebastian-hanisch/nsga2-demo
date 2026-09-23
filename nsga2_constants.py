"""Konstanten der NSGA-II-Demo: Vehikel (wie genetic-algorithm-demo, plus drittes Ziel Fahrzeit), NSGA-II-Regler, Presets
(Presets folgen nach den Messungen)."""

# --- Vehikel: Lieferroute (wie genetic-algorithm-demo) -------------------------------------------------------------------------------------

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0
CLUSTER_MARGIN = 12.0
N_MIN, N_MAX, DEFAULT_N, N_STEP = 8, 100, 30, 2     # N_MIN=8 deckt die "Kleine Instanz"-Vergleichsinstanz (COMPARISON_N) mit ab
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25

# CO2- und Fahrzeit-Faktor je Straßenabschnitt (Kante), unabhängig voneinander und von der Distanz gezogen -
# CO2-Bereich bewusst identisch zu genetic-algorithm-demo, damit die kleine Vergleichsinstanz (n=8, Seed 19)
# bitidentisch ist und GAs eigener Befund (gewichtete Summe trifft 4 von 8 Frontpunkten) direkt zitierbar bleibt.
CO2_FACTOR_LO, CO2_FACTOR_HI = 0.6, 3.4
TIME_FACTOR_LO, TIME_FACTOR_HI = 0.6, 3.4
CO2_HIGHLIGHT_PERCENTILE = 75

# --- NSGA-II ---------------------------------------------------------------------------------------------------------------------------------

N_OBJ_OPTIONS = (2, 3)
N_OBJ_LABELS = {2: "2 (Distanz, CO2)", 3: "3 (Distanz, CO2, Fahrzeit)"}
DEFAULT_N_OBJ = 2

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 10, 200, 60, 10
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 150, 10
CX_MIN, CX_MAX, DEFAULT_CX, CX_STEP = 0.0, 1.0, 0.9, 0.05
MUT_MIN, MUT_MAX, DEFAULT_MUT, MUT_STEP = 0.0, 1.0, 0.2, 0.05
K_MIN, K_MAX, DEFAULT_K, K_STEP = 2, 8, 2, 1        # NSGA-II verwendet klassisch Binärturniere (k=2)
SEED_MAX = 999999
DEFAULT_SEED = 35                # Vehikel-Seed (wie genetic-algorithm-demo)
DEFAULT_RUN_SEED = 7             # Seed des NSGA-II-Laufs (Startpopulation, Turniere, Crossover, Mutation)

# --- Kleine Vergleichsinstanz: identisch zur Pareto-Front-Instanz der genetic-algorithm-demo ------------------------------------------------

COMPARISON_N = 8
COMPARISON_VEHICLE_SEED = 19
GA_WEIGHTED_SUM_FRONT_SIZE = 8    # gemessen in genetic-algorithm-demo (ga_constants.PARETO_N=8, seed=19)
GA_WEIGHTED_SUM_REACHED = 4       # gemessen in genetic-algorithm-demo (test_pareto_experiment_headline_claims)
COMPARISON_SEEDS = tuple(range(600000, 600010))     # NSGA-II-Läufe für die Abdeckungs-Mehrheitsmessung
COMPARISON_POP, COMPARISON_GENS = 60, 150

# Eigenes, knappes Budget für das 2-vs-3-Ziele-Experiment: bei großzügigem Budget (COMPARISON_POP) deckt NSGA-II beide
# Frontgrößen ab (gemessen: 3 Ziele dort sogar BESSER als 2), der Effekt zeigt sich erst unter Druck - siehe README.
OBJCOUNT_POP, OBJCOUNT_GENS = 10, 150

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------

HYPERVOLUME_REFERENCE_MARGIN = 1.10    # Referenzpunkt = 1.10 * (max Distanz, max CO2) der Startpopulation

SWEEP_SEEDS = tuple(range(700000, 700005))
SWEEP_VALUES = {"pop": (10, 20, 40, 60, 100, 150), "gens": (20, 50, 100, 150, 250, 400)}
SWEEP_LABELS = {"pop": "Populationsgröße", "gens": "Generationen"}


def _preset(n=DEFAULT_N, ballung=DEFAULT_BALLUNG, n_obj=DEFAULT_N_OBJ, pop=DEFAULT_POP, gens=DEFAULT_GEN, cx=DEFAULT_CX, mut=DEFAULT_MUT, k=DEFAULT_K, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"n": n, "ballung": ballung, "n_obj": n_obj, "seed": seed, "pop": pop, "gens": gens, "cx": cx, "mut": mut, "k": k, "run_seed": run_seed}


PRESETS = {
    "Standardfall (2 Ziele)": _preset(),
    "Drei Ziele (+ Fahrzeit)": _preset(n_obj=3),
    "Kleine Population": _preset(pop=15),
    "Große Population": _preset(pop=150),
    "Kleine Instanz (Vergleich mit Brute-Force)": _preset(n=COMPARISON_N, seed=COMPARISON_VEHICLE_SEED, pop=COMPARISON_POP, gens=COMPARISON_GENS),
}
# Gemessen mit dem jeweiligen Preset-Seed (siehe PRESETS) - ein einzelner Lauf, kein Mittel über mehrere Seeds
# (Ausnahme: der Verweis auf das Experiment weiter unten in der App, das selbst über mehrere Seeds mittelt).
PRESET_HELP = {
    "Standardfall (2 Ziele)": "30 Lieferstopps, Populationsgröße 60, 150 Generationen: Front 1 wächst von 1 (Startpopulation) auf alle 60 Individuen - am Ende ist die gesamte Population nicht-dominiert. 2D-Hypervolumen (dominierte Fläche gegen einen festen Referenzpunkt): 6 759 025.",
    "Drei Ziele (+ Fahrzeit)": "Dieselbe Instanz, jetzt mit dem dritten Ziel Fahrzeit: Front 1 wächst ebenfalls auf die volle Population (60 von 60) - bei diesem großzügigen Budget zeigt sich noch keine Schwäche durch das dritte Ziel (siehe Experiment „Wird die Abdeckung mit mehr Zielen schwächer?“ weiter unten für die Bedingungen, unter denen sie doch sichtbar wird).",
    "Kleine Population": "Nur 15 Individuen: auch hier wird am Ende die ganze Population nicht-dominiert (15 von 15), aber das 2D-Hypervolumen ist mit 5 662 249 rund 16 % kleiner als im Standardfall (6 759 025) - eine kleinere Population deckt eine kleinere Fläche des Zielraums ab, auch wenn sie intern vollständig konvergiert.",
    "Große Population": "150 statt 60 Individuen: das 2D-Hypervolumen wächst um rund 9 % (7 372 915 gegenüber 6 759 025 im Standardfall) - mehr Individuen erkunden eine breitere Front.",
    "Kleine Instanz (Vergleich mit Brute-Force)": "Dieselbe kleine Instanz (8 Stopps) wie das Experiment „Trifft NSGA-II die Pareto-Front besser als eine feste Gewichtung?“ weiter unten - dort trifft NSGA-II im Median 6 von 8 Frontpunkten, eine feste Gewichtung nur 4.",
}
