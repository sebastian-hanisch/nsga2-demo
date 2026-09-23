"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 150-Generationen-NSGA-II-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (welche Route bei
einem Gleichstand gewinnt, kann über viele Generationen hinweg kaskadieren) - dieselbe Ursache wie bei jeder anderen
CI-Linux-vs-Windows-Abweichung in diesem Portfolio (siehe feedback_ci_platform_robust_tests.md). Zahlen aus einem EINZELNEN
Lauf (Presets) bekommen deshalb großzügige Bänder statt enger Toleranzen; Zahlen, die über mehrere Seeds mitteln
(Experimente), sind von Natur aus robuster und behalten engere Bänder."""

import pytest

import nsga2_constants as C
import nsga2_evaluation as E


def _preset_result(name, keep_history=True):
    p = C.PRESETS[name]
    s = E.Settings(n=p["n"], cluster_share=p["ballung"], n_obj=p["n_obj"], seed=p["seed"], pop=p["pop"], gens=p["gens"],
                   cx=p["cx"], mut=p["mut"], k=p["k"], run_seed=p["run_seed"])
    return E.analyse(s, keep_history=keep_history).result


def test_standardfall_preset_claims():
    r = _preset_result("Standardfall (2 Ziele)")
    assert int((r.generations[0].rank == 0).sum()) == 1
    assert len(r.front1) >= 55           # praktisch die ganze Population, Einzellauf: nicht zwingend exakt 60
    p = C.PRESETS["Standardfall (2 Ziele)"]
    ref = E.hypervolume_reference(p["n"], p["ballung"], p["seed"])
    hv = E.hypervolume_2d(r.front1, ref)
    assert hv == pytest.approx(6_759_025, rel=0.03)


def test_drei_ziele_preset_claims():
    r = _preset_result("Drei Ziele (+ Fahrzeit)")
    assert len(r.front1) >= 55


def test_kleine_population_preset_claims():
    r = _preset_result("Kleine Population")
    assert len(r.front1) >= 13           # praktisch die ganze Population (15), Einzellauf
    p = C.PRESETS["Kleine Population"]
    ref = E.hypervolume_reference(p["n"], p["ballung"], p["seed"])
    hv = E.hypervolume_2d(r.front1, ref)
    assert hv == pytest.approx(5_662_249, rel=0.03)
    hv_standard = 6_759_025
    reduction = 100.0 * (hv_standard - hv) / hv_standard
    assert reduction == pytest.approx(16, abs=5)


def test_grosse_population_preset_claims():
    r = _preset_result("Große Population")
    assert len(r.front1) >= 140          # praktisch die ganze Population (150), Einzellauf
    p = C.PRESETS["Große Population"]
    ref = E.hypervolume_reference(p["n"], p["ballung"], p["seed"])
    hv = E.hypervolume_2d(r.front1, ref)
    assert hv == pytest.approx(7_372_915, rel=0.03)
    hv_standard = 6_759_025
    increase = 100.0 * (hv - hv_standard) / hv_standard
    assert increase == pytest.approx(9, abs=5)


def test_kleine_instanz_preset_matches_the_comparison_instance():
    p = C.PRESETS["Kleine Instanz (Vergleich mit Brute-Force)"]
    assert p["n"] == C.COMPARISON_N and p["seed"] == C.COMPARISON_VEHICLE_SEED


# --- Headlinezahlen der beiden Experimente -------------------------------------------------------------------------------------------------


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["front_size"] == 8
    assert report["reached_median"] == pytest.approx(6, abs=0.5)
    assert report["ga_weighted_sum_reached"] == 4


def test_objective_count_experiment_headline_claims():
    rows = {r["n_obj"]: r for r in E.objective_count_experiment()}
    assert rows[2]["front_size"] == 8 and rows[3]["front_size"] == 16
    assert rows[2]["coverage_share"] == pytest.approx(0.5, abs=0.05)
    assert rows[3]["coverage_share"] == pytest.approx(0.375, abs=0.05)
    assert rows[3]["coverage_share"] < rows[2]["coverage_share"]


def test_objective_count_experiment_reverses_under_a_generous_budget():
    """Ehrliche Gegenprobe zur eigenen Kernaussage: bei großzügigem Budget (Standardeinstellung) zeigt sich KEIN Rückgang -
    im Gegenteil, 3 Ziele decken sogar vollständiger ab. Nur unter knappem Budget (siehe Test oben) zeigt sich der erwartete
    Effekt, und selbst dort ist er klein und seed-empfindlich (siehe App-Text)."""
    rows = {r["n_obj"]: r for r in E.objective_count_experiment(pop=C.COMPARISON_POP, gens=C.COMPARISON_GENS)}
    assert rows[3]["coverage_share"] >= rows[2]["coverage_share"]
