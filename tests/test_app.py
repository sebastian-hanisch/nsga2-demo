"""AppTest-Rauchtests: Voreinstellung, jedes Preset, 2-vs-3-Ziele-Umschalter (2D/3D-Chart), Generation-Slider inkl. Abspielen ohne
doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, beide Experimente + Sweep auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nsga2_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_shows_metrics():
    at = _run()
    _ok(at)
    assert at.metric


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_obj_select"] == p["n_obj"] and at.session_state["pop_slider"] == p["pop"]
    assert at.metric


def test_three_objectives_switches_to_3d_chart():
    at = _run(n_obj_select=2, gens_slider=15, pop_slider=15)
    _ok(at)
    at.selectbox(key="n_obj_select").set_value(3).run()
    _ok(at)
    assert at.get("plotly_chart")


@pytest.mark.parametrize("n_obj", [2, 3])
def test_generation_slider_runs_at_various_positions(n_obj):
    at = _run(n_obj_select=n_obj, gens_slider=20)
    _ok(at)
    gen_slider = next(s for s in at.slider if s.key == "nsga2_gen")
    assert gen_slider.value == 20
    gen_slider.set_value(0).run()
    _ok(at)
    assert at.get("plotly_chart")
    gen_slider.set_value(10).run()
    _ok(at)
    assert at.get("plotly_chart")


def test_play_runs_without_duplicate_keys():
    at = _run(gens_slider=20, pop_slider=20)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_dice_buttons_change_the_seeds():
    at = _run()
    old_seed = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old_seed
    old_run_seed = at.session_state["run_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["run_seed_input"] != old_run_seed


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["pop"] = "17"
    at.query_params["nobj"] = "5"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["pop_slider"] == 20
    assert at.session_state["n_obj_select"] == C.DEFAULT_N_OBJ


@pytest.mark.parametrize("kw", [dict(n_obj_select=3), dict(n_slider=10, ballung_slider=100), dict(pop_slider=C.POP_MIN, gens_slider=C.GEN_MIN), dict(mut_slider=1.0, cx_slider=0.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_sweep_runs_on_demand():
    at = _run(gens_slider=15, pop_slider=15)
    at.selectbox(key="sweep_select").set_value("gens").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_comparison_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "COMPARISON_N", 6)
    monkeypatch.setattr(C, "COMPARISON_SEEDS", (1, 2))
    monkeypatch.setattr(C, "COMPARISON_POP", 20)
    monkeypatch.setattr(C, "COMPARISON_GENS", 20)
    at = _run()
    next(b for b in at.button if b.key == "comparison_start").click().run()
    _ok(at)
    assert at.session_state["comparison_on"] and at.get("plotly_chart")


def test_objective_count_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "COMPARISON_N", 6)
    monkeypatch.setattr(C, "COMPARISON_SEEDS", (1, 2))
    monkeypatch.setattr(C, "COMPARISON_POP", 20)
    monkeypatch.setattr(C, "COMPARISON_GENS", 20)
    at = _run()
    next(b for b in at.button if b.key == "objcount_start").click().run()
    _ok(at)
    assert at.session_state["objcount_on"] and at.get("plotly_chart")


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Crowding-Distance trennt Individuen" in m.value for m in at.markdown)
