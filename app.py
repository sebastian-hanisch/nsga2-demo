"""NSGA-II - eine Population, sortiert nach Pareto-Dominanz statt einer festen Gewichtung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, Fortsetzung der genetic-algorithm-demo: deren eigenes
Experiment zeigt, dass eine feste Gewichtung mehrerer Ziele nur einen Teil der Pareto-Front erreicht. NSGA-II behebt das, indem es
die Population direkt nach Pareto-Dominanz sortiert (schnelle nicht-dominierte Sortierung) und die Vielfalt über eine
Crowding-Distance statt einer Gewichtung erhält. Vehikel: dieselbe Lieferroute, jetzt mit zwei oder drei Zielen (Distanz, CO2,
optional Fahrzeit) - die drei-Ziele-Ansicht bereitet die eigene Schwäche vor (Crowding-Distance versagt bei vielen Zielen), an der
der nächste Nachfolger NSGA-III ansetzt.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import nsga2_constants as C
from nsga2_evaluation import Settings, analyse, comparison_experiment, hypervolume_2d, hypervolume_reference, instance, objective_count_experiment, sweep
from nsga2_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from nsga2_visualization import build_comparison, build_front1_size_curve, build_objective_count, build_objective_scatter_2d, build_objective_scatter_3d, build_sweep

st.set_page_config(page_title="NSGA-II – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


@st.cache_data(show_spinner=False)
def _objective_count():
    return objective_count_experiment()


st.title("🧬 NSGA-II – Pareto-Dominanz statt fester Gewichtung")
st.markdown(
    """
Wie optimiert man **mehrere Ziele gleichzeitig**, ohne sie vorab zu einer einzigen Zahl zusammenzurechnen? **NSGA-II**
(Non-dominated Sorting Genetic Algorithm II) sortiert die Population direkt nach **Pareto-Dominanz**: eine Lösung ist besser als
eine andere, wenn sie in **jedem** Ziel mindestens gleich gut und in **mindestens einem** Ziel echt besser ist. Nicht-dominierte
Lösungen bilden **Front 1** - keine davon ist eindeutig die beste, sie zeigen den ganzen erreichbaren Kompromiss. Eine
**Crowding-Distance** sorgt zusätzlich dafür, dass die Front breit gefächert bleibt statt sich an einer Stelle zu ballen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweites "
    "Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, Fortsetzung der "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) - **ein** Verfahren an einem "
    "wachsenden Beispiel. Vehikel ist dieselbe Lieferroute (Depot + n Stopps, 100 × 100 km), jetzt mit zwei oder drei Zielen: "
    "Distanz, CO2 und optional Fahrzeit (je ein unabhängiger Faktor pro Straßenabschnitt)."
)

with st.expander("So funktioniert NSGA-II", expanded=True):
    st.markdown(
        """
1. **Population.** Wie beim Genetischen Algorithmus: viele zufällige Touren gleichzeitig.
2. **Ziele statt einer Fitness.** Jede Tour bekommt mehrere Werte (Distanz, CO2, ggf. Fahrzeit) - keine Gewichtung, kein Zusammenrechnen.
3. **Nicht-dominierte Sortierung.** Front 1: keine andere Tour ist in jedem Ziel mindestens so gut und in einem echt besser. Front 2:
   dieselbe Regel, nachdem Front 1 entfernt wurde. Und so weiter.
4. **Crowding-Distance.** Innerhalb einer Front: wie weit eine Lösung von ihren Nachbarn im Zielraum entfernt ist (Randlösungen
   zählen als unendlich weit). Hält die Front breit gefächert statt gedrängt an einer Stelle.
5. **Turnier + Crossover + Mutation.** Eltern werden über ein Turnier gewählt (niedrigerer Rang gewinnt, bei Gleichstand die höhere
   Crowding-Distance) - Nachkommen entstehen wie beim GA (Order Crossover, Tausch-Mutation).
6. **Überleben.** Eltern und Nachkommen (2N) werden zusammen sortiert; Front für Front aufgefüllt, bis die letzte Front nicht mehr
   ganz hineinpasst - dort entscheidet die Crowding-Distance, wer bleibt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_obj = st.selectbox("Anzahl Ziele", C.N_OBJ_OPTIONS, key="n_obj_select", format_func=lambda k: C.N_OBJ_LABELS[k],
                          help="2 Ziele: Distanz und CO2 (identisch zur genetic-algorithm-demo). 3 Ziele: zusätzlich Fahrzeit - "
                               "bereitet die eigene Schwäche von NSGA-II vor (Crowding-Distance wird mit mehr Zielen weniger "
                               "trennscharf), siehe Experiment weiter unten.")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    cluster_share = st.slider("Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
                               help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.")
    st.markdown("**NSGA-II**")
    pop_size = st.slider("Populationsgröße", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP, help="Zahl der Individuen je Generation.")
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    cx_prob = st.slider("Crossover-Rate", *bounds("cx_slider"), key="cx_slider", step=C.CX_STEP, format="%.2f")
    mut_prob = st.slider("Mutationsrate", *bounds("mut_slider"), key="mut_slider", step=C.MUT_STEP, format="%.2f")
    tournament_k = st.slider("Turniergröße k", *bounds("k_slider"), key="k_slider", step=C.K_STEP,
                              help="NSGA-II vergleicht Turnierteilnehmer zuerst nach Rang (Front), bei Gleichstand nach Crowding-Distance.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des NSGA-II-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "n_obj_select": int(n_obj), "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed),
    "pop_slider": int(pop_size), "gens_slider": int(generations), "cx_slider": float(cx_prob), "mut_slider": float(mut_prob), "k_slider": int(tournament_k), "run_seed_input": int(run_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(n_obj), int(seed), int(pop_size), int(generations), float(cx_prob), float(mut_prob), int(tournament_k), int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings

# --- NSGA-II in Aktion --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 NSGA-II in Aktion")
if "nsga2_gen" not in st.session_state or st.session_state.get("nsga2_gen_owner") != data_key:
    st.session_state["nsga2_gen"] = n_gens_run
    st.session_state["nsga2_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="nsga2_gen", help="0 = Startpopulation.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    with view_slot.container():
        c1, c2 = st.columns(2)
        c1.markdown(f"**Generation {g} von {n_gens_run} – Front 1: {int((gd.rank == 0).sum())} von {settings.pop} Individuen**")
        if settings.n_obj == 2:
            c1.plotly_chart(build_objective_scatter_2d(gd.objectives, gd.rank), width="stretch", key=f"g_scatter_{g}")
        else:
            c1.plotly_chart(build_objective_scatter_3d(gd.objectives, gd.rank), width="stretch", key=f"g_scatter_{g}")
        c2.markdown("**Größe von Front 1 über die Generationen**")
        c2.plotly_chart(build_front1_size_curve(result.front1_history[:g + 1]), width="stretch", key=f"g_curve_{g}")


if auto_play:
    for f in _frames():
        _render(f)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was NSGA-II gefunden hat")
m1, m2, m3 = st.columns(3)
m1.metric("Größe von Front 1", f"{len(result.front1)}", delta=f"Start {int((result.generations[0].rank == 0).sum())}" if result.generations else None, delta_color="off",
          help="Wie viele Individuen der Endpopulation nicht dominiert sind.")
if settings.n_obj == 2:
    ref = hypervolume_reference(settings.n, settings.cluster_share, settings.seed)
    hv = hypervolume_2d(result.front1, ref)
    m2.metric("2D-Hypervolumen", f"{hv:,.0f}".replace(",", "."), help="Von der Front dominierte Fläche im Zielraum gegen einen festen, schlechteren Referenzpunkt - je größer, desto besser die Front (Güte UND Breite zusammen).")
else:
    m2.metric("2D-Hypervolumen", "nur bei 2 Zielen", help="Diese Demo berechnet das Hypervolumen nur für den 2-Ziele-Fall (exakte Flächenrechnung); bei 3 Zielen bräuchte es ein Volumenintegral.")
m3.metric("Generationen × Population", f"{settings.gens} × {settings.pop}")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt die Frontqualität von Populationsgröße und Generationen ab?")
st.caption("Gemessen mit 2 Zielen (Hypervolumen ist dort exakt definiert), Mittel über 5 feste Vehikel-Seeds.")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(n=settings.n, cluster_share=settings.cluster_share, n_obj=2, cx=settings.cx, mut=settings.mut, k=settings.k)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: schließt GAs Cliffhanger -------------------------------------------------------------------------------------------

st.subheader("🔬 Trifft NSGA-II die Pareto-Front besser als eine feste Gewichtung?")
st.caption(f"Dieselbe kleine Instanz wie das Pareto-Experiment der genetic-algorithm-demo ({C.COMPARISON_N} Stopps, Vehikel-Seed {C.COMPARISON_VEHICLE_SEED}) - dort traf eine gewichtete Summe {C.GA_WEIGHTED_SUM_REACHED} von {C.GA_WEIGHTED_SUM_FRONT_SIZE} Frontpunkten.")
if st.button("Brute-Force-Front gegen NSGA-II rechnen (dauert etwa 20 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne die Brute-Force-Front und mehrere NSGA-II-Läufe..."):
        report = _comparison()
    st.plotly_chart(build_comparison(report), width="stretch", key="comparison_chart")
    c1, c2 = st.columns(2)
    c1.metric("NSGA-II, Median über 10 Läufen", f"{report['reached_median']:.0f} von {report['front_size']}")
    c2.metric("Gewichtete Summe (GA-Demo)", f"{report['ga_weighted_sum_reached']} von {report['front_size']}")

st.markdown("---")

# --- Experiment 2: 2 vs. 3 Ziele (bereitet NSGA-III vor) ------------------------------------------------------------------------------

st.subheader("🔬 Wird die Abdeckung mit mehr Zielen schwächer?")
st.caption(f"Dieselbe kleine Instanz, einmal mit 2 und einmal mit 3 Zielen (Distanz, CO2, Fahrzeit) - mit knappem Budget (Population {C.OBJCOUNT_POP}, unabhängig von der Seitenleiste). "
           "Bei großzügigem Budget (z. B. der Standardeinstellung) deckt NSGA-II beide Fälle vollständig oder fast vollständig ab, der Unterschied verschwindet oder kehrt sich sogar um - "
           "erst unter Druck wird sichtbar, dass mehr Ziele die Suche schwerer machen.")
if st.button("2 gegen 3 Ziele rechnen (dauert etwa 15 Sekunden)", key="objcount_start"):
    st.session_state["objcount_on"] = True
if st.session_state.get("objcount_on"):
    with st.spinner("Rechne Brute-Force-Fronten und NSGA-II-Läufe für 2 und 3 Ziele..."):
        rows_oc = _objective_count()
    st.plotly_chart(build_objective_count(rows_oc), width="stretch", key="objcount_chart")
    oc_cols = st.columns(len(rows_oc))
    for col, r in zip(oc_cols, rows_oc):
        col.metric(f"{r['n_obj']} Ziele", f"{r['coverage_share']:.0%}", help=f"Front-Größe {r['front_size']}, im Median {r['reached_median']:.0f} getroffen.")
    st.caption("Die Front selbst wächst mit der Zielzahl (hier von 8 auf 16 Punkte) - bei diesem winzigen Beispiel mit nur drei Zielen reicht das für eine ehrliche, aber kleine und "
               "seed-empfindliche Demonstration. NSGA-IIIs eigentliche Motivation liegt bei echt vielen Zielen (vier und mehr), die diese Demo bewusst nicht zeigt.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Crowding-Distance trennt Individuen ausreichend** | Mit mehr Zielen wird der Zielraum dünner besetzt - unter knappem Budget zeigt das Experiment oben einen (kleinen, seed-empfindlichen) Rückgang; bei genügend Ressourcen zeigt sich in dieser winzigen Demo noch nichts. Die eigentliche Schwäche wird erst bei echt vielen Zielen (vier und mehr) deutlich. | **NSGA-III** (referenzpunktbasierte Nischenbildung statt Crowding-Distance) |
| **Pareto-Dominanz ist die richtige Vergleichsregel** | Ein anderer Mechanismus - Zerlegung in Skalarisierungs-Unterprobleme statt Dominanz-Sortierung - löst dieselbe Aufgabe anders. | **MOEA/D** (Kontrast, kein Fix) |
| **O(N²) nicht-dominierte Sortierung ist schnell genug** | Bei sehr großen Populationen wird die Sortierung selbst zum Engpass. | außerhalb dieser Linie (z. B. MOEA/D vermeidet das durch Zerlegung) |
| **Feste Operatorraten** | Wie beim GA werden Crossover-/Mutationsrate von Hand eingestellt, nicht gelernt. | **CMA-ES** (für kontinuierliche Ziele) |
"""
)
st.caption(
    "Direkte Nachfolger von NSGA-II in der Populations-Linie: **NSGA-III** (mehr als drei Ziele) und **MOEA/D** (Kontrast: Zerlegung "
    "statt Dominanz-Sortierung). Vorgänger: die "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/), deren gewichtete Summe hier direkt verglichen wird."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Pareto-Dominanz.** $a$ dominiert $b$ (Minimierung), wenn $a_i \le b_i$ für alle Ziele $i$ und $a_j < b_j$ für mindestens ein $j$.
Front 1 = die nicht dominierten Individuen der Population; Front $k$ = die nicht dominierten Individuen, nachdem Front $1, \dots, k-1$
entfernt wurden.

**Crowding-Distance.** Für ein Individuum $p$ innerhalb einer Front mit $M$ Zielen: $d_p = \frac{1}{M}\sum_{i=1}^{M} \frac{f_i(p^{+}) - f_i(p^{-})}{f_i^{\max} - f_i^{\min}}$,
das Mittel der normierten Abstände zu den beiden im Ziel $i$ benachbarten Individuen (wie in der pymoo-Bibliothek, die Division durch
$M$ ist eine reine Reskalierung und ändert keinen Vergleich); Randindividuen je Ziel erhalten $d_p = \infty$.

**Crowded-Comparison-Operator.** $a \prec_n b$, wenn $\text{rang}(a) < \text{rang}(b)$, oder bei Rang-Gleichstand $d_a > d_b$.

**Überlebensauswahl.** Eltern- und Nachkommenpopulation ($2N$) werden nicht-dominiert sortiert; Fronten werden der Reihe nach in die
neue Population übernommen, bis die nächste Front nicht mehr vollständig passt - dort werden die $d_p$-größten Individuen gewählt.

**Zielfunktionen (Lieferroute).** Distanz: $f_1(x) = \sum_k d_{\pi(k)\pi(k+1)}$. CO2/Fahrzeit: $f_i(x) = \sum_k d_{\pi(k)\pi(k+1)} \cdot c_{\pi(k)\pi(k+1)}^{(i)}$
mit einem je Kante unabhängig gezogenen Faktor $c^{(i)}$.

**2D-Hypervolumen.** Für eine nicht-dominierte Front (aufsteigend nach $f_1$ sortiert) und einen Referenzpunkt $R$ (schlechter als
jeder Frontpunkt in jedem Ziel): $HV = \sum_i (x_{i+1} - x_i)(R_2 - y_i)$, mit $x_{|F|+1} := R_1$.

Implementiert in `nsga2_algorithm.py` (Sortierung, Crowding-Distance, Hauptschleife), `nsga2_scenario.py` (Vehikel),
`nsga2_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
