# 🧬 NSGA-II – Pareto-Dominanz statt fester Gewichtung

**[→ Demo live ausprobieren](https://sebastianhanisch-nsga2-demo.streamlit.app/)**

Zweites Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Fortsetzung der [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/):
deren eigenes Experiment zeigt, dass eine feste Gewichtung mehrerer Ziele nur einen Teil der Pareto-Front erreicht. NSGA-II
(Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002) behebt das über **Pareto-Dominanz-Sortierung** statt einer festen
Gewichtung, mit einer **Crowding-Distance** zur Diversitätserhaltung.

Vehikel ist dieselbe Lieferroute wie die genetic-algorithm-demo (Depot + n Stopps, 100 × 100 km) - jetzt mit zwei oder drei Zielen
(Distanz, CO2, optional Fahrzeit). Die direkten Nachfolger in der Linie sind **NSGA-III** (fix: Crowding-Distance versagt bei vielen
Zielen) und **MOEA/D** (Kontrast: Zerlegung in Skalarisierungs-Unterprobleme statt Dominanz-Sortierung) - keiner davon ist gebaut.

## Warum dieses Problem

Reale Entscheidungen haben selten nur ein Ziel. Eine feste Gewichtung mehrerer Ziele erzwingt eine Entscheidung, bevor man die
Alternativen überhaupt gesehen hat, und - wie die genetic-algorithm-demo zeigt - erreicht sie ohnehin nur einen Teil der
möglichen Kompromisse. NSGA-II zeigt stattdessen die ganze Front gleichzeitig: Selektion und Überlebensauswahl richten sich direkt
nach Pareto-Dominanz, eine Crowding-Distance hält die Front breit gefächert statt an einer Stelle geballt.

## Modell

Dieselbe Lieferroute wie die genetic-algorithm-demo: ein Depot in der Mitte und *n* Kundenstopps in einem 100 × 100-km-Gebiet,
euklidische Entfernungen, eine Rundtour. Zwei zusätzliche, voneinander unabhängige Kantenmerkmale (nicht an der Distanz hängend):
CO2-Faktor und Fahrzeit-Faktor je Straßenabschnitt (0,6–3,4), identisch konstruiert. Ziele: Distanz, CO2, optional Fahrzeit.

**xy und CO2-Faktor der kleinen Vergleichsinstanz (8 Stopps, Vehikel-Seed 19) sind bitidentisch zur genetic-algorithm-demo** -
`nsga2_scenario.generate_perm` reproduziert deren Erzeugung wortgleich, bevor das dritte Merkmal (Fahrzeit) dazukommt. Damit
zitiert diese Demo den dort gemessenen Befund (gewichtete Summe trifft 4 von 8 Frontpunkten) auf derselben Instanz, nicht nur auf
einer ähnlichen.

## Methodik

NSGA-II-Kern (`nsga2_algorithm.py`): schnelle nicht-dominierte Sortierung (vektorisiert über eine (N, N)-Dominanzmatrix - für die
Population, einige hundert Individuen, schnell genug) + Crowding-Distance (normiert durch die Zielzahl, wie die pymoo-Bibliothek) +
elitistische Überlebensauswahl (Eltern + Nachkommen zusammen sortieren, Front für Front auffüllen, letzte Front nach
Crowding-Distance abschneiden). Order Crossover und Tausch-Mutation sind wortgleich aus `ga_algorithm.py` der
genetic-algorithm-demo kopiert.

Für die Brute-Force-Referenz (alle Touren einer kleinen Instanz, bis zu 40 320 bei 9 Knoten) wäre dieselbe (N, N)-Matrix zu
langsam/speicherhungrig (1,6 Milliarden Paare) - `non_dominated_mask` filtert stattdessen gegen eine laufend aktualisierte, meist
kleine Front (O(N · Frontgröße)).

## Befunde / Korrekturen

Zwei echte Fehler wurden beim ersten Live-Test gefunden, nicht vorab angenommen:

1. Die Brute-Force-Referenz zählte anfangs **Touren** statt eindeutiger Zielwerte - zwei verschiedene Touren können denselben
   (Distanz, CO2)-Wert erreichen, was die kleine Vergleichsinstanz auf 9 statt der von der genetic-algorithm-demo gemessenen 8
   Frontpunkte aufblies. Fix: die Brute-Force-Front dedupliziert jetzt vor der Nicht-Dominanz-Prüfung, wie die
   genetic-algorithm-demo es implizit auch tut.
2. Das Experiment "2 vs. 3 Ziele" zeigte bei großzügigem Budget (Populationsgröße 60) **keinen** Rückgang der Abdeckung mit mehr
   Zielen - im Gegenteil, 3 Ziele deckten sogar vollständiger ab (Front wächst von 8 auf 16 Punkte, beide werden bei genug
   Population vollständig erreicht). Erst mit einem eigenen, deutlich knapperen Budget (Populationsgröße 10) zeigt sich der
   erwartete Rückgang (50 % → 37,5 % Abdeckung) - klein und seed-empfindlich, aber in die erwartete Richtung. Ehrlich so berichtet,
   inklusive der Umkehrung bei großzügigem Budget (siehe Tests).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Trifft NSGA-II die Pareto-Front besser als eine feste Gewichtung? | Auf derselben 8-Stopp-Instanz: NSGA-II trifft im Median **6 von 8** Frontpunkten, die gewichtete Summe der genetic-algorithm-demo nur **4 von 8** | `test_comparison_experiment_headline_claims` |
| Wird die Abdeckung mit mehr Zielen schwächer? | Nur unter knappem Budget (Population 10): **50 % → 37,5 %** Abdeckung (2 → 3 Ziele). Bei großzügigem Budget kehrt sich das um | `test_objective_count_experiment_headline_claims`, `test_objective_count_experiment_reverses_under_a_generous_budget` |
| Hilft eine größere Population? | 150 statt 60 Individuen: 2D-Hypervolumen **+9 %**; 15 statt 60: **-16 %** | `test_grosse_population_preset_claims`, `test_kleine_population_preset_claims` |
| Konvergiert die Population zur vollen Front? | Im Standardfall wächst Front 1 von 1 (Startpopulation) auf alle 60 Individuen | `test_standardfall_preset_claims` |

## Ehrliche Grenzen

- **Crowding-Distance ist keine Garantie für gleichmäßige Abdeckung bei vielen Zielen** - dieses Demo-Beispiel mit nur drei Zielen
  ist zu klein, um NSGA-IIIs eigentliche Motivation (vier und mehr Ziele) sauber zu zeigen; der gemessene Effekt ist real, aber
  klein und nur unter knappem Budget sichtbar.
- **Pareto-Dominanz-Sortierung ist O(N²)** in der Populationsgröße - bei sehr großen Populationen wird das selbst zum Engpass
  (MOEA/D vermeidet das durch Zerlegung).
- **Feste Operatorraten** - wie beim GA von Hand eingestellt, nicht aus dem Verlauf der Suche gelernt (CMA-ES passt die
  Suchverteilung selbst an, für kontinuierliche Ziele).
- **2D-Hypervolumen nur bei zwei Zielen** - bei drei Zielen bräuchte es ein Volumenintegral statt der exakten Rechteck-Zerlegung.

## Tests

127 Tests (`pytest tests/ -v`): Kern gegen Handrechnung (6-Punkte-Beispiel für nicht-dominierte Sortierung und
Crowding-Distance) und Bibliotheksgegenprobe (**pymoo** - `fast_non_dominated_sort`/`calc_crowding_distance` stimmen exakt
überein, geprüft über 30 bzw. 20 zufällige Instanzen mit 2-3 Zielen), `non_dominated_mask` gegen die vollständige Dominanzmatrix
kreuzgeprüft, NSGA-II findet auf einer sehr kleinen Instanz nachweislich die Mehrheit der Brute-Force-Front, AppTest-Rauchtests
(jedes Preset, der 2/3-Ziele-Umschalter inkl. 2D/3D-Wechsel, der Generation-Slider inkl. Abspielen, Permalink-Grenzen, beide
Experimente + Sweep auf Abruf) und `test_claims.py` (jede Zahl aus diesem README).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `nsga2_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `nsga2_presets.py` | Permalink/Presets-Mechanik, 2/3-Ziele-Umschalter |
| `nsga2_scenario.py` | Vehikel-Erzeuger (Lieferroute, Distanz/CO2/Fahrzeit) |
| `nsga2_algorithm.py` | NSGA-II-Kern: Sortierung, Crowding-Distance, Selektion, Hauptschleife |
| `nsga2_evaluation.py` | Kennzahlen, Brute-Force-Referenz, Hypervolumen, Sweep, beide Experimente |
| `nsga2_visualization.py` | Plotly-Abbildungen (2D/3D-Zielraum) |

## Bewusst nicht umgesetzt

- Mehr als drei Ziele oder eine allgemeine Referenzpunkt-Nischenbildung - das bringt der geplante Nachfolger NSGA-III.
- Ein 3D- oder allgemeines M-dimensionales Hypervolumen - nur die exakte 2D-Variante ist implementiert.
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
