"""Plotly-Abbildungen der NSGA-II-Demo: Population im Zielraum (2D/3D, eingefärbt nach Rang), Abdeckungs- und Sweep-Abbildungen.
Achsen sind gesperrt (fixedrange), damit Touch-Scrollen auf der Seite nicht am Chart hängen bleibt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FRONT1_COLOR = "#54a24b"
OTHER_COLOR = "#9ecae9"
REF_COLOR = "#7f7f7f"
GA_COLOR = "#e45756"
TOUR_COLOR = "#4c78a8"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_objective_scatter_2d(objectives, rank, true_front=None):
    """Population im Zielraum (Distanz vs. CO2): Front 1 hervorgehoben (grün, verbunden), übrige Ränge blass-blau.
    `true_front`: optionale Brute-Force-Referenzfront (graue gestrichelte Linie) für die kleine Vergleichsinstanz."""
    fig = go.Figure()
    others = objectives[rank > 0]
    if len(others):
        fig.add_trace(go.Scatter(x=others[:, 0], y=others[:, 1], mode="markers", marker=dict(size=6, color=OTHER_COLOR), name="dominiert"))
    front1 = objectives[rank == 0]
    order = np.argsort(front1[:, 0])
    fig.add_trace(go.Scatter(x=front1[order, 0], y=front1[order, 1], mode="markers+lines", line=dict(color=FRONT1_COLOR, width=1.5, dash="dot"),
                              marker=dict(size=9, color=FRONT1_COLOR, line=dict(width=1, color="white")), name="Front 1 (nicht dominiert)"))
    if true_front is not None:
        t_order = np.argsort(true_front[:, 0])
        fig.add_trace(go.Scatter(x=true_front[t_order, 0], y=true_front[t_order, 1], mode="markers", marker=dict(size=11, symbol="circle-open", color=REF_COLOR, line=dict(width=2)),
                                  name="wahre Pareto-Front"))
    fig.update_xaxes(title_text="Distanz (km)")
    fig.update_yaxes(title_text="CO2-Kosten")
    return _base(fig, 380)


def build_objective_scatter_3d(objectives, rank):
    fig = go.Figure()
    others = objectives[rank > 0]
    if len(others):
        fig.add_trace(go.Scatter3d(x=others[:, 0], y=others[:, 1], z=others[:, 2], mode="markers", marker=dict(size=3, color=OTHER_COLOR), name="dominiert"))
    front1 = objectives[rank == 0]
    fig.add_trace(go.Scatter3d(x=front1[:, 0], y=front1[:, 1], z=front1[:, 2], mode="markers", marker=dict(size=5, color=FRONT1_COLOR), name="Front 1 (nicht dominiert)"))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05),
                       scene=dict(xaxis_title="Distanz (km)", yaxis_title="CO2-Kosten", zaxis_title="Fahrzeit"))
    return fig


def build_front1_size_curve(front1_history):
    xs = list(range(len(front1_history)))
    ys = [len(f) for f in front1_history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TOUR_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Größe von Front 1")
    return _base(fig, 260)


def build_comparison(report):
    """Abdeckung der Brute-Force-Front: NSGA-II (Streuung über mehrere Läufe) gegen die gewichtete Summe der genetic-algorithm-demo."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=report["reached_all"], name="NSGA-II", marker_color=FRONT1_COLOR, boxpoints="all"))
    fig.add_trace(go.Bar(x=["Gewichtete Summe (GA-Demo)"], y=[report["ga_weighted_sum_reached"]], marker_color=GA_COLOR, showlegend=False, width=0.4))
    fig.add_hline(y=report["front_size"], line=dict(color=REF_COLOR, dash="dot"), annotation_text="volle Front", annotation_position="top right")
    fig.update_yaxes(title_text="Getroffene Frontpunkte")
    return _base(fig, 340)


def build_objective_count(rows):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{r['n_obj']} Ziele" for r in rows], y=[r["coverage_share"] for r in rows], marker_color=FRONT1_COLOR, showlegend=False))
    fig.update_yaxes(title_text="Abdeckung der wahren Front", tickformat=".0%")
    return _base(fig, 300)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("2D-Hypervolumen", "Größe von Front 1"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["hypervolume"] for r in rows], mode="lines+markers", line=dict(color=TOUR_COLOR, width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["front_size"] for r in rows], mode="lines+markers", line=dict(color=FRONT1_COLOR, width=2.5), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text=param_label)
    return _base(fig, 300)
