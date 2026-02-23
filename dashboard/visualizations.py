from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _safe_histogram(frame: pd.DataFrame, column: str, title: str):
    if column not in frame.columns or frame[column].dropna().empty:
        return None
    return px.histogram(frame, x=column, nbins=30, title=title)


def _safe_box(frame: pd.DataFrame, x: str, y: str, title: str):
    if x not in frame.columns or y not in frame.columns:
        return None
    if frame[y].dropna().empty:
        return None
    return px.box(frame, x=x, y=y, title=title, points="outliers")


def global_charts(frame: pd.DataFrame) -> dict[str, go.Figure | None]:
    charts: dict[str, go.Figure | None] = {}
    charts["hist_duree"] = _safe_histogram(frame, "temps_reel", "Histogramme durée pointage")
    charts["hist_efficience"] = _safe_histogram(frame, "efficience_devis", "Histogramme efficience_devis")
    charts["hist_score"] = _safe_histogram(frame, "anomaly_score_global", "Histogramme anomaly_score_global")
    charts["box_efficience_type"] = _safe_box(
        frame, "dim_type_or", "efficience_devis", "Boxplot efficience par Type OR"
    )

    anomaly_by_type = (
        frame.groupby("dim_type_or", dropna=False, as_index=False)
        .agg(taux_anomalie=("is_anomaly", "mean"))
        .sort_values("taux_anomalie", ascending=False)
    )
    anomaly_by_type["taux_anomalie"] = anomaly_by_type["taux_anomalie"] * 100
    charts["bar_taux_type"] = px.bar(
        anomaly_by_type,
        x="dim_type_or",
        y="taux_anomalie",
        title="Taux anomalie par Type OR",
    )

    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True).fillna(0)
        charts["heatmap_corr"] = px.imshow(corr, title="Heatmap corrélation variables numériques")
    else:
        charts["heatmap_corr"] = None

    return charts


def monthly_charts(monthly_df: pd.DataFrame) -> go.Figure | None:
    if monthly_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_df["mois"], y=monthly_df["nb_or"], name="Nombre OR"))
    fig.add_trace(go.Scatter(x=monthly_df["mois"], y=monthly_df["pct_anomalies"], name="% anomalies", yaxis="y2"))
    fig.add_trace(go.Scatter(x=monthly_df["mois"], y=monthly_df["score_moyen"], name="Score moyen", yaxis="y3"))
    fig.update_layout(
        title="Évolution mensuelle",
        yaxis=dict(title="Nombre OR"),
        yaxis2=dict(title="% anomalies", overlaying="y", side="right"),
        yaxis3=dict(title="Score moyen", overlaying="y", side="right", anchor="free", position=1.0),
        legend=dict(orientation="h"),
    )
    return fig


def anomaly_scatter_duration_efficiency(frame: pd.DataFrame) -> go.Figure | None:
    if frame[["temps_reel", "efficience_devis", "anomaly_score_global"]].dropna().empty:
        return None
    return px.scatter(
        frame,
        x="temps_reel",
        y="efficience_devis",
        color="anomaly_score_global",
        hover_data=["or_id", "dim_type_or"],
        title="Durée vs Efficience",
    )


def anomaly_scatter_rule_ml(frame: pd.DataFrame) -> go.Figure | None:
    if frame[["anomaly_score_rule", "anomaly_score_ml"]].dropna().empty:
        return None
    return px.scatter(
        frame,
        x="anomaly_score_rule",
        y="anomaly_score_ml",
        color="anomaly_score_global",
        hover_data=["or_id", "anomaly_types"],
        title="Score rule vs score ML",
    )
