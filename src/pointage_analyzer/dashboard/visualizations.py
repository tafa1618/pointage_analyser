"""Visualizations Plotly pour le dashboard OR-level et scoring."""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _safe(frame: pd.DataFrame, col: str) -> bool:
    return col in frame.columns and not frame[col].dropna().empty


def render_global_charts(df: pd.DataFrame) -> dict[str, go.Figure]:
    """Graphiques globaux : distribution scores, anomalies par catégorie, position."""
    charts: dict[str, go.Figure] = {}

    # Distribution score final
    if _safe(df, "score_final"):
        fig = px.histogram(
            df,
            x="score_final",
            color="anomaly_flag" if "anomaly_flag" in df.columns else None,
            nbins=30,
            title="Distribution des scores d'anomalie",
            labels={"score_final": "Score final [0-1]", "count": "Nb OR"},
            color_discrete_map={True: "#E74C3C", False: "#2ECC71"},
        )
        fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                      annotation_text="Seuil anomalie (0.6)")
        charts["score_distribution"] = fig

    # Répartition par position
    if _safe(df, "position"):
        pos_counts = df["position"].value_counts().reset_index()
        pos_counts.columns = ["position", "count"]
        pos_label_map = {"EC": "En Cours", "CP": "Clôturé", "AF": "Att. Facture", "TT": "Autre"}
        pos_counts["label"] = pos_counts["position"].map(pos_label_map).fillna(pos_counts["position"])
        fig = px.pie(
            pos_counts, names="label", values="count",
            title="Répartition des OR par statut (Position)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        charts["positions"] = fig

    # Score moyen par équipe
    if _safe(df, "equipe_principale") and _safe(df, "score_final"):
        equipe_stats = (
            df.groupby("equipe_principale")["score_final"].mean()
            .reset_index()
            .sort_values("score_final", ascending=False)
        )
        fig = px.bar(
            equipe_stats,
            x="equipe_principale",
            y="score_final",
            color="score_final",
            color_continuous_scale=["#2ECC71", "#F39C12", "#E74C3C"],
            title="Score d'anomalie moyen par équipe",
            labels={"score_final": "Score moyen", "equipe_principale": "Équipe"},
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        charts["equipe_scores"] = fig

    return charts


def render_anomaly_scatter(df: pd.DataFrame) -> go.Figure | None:
    """Scatter : score règles vs score ML, coloré par flag anomalie."""
    if not (_safe(df, "rule_score_total") and _safe(df, "ml_score")):
        return None

    hover_col = next(
        (c for c in ["type_or", "nature_materiel", "or_id"] if c in df.columns),
        None,
    )
    fig = px.scatter(
        df,
        x="rule_score_total",
        y="ml_score",
        color="anomaly_flag" if "anomaly_flag" in df.columns else None,
        size="total_heures" if "total_heures" in df.columns else None,
        hover_name=hover_col,
        hover_data={
            "score_final": ":.3f" if "score_final" in df.columns else False,
            "technicien_principal_nom": True if "technicien_principal_nom" in df.columns else False,
            "rule_anomaly_types": True if "rule_anomaly_types" in df.columns else False,
        },
        title="Score règles vs Score ML — détection d'anomalies",
        labels={
            "rule_score_total": "Score règles métier",
            "ml_score": "Score ML (Isolation Forest)",
        },
        color_discrete_map={True: "#E74C3C", False: "#95A5A6"},
        opacity=0.75,
    )

    fig.add_shape(
        type="rect",
        x0=0.6, y0=0.6, x1=1.0, y1=1.0,
        fillcolor="rgba(231, 76, 60, 0.1)",
        line=dict(color="red", dash="dash"),
    )
    fig.add_annotation(
        x=0.8, y=0.95, text="Zone critique",
        showarrow=False, font=dict(color="red", size=11),
    )
    return fig


def render_rule_breakdown(df: pd.DataFrame) -> go.Figure | None:
    """Barres horizontales : nb OR déclenché par règle."""
    rule_cols = {
        "rule_heures_negatives": "Heures négatives",
        "rule_heures_excessives": "Heures excessives",
        "rule_variance_anormale": "Variance anormale",
        "rule_or_sans_pointage": "OR sans pointage",
        "rule_retard_premier_pointage": "Retard 1er pointage",
        "rule_or_cloture_sans_finance": "Clôturé sans BO",
        "rule_surconsommation": "Surconsommation",
        "rule_efficience_faible": "Efficience faible",
    }

    data = []
    for col, label in rule_cols.items():
        if col in df.columns:
            nb = (df[col] > 0.1).sum()
            data.append({"règle": label, "nb_or": nb})

    if not data:
        return None

    df_rules = pd.DataFrame(data).sort_values("nb_or", ascending=True)
    fig = px.bar(
        df_rules,
        x="nb_or",
        y="règle",
        orientation="h",
        title="OR déclenchés par règle métier",
        labels={"nb_or": "Nombre d'OR", "règle": ""},
        color="nb_or",
        color_continuous_scale=["#2ECC71", "#F39C12", "#E74C3C"],
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
    return fig


def render_technicien_chart(tech_stats: pd.DataFrame) -> go.Figure | None:
    """Scatter techniciens : nb OR vs taux anomalie."""
    if tech_stats.empty or "nb_or" not in tech_stats.columns:
        return None

    fig = px.scatter(
        tech_stats,
        x="nb_or",
        y="taux_anomalie" if "taux_anomalie" in tech_stats.columns else "score_moyen",
        size="heures_totales" if "heures_totales" in tech_stats.columns else None,
        hover_name="technicien_principal_nom",
        color="equipe" if "equipe" in tech_stats.columns else None,
        title="Techniciens : volume OR vs taux d'anomalie",
        labels={
            "nb_or": "Nombre d'OR",
            "taux_anomalie": "Taux d'anomalie",
        },
        opacity=0.8,
    )
    return fig
