"""
Onglet Exhaustivité — Calendrier de présence technicien × jour.

Ce module est la couche UI pure de l'exhaustivité.
Toute la logique est dans pipeline/exhaustivite_builder.py.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pointage_analyzer.pipeline.exhaustivite_builder import (
    ExhaustiviteBuilder,
    ExhaustiviteError,
    PresenceStatus,
)
from pointage_analyzer.core.config import ScoringConfig


# Palette couleurs sémantiques (RGB Plotly)
STATUS_COLORS: dict[str, str] = {
    PresenceStatus.PRESENT:   "#2ECC71",   # vert
    PresenceStatus.ABSENT:    "#E74C3C",   # rouge
    PresenceStatus.EXCESSIF:  "#E67E22",   # orange
    PresenceStatus.WEEKEND:   "#BDC3C7",   # gris clair
    PresenceStatus.FERIE:     "#85C1E9",   # bleu clair
    PresenceStatus.NON_CONCERNE: "#FFFFFF", # blanc
}

STATUS_LABELS: dict[str, str] = {
    PresenceStatus.PRESENT:   "Présent (≤ 8h)",
    PresenceStatus.ABSENT:    "Absent (0h)",
    PresenceStatus.EXCESSIF:  "Excessif (> 8h)",
    PresenceStatus.WEEKEND:   "Week-end",
    PresenceStatus.FERIE:     "Jour férié",
}


def render_exhaustivite_tab(df_presence: pd.DataFrame, config: ScoringConfig) -> None:
    """
    Rendu de l'onglet Exhaustivité dans Streamlit.

    Args:
        df_presence: DataFrame brut issu de ExhaustiviteBuilder.build_presence_dataframe()
        config: ScoringConfig avec seuils
    """
    st.header("📅 Contrôle d'Exhaustivité — Calendrier de Présence")

    if df_presence.empty:
        st.error("Données de présence indisponibles. Vérifier le fichier Pointage.")
        return

    builder = ExhaustiviteBuilder(config=config)

    # ==================================================================
    # FILTRES
    # ==================================================================
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    with col_f1:
        equipes_dispo = builder.get_equipes_list(df_presence)
        equipe_filter = st.multiselect(
            "🏢 Équipe(s)",
            options=equipes_dispo,
            default=equipes_dispo[:1] if equipes_dispo else [],
            help="Laisser vide = toutes les équipes",
        )

    with col_f2:
        mois_dispo = builder.get_mois_list(df_presence)
        mois_sel = st.selectbox(
            "📆 Mois",
            options=["Tous"] + mois_dispo,
            index=len(mois_dispo) if mois_dispo else 0,
            help="Sélectionner un mois pour le calendrier",
        )

    with col_f3:
        vue_mode = st.radio(
            "Vue",
            options=["Individuelle", "Équipe"],
            index=0,
            horizontal=True,
        )

    # ==================================================================
    # FILTRAGE ET PIVOT
    # ==================================================================
    df_filtered = builder.get_filtered_matrix(
        df_presence,
        equipe_filter=equipe_filter if equipe_filter else None,
        mois_label=mois_sel if mois_sel != "Tous" else None,
    )

    if df_filtered.empty:
        st.warning("Aucune donnée pour cette sélection. Modifier les filtres.")
        return

    # ==================================================================
    # MÉTRIQUES JOURNALIÈRES
    # ==================================================================
    daily_stats = builder.compute_daily_stats(df_filtered)
    if not daily_stats.empty:
        ouvrable = daily_stats[~daily_stats["est_weekend"]]
        if not ouvrable.empty:
            taux_moy = ouvrable["taux_presence"].mean()
            nb_abs_total = ouvrable["nb_absents"].sum()
            nb_exc_total = ouvrable["nb_excessifs"].sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Taux de présence moyen",
                f"{taux_moy:.1%}",
                help="Jours ouvrables uniquement"
            )
            m2.metric("Absences cumulées", f"{nb_abs_total}")
            m3.metric("Jours excessifs (>8h)", f"{nb_exc_total}")
            m4.metric("Techniciens actifs", f"{df_filtered['salarie_nom'].nunique() if 'salarie_nom' in df_filtered.columns else 0}")

    st.markdown("---")

    # ==================================================================
    # CALENDRIER HEATMAP
    # ==================================================================
    if mois_sel == "Tous":
        st.info(
            "💡 Sélectionner un mois spécifique pour afficher le calendrier détaillé."
        )
        _render_monthly_summary_chart(df_filtered, daily_stats)
        return

    try:
        use_nom = vue_mode == "Individuelle"
        pivot_heures, status_matrix = builder.build_pivot_calendar(
            df_filtered, use_nom=use_nom
        )
        _render_heatmap_calendar(pivot_heures, status_matrix)
    except ExhaustiviteError as exc:
        st.error(f"Impossible de construire le calendrier: {exc}")
        return

    # ==================================================================
    # TABLEAU DÉTAILLÉ (optionnel)
    # ==================================================================
    with st.expander("📊 Données brutes (technicien × jour)"):
        st.dataframe(
            df_filtered.sort_values(["equipe_nom", "salarie_nom", "date"])
            if all(c in df_filtered.columns for c in ["equipe_nom", "salarie_nom", "date"])
            else df_filtered,
            use_container_width=True,
            height=350,
        )


def _render_heatmap_calendar(
    pivot_heures: pd.DataFrame,
    status_matrix: pd.DataFrame,
) -> None:
    """Rendu du calendrier en heatmap Plotly interactive."""

    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in pivot_heures.columns]
    techniciens = list(pivot_heures.index)

    # Construction de la matrice couleur et texte
    z_colors = []
    z_text = []
    customdata = []

    status_to_num = {
        PresenceStatus.PRESENT:    1,
        PresenceStatus.ABSENT:     0,
        PresenceStatus.EXCESSIF:   2,
        PresenceStatus.WEEKEND:    -1,
        PresenceStatus.FERIE:      -2,
        PresenceStatus.NON_CONCERNE: -3,
    }

    colorscale_map = [
        [-3, STATUS_COLORS[PresenceStatus.NON_CONCERNE]],
        [-2, STATUS_COLORS[PresenceStatus.FERIE]],
        [-1, STATUS_COLORS[PresenceStatus.WEEKEND]],
        [0, STATUS_COLORS[PresenceStatus.ABSENT]],
        [1, STATUS_COLORS[PresenceStatus.PRESENT]],
        [2, STATUS_COLORS[PresenceStatus.EXCESSIF]],
    ]

    for tech in techniciens:
        row_colors = []
        row_text = []
        row_custom = []
        for col_date in pivot_heures.columns:
            heures = pivot_heures.loc[tech, col_date]
            statut = status_matrix.loc[tech, col_date]
            row_colors.append(status_to_num.get(statut, -3))
            row_text.append(
                f"{heures:.1f}h" if heures > 0 else ""
            )
            row_custom.append([heures, STATUS_LABELS.get(statut, statut)])
        z_colors.append(row_colors)
        z_text.append(row_text)
        customdata.append(row_custom)

    # Plotly heatmap
    fig = go.Figure(
        go.Heatmap(
            z=z_colors,
            x=dates,
            y=techniciens,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 9, "color": "black"},
            customdata=customdata,
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata[0]:.1f}h — %{customdata[1]}<extra></extra>",
            colorscale=[
                [0.0,   STATUS_COLORS[PresenceStatus.NON_CONCERNE]],
                [0.167, STATUS_COLORS[PresenceStatus.FERIE]],
                [0.333, STATUS_COLORS[PresenceStatus.WEEKEND]],
                [0.5,   STATUS_COLORS[PresenceStatus.ABSENT]],
                [0.667, STATUS_COLORS[PresenceStatus.PRESENT]],
                [1.0,   STATUS_COLORS[PresenceStatus.EXCESSIF]],
            ],
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )

    fig.update_layout(
        title="Calendrier de présence — heures pointées par technicien",
        xaxis_title="Date",
        yaxis_title="Technicien",
        height=max(400, 30 * len(techniciens) + 150),
        margin=dict(l=200, r=20, t=60, b=60),
        plot_bgcolor="white",
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Légende manuelle
    col_leg = st.columns(len(STATUS_LABELS))
    for i, (status, label) in enumerate(STATUS_LABELS.items()):
        with col_leg[i]:
            color = STATUS_COLORS[status]
            st.markdown(
                f'<div style="background:{color};border-radius:4px;padding:4px 8px;'
                f'text-align:center;font-size:12px;color:{"black" if status != PresenceStatus.ABSENT else "white"}">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )


def _render_monthly_summary_chart(
    df_filtered: pd.DataFrame, daily_stats: pd.DataFrame
) -> None:
    """Vue résumée multi-mois quand aucun mois spécifique n'est sélectionné."""
    if daily_stats.empty or "date" not in daily_stats.columns:
        return

    import plotly.express as px

    fig = px.bar(
        daily_stats[~daily_stats["est_weekend"]],
        x="date",
        y="taux_presence",
        color="taux_presence",
        color_continuous_scale=["#E74C3C", "#F39C12", "#2ECC71"],
        title="Taux de présence journalier (jours ouvrables)",
        labels={"taux_presence": "Taux présence", "date": "Date"},
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
