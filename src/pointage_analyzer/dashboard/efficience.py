"""
dashboard/efficience.py

Onglet Efficience OR — visualisation des ratios pointé/référence par OR,
équipe et technicien, avec filtres temporels (année, mois).
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pointage_analyzer.core.config import ScoringConfig
from pointage_analyzer.pipeline.efficience_builder import (
    CAT_DEPASSE,
    CAT_NON_EVALUABLE,
    CAT_NORMAL,
    CAT_SOUS_PRODUCTIF,
    EfficienceBuilder,
    EfficienceResult,
)

logger = logging.getLogger(__name__)

# Palette couleurs cohérente
_COULEURS_CAT = {
    CAT_SOUS_PRODUCTIF: "#ef4444",  # rouge
    CAT_NORMAL:         "#22c55e",  # vert
    CAT_DEPASSE:        "#f59e0b",  # orange/jaune
    CAT_NON_EVALUABLE:  "#94a3b8",  # gris
}

_MOIS_LABELS = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}


def render_efficience_tab(
    result: EfficienceResult,
    config: ScoringConfig,
) -> None:
    """Point d'entrée Streamlit — rendu de l'onglet Efficience."""

    st.subheader("⚡ Analyse d'Efficience OR")
    st.caption(
        "Ratio **heures pointées / heures de référence** (temps vendu ou temps prévu devis). "
        "Un ratio de 100% signifie que l'OR a consommé exactement ce qui était prévu."
    )

    builder = EfficienceBuilder(config=config)
    df_all  = result.df_or

    if df_all.empty:
        st.warning(
            "Aucune donnée d'efficience disponible. "
            "Vérifier que le fichier BO contient les colonnes Temps vendu et Temps prévu devis."
        )
        return

    # ----------------------------------------------------------------
    # FILTRES TEMPORELS
    # ----------------------------------------------------------------
    st.markdown("---")
    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 2])

    annees = builder.get_annees_disponibles(df_all)
    annee_choices = ["Toutes"] + [str(a) for a in annees]
    annee_sel_str = col_f1.selectbox("📅 Année", annee_choices, key="eff_annee")
    annee_sel = int(annee_sel_str) if annee_sel_str != "Toutes" else None

    mois_dispo = builder.get_mois_disponibles(df_all, annee=annee_sel)
    mois_choices = ["Tous"] + [f"{m:02d} – {_MOIS_LABELS[m]}" for m in mois_dispo]
    mois_sel_str = col_f2.selectbox("📅 Mois", mois_choices, key="eff_mois")
    mois_sel = int(mois_sel_str.split("–")[0].strip()) if mois_sel_str != "Tous" else None

    # Filtre équipe
    equipes = ["Toutes"] + sorted(df_all["equipe_principale"].dropna().unique().tolist()) \
        if "equipe_principale" in df_all.columns else ["Toutes"]
    equipe_sel = col_f3.selectbox("🏢 Équipe", equipes, key="eff_equipe")

    df = builder.filter_by_period(df_all, annee=annee_sel, mois=mois_sel)
    if equipe_sel != "Toutes" and "equipe_principale" in df.columns:
        df = df[df["equipe_principale"] == equipe_sel]

    if df.empty:
        st.info("Aucune donnée pour cette période / équipe.")
        return

    nb_eval = int((df["efficience_categorie"] != CAT_NON_EVALUABLE).sum())
    nb_non  = int((df["efficience_categorie"] == CAT_NON_EVALUABLE).sum())
    eval_rows = df[df["efficience_ratio"].notna()]
    ratio_moy = float(eval_rows["efficience_ratio"].mean()) if not eval_rows.empty else None

    # ----------------------------------------------------------------
    # KPIs
    # ----------------------------------------------------------------
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 OR évaluables",    nb_eval)
    k2.metric("⚪ Non évaluables",    nb_non)
    k3.metric("📈 Ratio moyen",
              f"{ratio_moy:.0%}" if ratio_moy else "—",
              delta=f"{(ratio_moy-1)*100:+.1f}pp" if ratio_moy else None,
              delta_color="inverse" if ratio_moy and ratio_moy > 1.2 else "normal")
    k4.metric("🔴 Sous-productifs",
              int((df["efficience_categorie"] == CAT_SOUS_PRODUCTIF).sum()))
    k5.metric("🟡 Dépassements",
              int((df["efficience_categorie"] == CAT_DEPASSE).sum()))

    st.markdown("---")

    # ----------------------------------------------------------------
    # GRAPHIQUES
    # ----------------------------------------------------------------
    tab_or, tab_equipe, tab_tech = st.tabs(["📋 Par OR", "🏢 Par Équipe", "👷 Par Technicien"])

    # --- TAB : Par OR ---
    with tab_or:
        col_g1, col_g2 = st.columns([2, 1])

        # Distribution des catégories d'efficience
        cat_counts = df["efficience_categorie"].value_counts().reset_index()
        cat_counts.columns = ["Catégorie", "Nb OR"]
        fig_pie = px.pie(
            cat_counts,
            names="Catégorie", values="Nb OR",
            color="Catégorie",
            color_discrete_map=_COULEURS_CAT,
            title="Répartition OR par catégorie d'efficience",
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        col_g1.plotly_chart(fig_pie, use_container_width=True)

        # Histogramme du ratio
        eval_df = df[df["efficience_ratio"].notna()]
        if not eval_df.empty:
            fig_hist = px.histogram(
                eval_df, x="efficience_ratio",
                nbins=20,
                color="efficience_categorie",
                color_discrete_map=_COULEURS_CAT,
                title="Distribution du ratio d'efficience",
                labels={"efficience_ratio": "Ratio (pointé/référence)", "count": "Nb OR"},
            )
            fig_hist.add_vline(x=config.efficience_low,  line_dash="dash",
                               line_color="red",    annotation_text="Seuil sous-productif")
            fig_hist.add_vline(x=1.0,                   line_dash="dot",
                               line_color="green",  annotation_text="Idéal (100%)")
            fig_hist.add_vline(x=config.efficience_high, line_dash="dash",
                               line_color="orange", annotation_text="Seuil dépassement")
            col_g2.plotly_chart(fig_hist, use_container_width=True)

        # Scatter : ratio vs heures pointées
        if "total_heures" in df.columns and not eval_df.empty:
            fig_scatter = px.scatter(
                eval_df,
                x="total_heures",
                y="efficience_ratio",
                color="efficience_categorie",
                color_discrete_map=_COULEURS_CAT,
                hover_data=[c for c in [
                    "or_id", "nom_client", "equipe_principale",
                    "technicien_principal_nom", "temps_reference", "source_reference",
                ] if c in eval_df.columns],
                title="Ratio d'efficience vs Heures pointées",
                labels={
                    "total_heures": "Heures pointées (Pointage)",
                    "efficience_ratio": "Ratio efficience",
                },
            )
            fig_scatter.add_hline(y=1.0,                    line_dash="dot",  line_color="green")
            fig_scatter.add_hline(y=config.efficience_low,  line_dash="dash", line_color="red")
            fig_scatter.add_hline(y=config.efficience_high, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Tableau détail
        st.subheader("Détail OR")
        detail_cols = [c for c in [
            "or_id", "nom_client", "equipe_principale", "technicien_principal_nom",
            "temps_reference", "source_reference", "total_heures",
            "efficience_ratio", "efficience_categorie", "efficience_label",
            "alerte_incoherence",
        ] if c in df.columns]
        df_display = df[detail_cols].copy()
        if "efficience_ratio" in df_display.columns:
            df_display["efficience_ratio"] = df_display["efficience_ratio"].map(
                lambda v: f"{v:.0%}" if pd.notna(v) else "—"
            )
        st.dataframe(df_display, use_container_width=True, height=400)

        # Export
        csv = df[detail_cols].to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button("⬇️ Exporter CSV", csv, "efficience_or.csv", "text/csv")

    # --- TAB : Par Équipe ---
    with tab_equipe:
        # Recalcul avec filtre courant
        eff_builder = EfficienceBuilder(config=config)
        df_eq = eff_builder._aggregate_by_equipe(df)

        if df_eq.empty:
            st.info("Données insuffisantes pour l'agrégation par équipe.")
        else:
            fig_eq_bar = px.bar(
                df_eq.sort_values("ratio_moyen"),
                x="ratio_moyen_pct",
                y="equipe_principale",
                color="ratio_moyen_pct",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e", "#f59e0b", "#ef4444"],
                color_continuous_midpoint=100,
                orientation="h",
                text="ratio_moyen_pct",
                title="Ratio d'efficience moyen par équipe (%)",
                labels={
                    "ratio_moyen_pct": "Ratio moyen (%)",
                    "equipe_principale": "Équipe",
                },
            )
            fig_eq_bar.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig_eq_bar.add_vline(x=80,  line_dash="dash", line_color="red",    annotation_text="Seuil bas (80%)")
            fig_eq_bar.add_vline(x=100, line_dash="dot",  line_color="green",  annotation_text="Idéal (100%)")
            fig_eq_bar.add_vline(x=120, line_dash="dash", line_color="orange", annotation_text="Seuil haut (120%)")
            st.plotly_chart(fig_eq_bar, use_container_width=True)

            # Tableau comparatif équipes
            eq_disp = df_eq[[c for c in [
                "equipe_principale", "nb_or", "ratio_moyen_pct", "ratio_median",
                "nb_sous_productif", "nb_normal", "nb_depasse", "pct_sous_productif",
                "heures_pointees_total", "heures_reference_total",
            ] if c in df_eq.columns]].copy()
            if "ratio_median" in eq_disp.columns:
                eq_disp["ratio_median"] = eq_disp["ratio_median"].map(lambda v: f"{v:.0%}")
            st.dataframe(eq_disp, use_container_width=True)

            # Insight coaching
            if not df_eq.empty:
                worst = df_eq.iloc[0]  # déjà trié par ratio_moyen asc
                st.info(
                    f"💡 **À cibler en priorité :** l'équipe **{worst['equipe_principale']}** "
                    f"affiche le ratio moyen le plus bas ({worst['ratio_moyen_pct']:.0f}%) "
                    f"avec {int(worst['nb_sous_productif'])} OR sous-productifs sur {int(worst['nb_or'])}."
                )

    # --- TAB : Par Technicien ---
    with tab_tech:
        eff_builder2 = EfficienceBuilder(config=config)
        df_tech = eff_builder2._aggregate_by_technicien(df)

        if df_tech.empty:
            st.info("Données insuffisantes pour l'agrégation par technicien.")
        else:
            # Option équipe pour filtrer
            if "equipe_principale" in df_tech.columns:
                eq_opts = ["Toutes"] + sorted(df_tech["equipe_principale"].dropna().unique().tolist())
                eq_drill = st.selectbox("Filtrer par équipe", eq_opts, key="eff_tech_eq")
                if eq_drill != "Toutes":
                    df_tech = df_tech[df_tech["equipe_principale"] == eq_drill]

            fig_tech = px.bar(
                df_tech.sort_values("ratio_moyen"),
                x="ratio_moyen_pct",
                y="technicien_principal_nom",
                color="ratio_moyen_pct",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e", "#f59e0b", "#ef4444"],
                color_continuous_midpoint=100,
                orientation="h",
                text="ratio_moyen_pct",
                title="Ratio d'efficience moyen par technicien (%)",
                labels={
                    "ratio_moyen_pct": "Ratio moyen (%)",
                    "technicien_principal_nom": "Technicien",
                },
                height=max(400, len(df_tech) * 22),
            )
            fig_tech.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig_tech.add_vline(x=80,  line_dash="dash", line_color="red")
            fig_tech.add_vline(x=100, line_dash="dot",  line_color="green")
            fig_tech.add_vline(x=120, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_tech, use_container_width=True)

            st.dataframe(
                df_tech[[c for c in [
                    "technicien_principal_nom", "equipe_principale", "nb_or",
                    "ratio_moyen_pct", "nb_sous_productif", "nb_depasse",
                    "heures_pointees", "heures_reference",
                ] if c in df_tech.columns]],
                use_container_width=True,
            )
