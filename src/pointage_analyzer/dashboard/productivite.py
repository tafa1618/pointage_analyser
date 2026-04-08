"""
Onglet Productivité — analyse des heures facturables vs totales.

Filtre année intégré :
  - Toutes les visualisations changent en fonction de l'année sélectionnée
  - YTD = Jan → dernier mois disponible de l'année choisie
  - Évolution mensuelle = mois de l'année choisie uniquement
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pointage_analyzer.pipeline.productivite_builder import (
    ProductiviteBuilder,
    ProductiviteResult,
    SEUIL_EXCELLENT,
    SEUIL_BON,
    SEUIL_FAIBLE,
)

_COULEUR_EXCELLENT = "#22c55e"
_COULEUR_BON       = "#FFCD11"
_COULEUR_FAIBLE    = "#f97316"
_COULEUR_CRITIQUE  = "#ef4444"


def _couleur_prod(ratio: float) -> str:
    if ratio >= SEUIL_EXCELLENT: return _COULEUR_EXCELLENT
    elif ratio >= SEUIL_BON:     return _COULEUR_BON
    elif ratio >= SEUIL_FAIBLE:  return _COULEUR_FAIBLE
    else:                         return _COULEUR_CRITIQUE


def _filter_by_year(pt_harm: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filtre le DataFrame pointage sur l'année sélectionnée."""
    date_col = "date_saisie" if "date_saisie" in pt_harm.columns else "Saisie heures - Date"
    dates = pd.to_datetime(pt_harm[date_col], errors="coerce")
    return pt_harm[dates.dt.year == year].copy()


def render_productivite_tab(productivite: ProductiviteResult, pt_harm: pd.DataFrame) -> None:
    """
    Point d'entrée depuis app.py.

    Args:
        productivite: résultat pré-calculé (année complète — non filtré)
        pt_harm:      DataFrame Pointage harmonisé brut (pour recalcul par année)
    """
    st.markdown("## 📊 Analyse de la Productivité")

    # ── Sélecteur année ───────────────────────────────────────────────
    date_col = "date_saisie" if "date_saisie" in pt_harm.columns else "Saisie heures - Date"
    dates_all = pd.to_datetime(pt_harm[date_col], errors="coerce")
    annees_dispo = sorted(dates_all.dt.year.dropna().unique().astype(int).tolist(), reverse=True)

    if not annees_dispo:
        st.error("Aucune date valide dans les données.")
        return

    col_ann, col_spacer = st.columns([1, 3])
    with col_ann:
        annee_sel = st.selectbox(
            "📅 Année",
            options=annees_dispo,
            index=0,
            key="prod_annee",
        )

    # Recalcul sur l'année sélectionnée
    pt_filtered = _filter_by_year(pt_harm, annee_sel)
    if pt_filtered.empty:
        st.warning(f"Aucune donnée pour {annee_sel}.")
        return

    builder = ProductiviteBuilder()
    result  = builder.build(pt_filtered)

    if result.ytd_facturable == 0 and result.ytd_non_facturable == 0:
        st.warning("Données insuffisantes pour calculer la productivité.")
        return

    # ── KPI Cards ─────────────────────────────────────────────────────
    _render_kpi_cards(result, annee_sel)
    st.divider()

    # ── Évolution mensuelle ───────────────────────────────────────────
    _render_evolution_mensuelle(result)
    st.divider()

    # ── Barres équipes ────────────────────────────────────────────────
    _render_barres_equipes(result)
    st.divider()

    # ── Heatmap technicien × mois ─────────────────────────────────────
    _render_heatmap_tech_mois(result)
    st.divider()

    # ── Tableau techniciens ───────────────────────────────────────────
    _render_tableau_techniciens(result)
    st.divider()

    # ── Analyse équipe proxy ──────────────────────────────────────────
    _render_analyse_equipe(result)


# ══════════════════════════════════════════════════════════════════════
# SECTIONS
# ══════════════════════════════════════════════════════════════════════

def _render_kpi_cards(result: ProductiviteResult, annee: int) -> None:
    couleur = _couleur_prod(result.ytd_productivite)
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.markdown(
        f"""<div style="background:#1a1a2e;border-left:4px solid {couleur};
        border-radius:4px;padding:12px 16px">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px">
        Productivité YTD {annee}</div>
        <div style="font-size:32px;font-weight:700;color:{couleur}">
        {result.ytd_productivite:.1%}</div>
        <div style="font-size:10px;color:#555">{result.periode_debut} → {result.periode_fin}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    m2.metric("Heures facturables",   f"{result.ytd_facturable:,.0f}h")
    m3.metric("Heures non facturables", f"{result.ytd_non_facturable:,.0f}h")
    m4.metric("Techniciens",          f"{result.nb_techniciens}")
    m5.metric("Équipes",              f"{result.nb_equipes}")


def _render_evolution_mensuelle(result: ProductiviteResult) -> None:
    st.markdown("### 📈 Évolution mensuelle — Productivité YTD")

    df = result.par_mois.copy()
    if df.empty:
        st.info("Aucune donnée mensuelle.")
        return

    # YTD cumulatif (Jan→M)
    df = df.sort_values("mois").reset_index(drop=True)
    df["fact_cum"]    = df["facturable"].cumsum()
    df["nonfact_cum"] = df["non_facturable"].cumsum()
    df["prod_ytd"]    = df["fact_cum"] / (df["fact_cum"] + df["nonfact_cum"]).replace(0, np.nan)
    df["prod_ytd"]    = df["prod_ytd"].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Productivité YTD cumulée**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["mois"], y=df["prod_ytd"] * 100,
            mode="lines+markers+text",
            text=[f"{v:.1%}" for v in df["prod_ytd"]],
            textposition="top center",
            textfont=dict(size=10),
            line=dict(color="#FFCD11", width=2),
            marker=dict(size=8),
            name="YTD",
        ))
        for seuil, label, color in [
            (SEUIL_EXCELLENT * 100, "Excellent", _COULEUR_EXCELLENT),
            (SEUIL_BON       * 100, "Bon",       _COULEUR_BON),
        ]:
            fig.add_hline(y=seuil, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_position="right")
        fig.update_layout(
            height=320, yaxis_ticksuffix="%", yaxis_range=[0, 100],
            plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
            font_color="#d8dce0", margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Répartition mensuelle des heures**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df["mois"], y=df["facturable"],
                              name="Facturable", marker_color="#22c55e"))
        fig2.add_trace(go.Bar(x=df["mois"], y=df["non_facturable"],
                              name="Non Facturable", marker_color="#ef4444"))
        fig2.update_layout(
            barmode="stack", height=320,
            plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
            font_color="#d8dce0", margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_barres_equipes(result: ProductiviteResult) -> None:
    st.markdown("### 🏢 Productivité par Équipe")

    df = result.par_equipe.copy()
    if df.empty:
        st.info("Aucune donnée équipe.")
        return

    df = df.sort_values("productivite", ascending=True)
    colors = [_couleur_prod(v) for v in df["productivite"]]

    fig = go.Figure(go.Bar(
        x=df["productivite"] * 100,
        y=df["equipe"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in df["productivite"]],
        textposition="outside",
        customdata=df[["nb_techniciens", "facturable", "non_facturable"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Productivité: %{x:.1f}%<br>"
            "Techniciens: %{customdata[0]}<br>"
            "Fact: %{customdata[1]:,.0f}h | Non-fact: %{customdata[2]:,.0f}h"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=SEUIL_EXCELLENT * 100, line_dash="dot",
                  line_color=_COULEUR_EXCELLENT, annotation_text="60%")
    fig.add_vline(x=SEUIL_BON * 100, line_dash="dot",
                  line_color=_COULEUR_BON, annotation_text="40%")
    fig.update_layout(
        height=max(300, 35 * len(df)),
        xaxis=dict(ticksuffix="%", range=[0, 110]),
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font_color="#d8dce0", margin=dict(t=20, b=20, l=200),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap_tech_mois(result: ProductiviteResult) -> None:
    st.markdown("### 🗓️ Heatmap Productivité — Technicien × Mois")

    df = result.par_tech_mois.copy()
    if df.empty:
        st.info("Aucune donnée.")
        return

    # Filtre équipe optionnel
    equipes = sorted(df["equipe"].unique().tolist())
    eq_sel  = st.selectbox("Équipe", ["Toutes"] + equipes, key="prod_hm_eq")
    if eq_sel != "Toutes":
        df = df[df["equipe"] == eq_sel]

    pivot = df.pivot_table(
        index="technicien", columns="mois",
        values="productivite", aggfunc="mean"
    ).fillna(0)

    fig = px.imshow(
        pivot * 100,
        color_continuous_scale=["#ef4444", "#f97316", "#FFCD11", "#22c55e"],
        zmin=0, zmax=100,
        aspect="auto",
        labels=dict(color="Productivité (%)"),
    )
    fig.update_traces(
        text=[[f"{v:.0f}%" if v > 0 else "" for v in row] for row in pivot.values * 100],
        texttemplate="%{text}",
    )
    fig.update_layout(
        height=max(350, 25 * len(pivot)),
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font_color="#d8dce0", margin=dict(t=20, b=40),
        coloraxis_colorbar=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_tableau_techniciens(result: ProductiviteResult) -> None:
    st.markdown("### 👷 Tableau Techniciens")

    df = result.par_technicien.copy()
    if df.empty:
        st.info("Aucune donnée.")
        return

    col1, col2 = st.columns([2, 2])
    with col1:
        search = st.text_input("🔍 Rechercher", key="prod_tech_search")
    with col2:
        equipes = ["Toutes"] + sorted(df["equipe"].unique().tolist())
        eq_sel  = st.selectbox("Équipe", equipes, key="prod_tech_eq")

    if search: df = df[df["technicien"].str.contains(search, case=False, na=False)]
    if eq_sel != "Toutes": df = df[df["equipe"] == eq_sel]

    df["Productivité"] = df["productivite"].apply(lambda v: f"{v:.1%}")
    df["Perf"]         = df["perf_label"]
    df["Fact (h)"]     = df["facturable"].round(1)
    df["Non-Fact (h)"] = df["non_facturable"].round(1)

    def _color_perf(val):
        m = {"Excellent": _COULEUR_EXCELLENT, "Bon": _COULEUR_BON,
             "Faible": _COULEUR_FAIBLE, "Critique": _COULEUR_CRITIQUE}
        c = m.get(val, "#888")
        return f"background-color:{c};color:white;font-weight:bold"

    display = df[["technicien","equipe","Productivité","Perf","Fact (h)","Non-Fact (h)","nb_jours"]].rename(
        columns={"technicien":"Technicien","equipe":"Équipe","nb_jours":"Jours"})

    styled = display.style.applymap(_color_perf, subset=["Perf"])
    st.dataframe(styled, use_container_width=True, height=400)
    st.caption(f"{len(display)} technicien(s) affiché(s)")


def _render_analyse_equipe(result: ProductiviteResult) -> None:
    st.markdown("### 🔬 Analyse par Équipe — Impact sur la Productivité Globale")

    df_em = result.par_equipe_mois.copy()
    df_eq = result.par_equipe.copy()

    if df_em.empty or df_eq.empty:
        st.info("Données insuffisantes.")
        return

    equipes = sorted(df_eq["equipe"].unique().tolist())
    eq_sel  = st.selectbox("Équipe à analyser", equipes, key="prod_eq_analyse")

    sub = df_em[df_em["equipe"] == eq_sel].sort_values("mois")
    if sub.empty:
        st.info("Aucune donnée pour cette équipe.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub["mois"], y=sub["productivite"] * 100,
            mode="lines+markers",
            line=dict(color="#FFCD11", width=2),
            marker=dict(size=8),
            name=eq_sel,
        ))
        fig.update_layout(
            title=f"Évolution mensuelle — {eq_sel}",
            height=280, yaxis_ticksuffix="%", yaxis_range=[0, 100],
            plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
            font_color="#d8dce0", margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Impact si retrait de cette équipe
        eq_data  = df_eq[df_eq["equipe"] == eq_sel].iloc[0]
        ytd_fact = result.ytd_facturable
        ytd_nf   = result.ytd_non_facturable
        ytd_prod = result.ytd_productivite

        fact_sans  = ytd_fact - eq_data["facturable"]
        nf_sans    = ytd_nf   - eq_data["non_facturable"]
        prod_sans  = fact_sans / (fact_sans + nf_sans) if (fact_sans + nf_sans) > 0 else 0
        delta      = prod_sans - ytd_prod

        couleur_delta = _COULEUR_EXCELLENT if delta > 0 else _COULEUR_CRITIQUE

        st.markdown(
            f"""<div style="background:#1a1a2e;border-radius:6px;padding:16px;margin-top:8px">
            <div style="font-size:12px;color:#888;margin-bottom:8px">
            Impact si retrait de <b style="color:#FFCD11">{eq_sel}</b></div>
            <div style="display:flex;gap:24px">
              <div>
                <div style="font-size:10px;color:#888">Prod. actuelle</div>
                <div style="font-size:22px;font-weight:700;color:{_couleur_prod(ytd_prod)}">{ytd_prod:.1%}</div>
              </div>
              <div style="font-size:22px;color:#444;padding-top:12px">→</div>
              <div>
                <div style="font-size:10px;color:#888">Sans cette équipe</div>
                <div style="font-size:22px;font-weight:700;color:{_couleur_prod(prod_sans)}">{prod_sans:.1%}</div>
              </div>
              <div>
                <div style="font-size:10px;color:#888">Delta</div>
                <div style="font-size:22px;font-weight:700;color:{couleur_delta}">
                {'+' if delta >= 0 else ''}{delta:.1%}</div>
              </div>
            </div>
            <div style="font-size:10px;color:#555;margin-top:8px">
            Fact: {eq_data['facturable']:,.0f}h | Non-fact: {eq_data['non_facturable']:,.0f}h
            </div></div>""",
            unsafe_allow_html=True,
        )
