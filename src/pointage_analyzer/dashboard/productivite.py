"""
Onglet Productivité — analyse des heures facturables vs totales.

Formule officielle CAT (Rolling 12 mois) :
    Productivité = Σ Facturable / (Σ Facturable + Σ Non Facturable)
    Périmètre : 12 mois glissants à partir du dernier mois disponible

Seuils CAT DDC (Group 2) :
    Emerging  ≥ 78%
    Advanced  ≥ 82%
    Excellent ≥ 85%

Filtre année : sélection de l'année → recalcul complet sur les 12 mois
glissants se terminant au dernier mois disponible de l'année choisie.
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
)

# ── Seuils CAT officiels ──────────────────────────────────────────────
SEUIL_EXCELLENT = 0.85
SEUIL_ADVANCED  = 0.82
SEUIL_EMERGING  = 0.78

# ── Couleurs ──────────────────────────────────────────────────────────
_C_EXCELLENT = "#22c55e"
_C_ADVANCED  = "#FFCD11"
_C_EMERGING  = "#f97316"
_C_BASIC     = "#ef4444"


def _couleur(ratio: float) -> str:
    if ratio >= SEUIL_EXCELLENT: return _C_EXCELLENT
    elif ratio >= SEUIL_ADVANCED:  return _C_ADVANCED
    elif ratio >= SEUIL_EMERGING:  return _C_EMERGING
    else:                           return _C_BASIC


def _label(ratio: float) -> str:
    if ratio >= SEUIL_EXCELLENT: return "Excellent"
    elif ratio >= SEUIL_ADVANCED:  return "Advanced"
    elif ratio >= SEUIL_EMERGING:  return "Emerging"
    else:                           return "Basic"


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _filter_by_year(pt_harm: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filtre le DataFrame pointage sur l'année sélectionnée."""
    date_col = "date_saisie" if "date_saisie" in pt_harm.columns else "Saisie heures - Date"
    dates = pd.to_datetime(pt_harm[date_col], errors="coerce")
    return pt_harm[dates.dt.year == year].copy()


def _filter_rolling_12m(pt_harm: pd.DataFrame, year: int) -> tuple[pd.DataFrame, str, str]:
    """
    Retourne les données des 12 mois glissants se terminant au dernier
    mois disponible de l'année sélectionnée.

    Ex: année 2026, données jusqu'en mars → Apr 2025 → Mar 2026
    Ex: année 2025, données jusqu'en déc  → Jan 2025 → Déc 2025

    Returns:
        (df_filtré, date_debut_str, date_fin_str)
    """
    date_col = "date_saisie" if "date_saisie" in pt_harm.columns else "Saisie heures - Date"
    dates = pd.to_datetime(pt_harm[date_col], errors="coerce")

    # Dernier mois disponible dans l'année sélectionnée
    dates_annee = dates[dates.dt.year == year].dropna()
    if dates_annee.empty:
        return pd.DataFrame(), "", ""

    dernier_mois = dates_annee.max().to_period("M")
    premier_mois = dernier_mois - 11  # 12 mois glissants

    # Filtre sur la période rolling
    mois_series = dates.dt.to_period("M")
    mask = (mois_series >= premier_mois) & (mois_series <= dernier_mois)

    df_filtered = pt_harm[mask.fillna(False)].copy()
    debut = premier_mois.start_time.strftime("%Y-%m-%d")
    fin   = dernier_mois.end_time.strftime("%Y-%m-%d")

    return df_filtered, debut, fin


def render_productivite_tab(productivite: ProductiviteResult, pt_harm: pd.DataFrame) -> None:
    """Point d'entrée depuis app.py."""
    st.markdown("## 📊 Analyse de la Productivité")

    # ── Sélecteur année ───────────────────────────────────────────────
    date_col = "date_saisie" if "date_saisie" in pt_harm.columns else "Saisie heures - Date"
    dates_all = pd.to_datetime(pt_harm[date_col], errors="coerce")
    annees_dispo = sorted(dates_all.dt.year.dropna().unique().astype(int).tolist(), reverse=True)

    if not annees_dispo:
        st.error("Aucune date valide dans les données.")
        return

    col_ann, col_info = st.columns([1, 3])
    with col_ann:
        annee_sel = st.selectbox("📅 Année", options=annees_dispo, index=0, key="prod_annee")
    with col_info:
        st.caption(
            "**Rolling 12 mois** — périmètre : 12 mois glissants se terminant au dernier mois "
            f"disponible de {annee_sel}. Seuils CAT : Emerging ≥78% · Advanced ≥82% · Excellent ≥85%"
        )

    # ── Données année sélectionnée (pour tout sauf le KPI global) ────
    pt_annee = _filter_by_year(pt_harm, annee_sel)
    if pt_annee.empty:
        st.warning(f"Aucune donnée pour {annee_sel}.")
        return

    builder = ProductiviteBuilder()
    result  = builder.build(pt_annee)

    if result.ytd_facturable == 0 and result.ytd_non_facturable == 0:
        st.warning("Données insuffisantes pour calculer la productivité.")
        return

    # ── KPI global Rolling 12M (calcul séparé) ───────────────────────
    pt_rolling, debut_rolling, fin_rolling = _filter_rolling_12m(pt_harm, annee_sel)
    if not pt_rolling.empty:
        result_rolling = builder.build(pt_rolling)
        prod_rolling   = result_rolling.ytd_productivite
        fact_rolling   = result_rolling.ytd_facturable
        nf_rolling     = result_rolling.ytd_non_facturable
    else:
        prod_rolling = result.ytd_productivite
        fact_rolling = result.ytd_facturable
        nf_rolling   = result.ytd_non_facturable
        debut_rolling = result.periode_debut
        fin_rolling   = result.periode_fin

    _render_kpi_cards(prod_rolling, fact_rolling, nf_rolling, debut_rolling, fin_rolling, annee_sel)
    st.divider()
    _render_evolution_mensuelle(result)
    st.divider()
    _render_barres_equipes(result)
    st.divider()
    _render_heatmap_tech_mois(result)
    st.divider()
    _render_tableau_techniciens(result)
    st.divider()
    _render_simulateur_global(result)
    st.divider()
    _render_analyse_proxy(result)


# ══════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════

def _render_kpi_cards(
    prod_rolling: float,
    fact_rolling: float,
    nf_rolling: float,
    debut: str,
    fin: str,
    annee: int,
) -> None:
    couleur = _couleur(prod_rolling)
    niveau  = _label(prod_rolling)

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f"""<div style="background:#1a1a2e;border-left:4px solid {couleur};
        border-radius:4px;padding:12px 16px">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px">
        Productivité Rolling 12M {annee}</div>
        <div style="font-size:32px;font-weight:700;color:{couleur}">
        {prod_rolling:.1%}</div>
        <div style="font-size:11px;font-weight:600;color:{couleur};margin-top:2px">{niveau}</div>
        <div style="font-size:10px;color:#555;margin-top:2px">{debut} → {fin}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    m2.metric("Heures facturables (12M)",     f"{fact_rolling:,.0f}h")
    m3.metric("Heures non facturables (12M)", f"{nf_rolling:,.0f}h")

    # Barre de progression
    p = prod_rolling * 100
    st.markdown(
        f"""<div style="margin:10px 0 4px;display:flex;justify-content:space-between;font-size:10px;color:#666">
        <span>0%</span>
        <span style="color:{_C_EMERGING}">Emerging 78%</span>
        <span style="color:{_C_ADVANCED}">Advanced 82%</span>
        <span style="color:{_C_EXCELLENT}">Excellent 85%</span>
        <span>100%</span></div>
        <div style="height:8px;background:#222;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:{min(p,100):.1f}%;background:{couleur};border-radius:4px"></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# ÉVOLUTION MENSUELLE
# ══════════════════════════════════════════════════════════════════════

def _render_evolution_mensuelle(result: ProductiviteResult) -> None:
    st.markdown("### 📈 Évolution mensuelle — Productivité Rolling 12M")

    df = result.par_mois.copy()
    if df.empty:
        st.info("Aucune donnée mensuelle.")
        return

    df = df.sort_values("mois").reset_index(drop=True)
    df["fact_cum"]    = df["facturable"].cumsum()
    df["nonfact_cum"] = df["non_facturable"].cumsum()
    df["prod_rolling"] = df["fact_cum"] / (df["fact_cum"] + df["nonfact_cum"]).replace(0, np.nan)
    df["prod_rolling"] = df["prod_rolling"].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Productivité Rolling cumulée**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["mois"], y=df["prod_rolling"] * 100,
            mode="lines+markers+text",
            text=[f"{v:.1%}" for v in df["prod_rolling"]],
            textposition="top center",
            textfont=dict(size=10),
            line=dict(color="#FFCD11", width=2),
            marker=dict(size=8),
            name="Rolling",
        ))
        for seuil, label, color in [
            (SEUIL_EXCELLENT * 100, "Excellent 85%", _C_EXCELLENT),
            (SEUIL_ADVANCED  * 100, "Advanced 82%",  _C_ADVANCED),
            (SEUIL_EMERGING  * 100, "Emerging 78%",  _C_EMERGING),
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


# ══════════════════════════════════════════════════════════════════════
# BARRES ÉQUIPES
# ══════════════════════════════════════════════════════════════════════

def _render_barres_equipes(result: ProductiviteResult) -> None:
    st.markdown("### 🏢 Productivité par Équipe")

    df = result.par_equipe.copy()
    if df.empty:
        st.info("Aucune donnée équipe.")
        return

    df = df.sort_values("productivite", ascending=True)
    colors = [_couleur(v) for v in df["productivite"]]

    fig = go.Figure(go.Bar(
        x=df["productivite"] * 100, y=df["equipe"],
        orientation="h", marker_color=colors,
        text=[f"{v:.1%}" for v in df["productivite"]],
        textposition="outside",
        customdata=df[["nb_techniciens", "facturable", "non_facturable"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Productivité: %{x:.1f}%<br>"
            "Techniciens: %{customdata[0]}<br>"
            "Fact: %{customdata[1]:,.0f}h | Non-fact: %{customdata[2]:,.0f}h<extra></extra>"
        ),
    ))
    for seuil, label, color in [
        (SEUIL_EXCELLENT * 100, "85%", _C_EXCELLENT),
        (SEUIL_ADVANCED  * 100, "82%", _C_ADVANCED),
        (SEUIL_EMERGING  * 100, "78%", _C_EMERGING),
    ]:
        fig.add_vline(x=seuil, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top")
    fig.update_layout(
        height=max(300, 35 * len(df)),
        xaxis=dict(ticksuffix="%", range=[0, 115]),
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font_color="#d8dce0", margin=dict(t=20, b=20, l=200),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# HEATMAP TECH × MOIS — couleurs discrètes non agressives
# ══════════════════════════════════════════════════════════════════════

def _render_heatmap_tech_mois(result: ProductiviteResult) -> None:
    st.markdown("### 🗓️ Heatmap Productivité — Technicien × Mois")

    df = result.par_tech_mois.copy()
    if df.empty:
        st.info("Aucune donnée.")
        return

    equipes = sorted(df["equipe"].unique().tolist())
    eq_sel  = st.selectbox("Équipe", ["Toutes"] + equipes, key="prod_hm_eq")
    if eq_sel != "Toutes":
        df = df[df["equipe"] == eq_sel]

    pivot = df.pivot_table(
        index="technicien", columns="mois",
        values="productivite", aggfunc="mean"
    ).fillna(np.nan)

    # Colorscale sobre : blanc → bleu clair → bleu foncé
    # Pas de rouge/vert agressif — juste une intensité de bleu
    fig = px.imshow(
        pivot * 100,
        color_continuous_scale=[
            [0.0,  "#2d1b1b"],   # très faible — bordeaux sombre
            [0.3,  "#7a3535"],   # faible — bordeaux
            [0.5,  "#b87333"],   # moyen — bronze
            [0.78, "#4a7c59"],   # emerging — vert olive discret
            [0.82, "#2d6a4f"],   # advanced — vert forêt
            [1.0,  "#1b4332"],   # excellent — vert foncé
        ],
        zmin=0, zmax=100,
        aspect="auto",
        labels=dict(color="Prod. (%)"),
    )
    fig.update_traces(
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row]
              for row in (pivot.values * 100)],
        texttemplate="%{text}",
        textfont=dict(size=9, color="rgba(255,255,255,0.85)"),
    )
    fig.update_layout(
        height=max(350, 22 * len(pivot)),
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font_color="#d8dce0", margin=dict(t=20, b=40),
        coloraxis_colorbar=dict(ticksuffix="%", len=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TABLEAU TECHNICIENS
# ══════════════════════════════════════════════════════════════════════

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

    if search:        df = df[df["technicien"].str.contains(search, case=False, na=False)]
    if eq_sel != "Toutes": df = df[df["equipe"] == eq_sel]

    df["Productivité"] = df["productivite"].apply(_pct)
    df["Niveau"]       = df["productivite"].apply(_label)

    def _color_niveau(val):
        m = {"Excellent": _C_EXCELLENT, "Advanced": _C_ADVANCED,
             "Emerging": _C_EMERGING, "Basic": _C_BASIC}
        c = m.get(val, "#888")
        return f"background-color:{c};color:white;font-weight:bold"

    display = df[["technicien","equipe","Productivité","Niveau","facturable","non_facturable","nb_jours"]].rename(
        columns={"technicien":"Technicien","equipe":"Équipe",
                 "facturable":"Fact (h)","non_facturable":"Non-Fact (h)","nb_jours":"Jours"})
    display["Fact (h)"]     = display["Fact (h)"].round(1)
    display["Non-Fact (h)"] = display["Non-Fact (h)"].round(1)

    styled = display.style.applymap(_color_niveau, subset=["Niveau"])
    st.dataframe(styled, use_container_width=True, height=400)
    st.caption(f"{len(display)} technicien(s) affiché(s)")


# ══════════════════════════════════════════════════════════════════════
# SIMULATEUR PÉRIMÈTRE DYNAMIQUE
# ══════════════════════════════════════════════════════════════════════

def _render_simulateur_global(result: ProductiviteResult) -> None:
    """
    Sélection d'équipes → recalcul du KPI global en temps réel.
    Utile pour exclure le CRC ou toute autre équipe du périmètre.
    """
    st.markdown("### 🎛️ Simulateur Productivité Globale")
    st.caption(
        "Sélectionnez les équipes à inclure dans le calcul — "
        "utile pour exclure le CRC ou toute autre équipe du périmètre."
    )

    df_eq = result.par_equipe.copy()
    if df_eq.empty:
        st.info("Pas de données équipes.")
        return

    equipes_toutes = sorted(df_eq["equipe"].tolist())

    # Identifier équipes CRC par défaut
    crc_keywords = ["remontage transmission", "crc"]
    crc_default  = [e for e in equipes_toutes if any(k in e.lower() for k in crc_keywords)]

    col_sel, col_kpi = st.columns([2, 1])

    with col_sel:
        equipes_sel = st.multiselect(
            "Équipes incluses dans le calcul",
            options=equipes_toutes,
            default=[e for e in equipes_toutes if e not in crc_default],
            key="prod_sim_equipes",
        )

    if not equipes_sel:
        with col_kpi:
            st.warning("Sélectionnez au moins une équipe.")
        return

    df_sel    = df_eq[df_eq["equipe"].isin(equipes_sel)]
    f_sel     = df_sel["facturable"].sum()
    nf_sel    = df_sel["non_facturable"].sum()
    prod_sel  = f_sel / (f_sel + nf_sel) if (f_sel + nf_sel) > 0 else 0.0

    f_all     = df_eq["facturable"].sum()
    nf_all    = df_eq["non_facturable"].sum()
    prod_all  = f_all / (f_all + nf_all) if (f_all + nf_all) > 0 else 0.0
    delta     = prod_sel - prod_all

    with col_kpi:
        st.metric(
            label="🎯 Productivité périmètre sélectionné",
            value=_pct(prod_sel),
            delta=f"{delta:+.1%} vs toutes équipes",
        )

    # Tableau détail incluses/exclues
    df_eq["Statut"] = df_eq["equipe"].apply(
        lambda e: "✅ Incluse" if e in equipes_sel else "❌ Exclue"
    )

    def _color_statut(val):
        if val == "✅ Incluse": return "background-color:#d4edda;color:#155724"
        return "background-color:#f8d7da;color:#721c24"

    display = df_eq[["Statut","equipe","facturable","non_facturable","productivite"]].copy().rename(columns={
        "equipe": "Équipe", "facturable": "Fact. (h)",
        "non_facturable": "Non Fact. (h)", "productivite": "Productivité",
    })
    styled = (
        display.style
        .format({"Fact. (h)": "{:.1f}", "Non Fact. (h)": "{:.1f}", "Productivité": "{:.1%}"})
        .applymap(_color_statut, subset=["Statut"])
    )
    st.dataframe(styled, use_container_width=True, height=350)


# ══════════════════════════════════════════════════════════════════════
# ANALYSE ÉQUIPE PROXY
# ══════════════════════════════════════════════════════════════════════

def _render_analyse_proxy(result: ProductiviteResult) -> None:
    """
    Corrélation productivité équipe vs global (mensuel) +
    impact simulé si retrait de chaque équipe.
    """
    st.markdown("### 🔬 Analyse Équipe Proxy")
    st.caption(
        "Quelle équipe tire (ou plombe) la productivité globale ? "
        "Corrélation mensuelle + impact simulé si retrait."
    )

    df_em = result.par_equipe_mois.copy()
    df_eq = result.par_equipe.copy()
    df_m  = result.par_mois.copy()

    if df_em.empty or df_m.empty:
        st.info("Pas assez de données pour l'analyse proxy.")
        return

    # ── Corrélation mensuelle équipe vs global ────────────────────────
    pivot = df_em.pivot_table(
        index="mois", columns="equipe", values="productivite"
    ).fillna(0)
    pivot = pivot.join(df_m.set_index("mois")["productivite"].rename("_global"))

    corr_series = pivot.corr()["_global"].drop("_global").sort_values(ascending=False)
    corr_df     = corr_series.reset_index().dropna()
    corr_df.columns = ["equipe", "correlation"]

    # ── Impact retrait équipe sur prod globale ────────────────────────
    f_all    = df_eq["facturable"].sum()
    nf_all   = df_eq["non_facturable"].sum()
    prod_all = f_all / (f_all + nf_all) if (f_all + nf_all) > 0 else 0.0

    impacts = []
    for _, row in df_eq.iterrows():
        f_sans    = f_all  - row["facturable"]
        nf_sans   = nf_all - row["non_facturable"]
        prod_sans = f_sans / (f_sans + nf_sans) if (f_sans + nf_sans) > 0 else 0.0
        impacts.append({"equipe": row["equipe"], "prod_sans": prod_sans,
                        "delta": prod_sans - prod_all})
    impact_df = pd.DataFrame(impacts).sort_values("delta", ascending=False)

    # ── Fusion ────────────────────────────────────────────────────────
    proxy_df = corr_df.merge(impact_df, on="equipe", how="outer")
    proxy_df["prod_ytd"] = proxy_df["equipe"].map(df_eq.set_index("equipe")["productivite"])

    col_corr, col_impact = st.columns(2)

    with col_corr:
        st.markdown("#### 📊 Corrélation vs productivité globale")
        st.caption(f"Calculée sur {len(df_m)} mois disponibles")

        if corr_df.empty:
            st.info("Pas assez de mois pour calculer une corrélation.")
        else:
            corr_sorted = corr_df.sort_values("correlation", ascending=True)
            colors_c = [
                "rgba(34,197,94,0.7)"  if v >= 0.7 else
                "rgba(249,115,22,0.7)" if v >= 0.3 else
                "rgba(239,68,68,0.7)"
                for v in corr_sorted["correlation"]
            ]
            fig_c = go.Figure(go.Bar(
                x=corr_sorted["correlation"],
                y=corr_sorted["equipe"],
                orientation="h",
                marker_color=colors_c,
                text=[f"{v:.2f}" for v in corr_sorted["correlation"]],
                textposition="outside",
            ))
            fig_c.add_vline(x=0, line_color="#555")
            fig_c.add_vline(x=0.7,  line_dash="dot", line_color=_C_EXCELLENT, annotation_text="Fort +")
            fig_c.add_vline(x=-0.7, line_dash="dot", line_color=_C_BASIC,     annotation_text="Fort −")
            fig_c.update_layout(
                height=max(280, 35 * len(corr_sorted)),
                xaxis=dict(range=[-1.1, 1.3]),
                plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
                font_color="#d8dce0", margin=dict(t=20, b=20, l=200),
            )
            st.plotly_chart(fig_c, use_container_width=True)

        nb_mois = len(df_m)
        if nb_mois < 6:
            st.warning(f"⚠️ Seulement {nb_mois} mois — corrélation peu fiable (< 6 mois recommandés).")

    with col_impact:
        st.markdown("#### 💡 Impact si retrait de l'équipe")
        st.caption(f"Prod. actuelle : **{_pct(prod_all)}**")

        impact_sorted = impact_df.sort_values("delta", ascending=True)
        colors_i = [_C_EXCELLENT if v > 0 else _C_BASIC for v in impact_sorted["delta"]]

        fig_i = go.Figure(go.Bar(
            x=impact_sorted["delta"] * 100,
            y=impact_sorted["equipe"],
            orientation="h",
            marker_color=colors_i,
            text=[f"{v:+.1%}" for v in impact_sorted["delta"]],
            textposition="outside",
        ))
        fig_i.add_vline(x=0, line_color="#555")
        fig_i.update_layout(
            height=max(280, 35 * len(impact_sorted)),
            xaxis=dict(ticksuffix="%"),
            plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
            font_color="#d8dce0", margin=dict(t=20, b=20, l=200),
        )
        st.plotly_chart(fig_i, use_container_width=True)

    # Tableau fusion
    with st.expander("📋 Tableau synthèse corrélation + impact"):
        display = proxy_df.copy()
        display["Corrélation"]  = display["correlation"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        display["Delta retrait"] = display["delta"].apply(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")
        display["Prod. équipe"]  = display["prod_ytd"].apply(lambda v: _pct(v) if pd.notna(v) else "—")
        st.dataframe(
            display[["equipe","Prod. équipe","Corrélation","Delta retrait"]].rename(
                columns={"equipe": "Équipe"}),
            use_container_width=True,
        )
