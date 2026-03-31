"""
app.py — Thin shell Streamlit (< 150 lignes)

Responsabilité unique : orchestrer l'UI.
Zéro logique métier. Tout délégué à scorer.ORPerformanceScorer.
"""

from __future__ import annotations

import hashlib
import io
import logging
import traceback

import pandas as pd
import streamlit as st

from pointage_analyzer.core.config import ScoringConfig
from pointage_analyzer.engine.scorer import ORPerformanceScorer, ScoringError
from pointage_analyzer.dashboard.analytics import (
    GlobalKPIs,
    apply_filters,
    build_equipe_stats,
    build_monthly_metrics,
    build_technicien_stats,
    compute_global_kpis,
)
from pointage_analyzer.dashboard.visualizations import (
    render_anomaly_scatter,
    render_global_charts,
    render_rule_breakdown,
    render_technicien_chart,
)
from pointage_analyzer.dashboard.exhaustivite import render_exhaustivite_tab
from pointage_analyzer.dashboard.efficience import render_efficience_tab
from pointage_analyzer.dashboard.productivite import render_productivite_tab




logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Pointage Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# SIDEBAR — Upload + Config
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Pointage Analyzer")
    st.caption("Contrôle des pointages & analyse d'efficience OR")
    st.markdown("---")

    st.subheader("📁 Fichier source")
    uploaded_file = st.file_uploader(
        "Fichier Excel (IE + Pointage + BO)",
        type=["xlsx", "xls"],
        help="Glisser-déposer le fichier contenant les 3 feuilles",
    )

    # Sélecteurs de feuilles (avec valeurs par défaut)
    if uploaded_file is not None:
        try:
            _xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
            _sheets = _xl.sheet_names
            uploaded_file.seek(0)
        except Exception:
            _sheets = ["IE", "Pointage", "BO"]

        with st.expander("🗂️ Noms des feuilles", expanded=False):
            sheet_ie       = st.selectbox("Feuille IE",       _sheets, index=_sheets.index("IE")       if "IE"       in _sheets else 0)
            sheet_pointage = st.selectbox("Feuille Pointage", _sheets, index=_sheets.index("Pointage") if "Pointage" in _sheets else min(1, len(_sheets)-1))
            sheet_bo       = st.selectbox("Feuille BO",       _sheets, index=_sheets.index("BO")       if "BO"       in _sheets else min(2, len(_sheets)-1))
    else:
        sheet_ie, sheet_pointage, sheet_bo = "IE", "Pointage", "BO"

    st.markdown("---")
    st.subheader("⚙️ Configuration")
    contamination = st.slider("Taux d'anomalie attendu (%)", 1, 20, 8, 1)
    rule_weight   = st.slider("Poids règles métier (%)", 10, 90, 45, 5)
    show_debug    = st.checkbox("Mode debug", value=False)

    run = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

config = ScoringConfig(
    contamination=contamination / 100,
    rule_weight=rule_weight / 100,
    ml_weight=1 - rule_weight / 100,
)

# -----------------------------------------------------------------------
# CACHE — évite de recalculer si le fichier n'a pas changé
# -----------------------------------------------------------------------
def _file_hash(f) -> str | None:
    if f is None:
        return None
    f.seek(0)
    h = hashlib.md5(f.read()).hexdigest()
    f.seek(0)
    return h


@st.cache_data(show_spinner="⏳ Analyse en cours…")
def _run_pipeline(file_bytes, sheet_ie, sheet_pointage, sheet_bo, contamination, rule_weight):
    """Wrappé dans cache_data pour memoïsation. Un seul fichier, 3 feuilles."""
    cfg = ScoringConfig(
        contamination=contamination,
        rule_weight=rule_weight,
        ml_weight=1 - rule_weight,
    )
    scorer = ORPerformanceScorer(config=cfg)
    # On recrée 3 BytesIO depuis les mêmes bytes (chaque read_excel consomme le curseur)
    ie_f  = io.BytesIO(file_bytes)
    pt_f  = io.BytesIO(file_bytes)
    bo_f  = io.BytesIO(file_bytes)
    return scorer.run(ie_f, pt_f, bo_f,
                      ie_sheet=sheet_ie,
                      pointage_sheet=sheet_pointage,
                      bo_sheet=sheet_bo)
_run_pipeline.clear()


# -----------------------------------------------------------------------
# LOGIQUE PRINCIPALE
# -----------------------------------------------------------------------
if uploaded_file is None:
    st.info(
        "👈 Glisser-déposer le fichier Excel **(IE + Pointage + BO)** "
        "dans la barre latérale, puis cliquer **🚀 Lancer l'analyse**."
    )
    st.stop()

if run or "pipeline_result" not in st.session_state:
    file_bytes = uploaded_file.read()

    try:
        result = _run_pipeline(
            file_bytes,
            sheet_ie, sheet_pointage, sheet_bo,
            config.contamination, config.rule_weight,
        )
        st.session_state["pipeline_result"] = result
        st.success(
            f"✅ Analyse complète — {result.metadata.get('nb_or_total', 0)} OR analysés "
            f"· {result.metadata.get('nb_techniciens', 0)} techniciens"
        )
    except ScoringError as exc:
        st.error(f"❌ Erreur pipeline: {exc}")
        if show_debug:
            st.code(traceback.format_exc())
        st.stop()

result      = st.session_state["pipeline_result"]
df_or       = result.df_or
df_presence = result.df_presence

# -----------------------------------------------------------------------
# GLOBAL KPIs
# -----------------------------------------------------------------------
kpis = compute_global_kpis(df_or, df_presence)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🔢 OR Total",       f"{kpis.nb_or_total}")
k2.metric("🚨 Anomalies",      f"{kpis.nb_anomalies}", f"{kpis.taux_anomalie:.1%}")
k3.metric("🔓 OR Ouverts",     f"{kpis.nb_or_ouverts}")
k4.metric("🔒 OR Clôturés",    f"{kpis.nb_or_clotures}")
k5.metric("👷 Techniciens",    f"{kpis.nb_techniciens}")
k6.metric("⏱️ Heures totales", f"{kpis.heures_totales:,.0f}h")

st.markdown("---")

# -----------------------------------------------------------------------
# ONGLETS
# -----------------------------------------------------------------------
tab_vue, tab_anomalies, tab_equipes, tab_tech, tab_exh, tab_eff, tab_prod = st.tabs([
    "📋 Vue OR",
    "🚨 Anomalies",
    "🏢 Équipes",
    "👷 Techniciens",
    "📅 Exhaustivité",
    "⚡ Efficience",
    "🎯 Productivité",
])

# --- TAB 1 : Vue OR ---
with tab_vue:
    st.subheader("Tableau des Ordres de Réparation")

    with st.expander("Filtres", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        positions = sorted(df_or["position"].dropna().unique()) if "position" in df_or.columns else []
        pos_sel = col_f1.multiselect("Position", positions, default=positions)
        anomaly_only = col_f2.checkbox("Anomalies uniquement")
        equipes = sorted(df_or["equipe_principale"].dropna().unique()) if "equipe_principale" in df_or.columns else []
        eq_sel = col_f3.multiselect("Équipe principale", equipes)

    df_view = apply_filters(
        df_or,
        position_filter=pos_sel or None,
        anomaly_only=anomaly_only,
        equipe_filter=eq_sel or None,
    )

    display_cols = [c for c in [
        "or_id", "type_or", "position", "nature_materiel",
        "technicien_principal_nom", "equipe_principale", "nb_techniciens",
        "total_heures", "score_final", "severity", "rule_anomaly_types",
    ] if c in df_view.columns]

    st.dataframe(df_view[display_cols], use_container_width=True, height=500)

    csv = df_view.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("⬇️ Exporter CSV", csv, "or_dataset.csv", "text/csv")

# --- TAB 2 : Anomalies ---
with tab_anomalies:
    st.subheader("🚨 Analyse des Anomalies")
    charts = render_global_charts(df_or)

    if "score_distribution" in charts:
        st.plotly_chart(charts["score_distribution"], use_container_width=True)

    c1, c2 = st.columns(2)
    scatter = render_anomaly_scatter(df_or)
    if scatter:
        c1.plotly_chart(scatter, use_container_width=True)
    rule_chart = render_rule_breakdown(df_or)
    if rule_chart:
        c2.plotly_chart(rule_chart, use_container_width=True)

    st.subheader("Détail des OR anomaliques")
    df_anom = df_or[df_or["anomaly_flag"]] if "anomaly_flag" in df_or.columns else pd.DataFrame()
    if not df_anom.empty:
        anom_cols = [c for c in [
            "or_id", "severity", "score_final", "rule_score_total", "ml_score",
            "rule_anomaly_types", "technicien_principal_nom", "equipe_principale",
        ] if c in df_anom.columns]
        st.dataframe(
            df_anom[anom_cols].sort_values("score_final", ascending=False),
            use_container_width=True,
            height=400,
        )

# --- TAB 3 : Équipes ---
with tab_equipes:
    st.subheader("🏢 Performance par Équipe")
    equipe_stats = build_equipe_stats(df_or)
    if not equipe_stats.empty:
        if "equipe_scores" in charts:
            st.plotly_chart(charts["equipe_scores"], use_container_width=True)
        st.dataframe(equipe_stats, use_container_width=True)
    else:
        st.info("Données équipe indisponibles (colonne equipe_principale manquante).")

# --- TAB 4 : Techniciens ---
with tab_tech:
    st.subheader("👷 Performance par Technicien")
    tech_stats = build_technicien_stats(df_or)
    if not tech_stats.empty:
        fig_tech = render_technicien_chart(tech_stats)
        if fig_tech:
            st.plotly_chart(fig_tech, use_container_width=True)
        st.dataframe(tech_stats, use_container_width=True)
    else:
        st.info("Données technicien indisponibles.")

# --- TAB 5 : Exhaustivité ---
with tab_exh:
    render_exhaustivite_tab(df_presence, config)

# --- TAB 6 : Efficience ---
with tab_eff:
    if result.efficience:
        render_efficience_tab(result.efficience, config)
    else:
        st.info("Charger le fichier BO pour activer l'analyse d'efficience.")

# --- TAB 7 : Productivité ---
with tab_prod:
    if result.productivite:
        render_productivite_tab(result.productivite)
    else:
        st.info("Charger le fichier Pointage pour activer l'analyse de productivité.")
