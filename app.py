from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from dashboard.analytics import (
    DashboardAnalytics,
    DashboardAnalyticsError,
    apply_filters,
    build_monthly_metrics,
    compute_global_kpis,
)
from dashboard.visualizations import (
    anomaly_scatter_duration_efficiency,
    anomaly_scatter_rule_ml,
    global_charts,
    monthly_charts,
)
from scoring.scorer import ORPerformanceScorer, ScoringError


st.set_page_config(page_title="OR Performance Analyzer", layout="wide")


@st.cache_resource(show_spinner=False)
def get_scorer() -> ORPerformanceScorer:
    return ORPerformanceScorer()


@st.cache_resource(show_spinner=False)
def get_dashboard_analytics() -> DashboardAnalytics:
    return DashboardAnalytics()


def _render_figure(fig) -> None:
    if fig is None:
        st.info("Graphique indisponible (données insuffisantes).")
    else:
        st.plotly_chart(fig, use_container_width=True)


def _format_metric(value: float | int | None, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}".replace(",", " ")
    return f"{value:,}{suffix}".replace(",", " ")


def render_global_tab(frame: pd.DataFrame) -> None:
    kpis = compute_global_kpis(frame)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Nombre total OR", _format_metric(kpis["total_or"]))
    col2.metric("Nombre anomalies", _format_metric(kpis["anomalies"]))
    col3.metric("% anomalies", _format_metric(kpis["anomaly_pct"], "%"))
    col4.metric("Score moyen", _format_metric(kpis["score_moyen"]))
    col5.metric("Efficience moyenne", _format_metric(kpis["efficience_moyenne"]))
    col6.metric("Marge moyenne", _format_metric(kpis["marge_moyenne"]))

    charts = global_charts(frame)

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        _render_figure(charts["hist_duree"])
    with row1_col2:
        _render_figure(charts["hist_efficience"])
    with row1_col3:
        _render_figure(charts["hist_score"])

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        _render_figure(charts["box_efficience_type"])
    with row2_col2:
        _render_figure(charts["bar_taux_type"])

    _render_figure(charts["heatmap_corr"])


def render_advanced_eda_tab(frame: pd.DataFrame) -> None:
    st.subheader("Filtres dynamiques")

    col1, col2, col3 = st.columns(3)
    with col1:
        type_values = st.multiselect("Type OR", sorted(frame["dim_type_or"].dropna().unique().tolist()))
        loc_values = st.multiselect(
            "Localisation", sorted(frame["dim_localisation"].dropna().unique().tolist())
        )
    with col2:
        tech_values = st.multiselect(
            "Technicien", sorted(frame["dim_technicien"].dropna().unique().tolist())
        )
        model_values = st.multiselect("Modèle équipement", sorted(frame["dim_modele"].dropna().unique().tolist()))
    with col3:
        min_date = frame["date_reference"].min()
        max_date = frame["date_reference"].max()
        if pd.isna(min_date) or pd.isna(max_date):
            date_range = None
            st.info("Période indisponible (dates non renseignées).")
        else:
            selected = st.date_input("Période", value=(min_date.date(), max_date.date()))
            if isinstance(selected, tuple) and len(selected) == 2:
                date_range = (pd.Timestamp(selected[0]), pd.Timestamp(selected[1]))
            else:
                date_range = None

    filtered = apply_filters(frame, type_values, loc_values, tech_values, model_values, date_range)
    st.caption(f"Volume après filtres: {len(filtered)} OR")

    seg = (
        filtered.groupby("dim_segment", as_index=False)
        .agg(
            inefficience_moyenne=("surconsommation", "mean"),
            taux_anomalie=("is_anomaly", "mean"),
            score_moyen=("anomaly_score_global", "mean"),
        )
        .sort_values("inefficience_moyenne", ascending=False)
    )
    if not seg.empty:
        seg["taux_anomalie"] = seg["taux_anomalie"] * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Top segments les plus inefficients**")
        st.dataframe(seg[["dim_segment", "inefficience_moyenne"]].head(10), use_container_width=True)
    with c2:
        st.markdown("**Taux anomalie par segment**")
        st.dataframe(seg[["dim_segment", "taux_anomalie"]].head(10), use_container_width=True)
    with c3:
        st.markdown("**Moyenne score par segment**")
        st.dataframe(seg[["dim_segment", "score_moyen"]].head(10), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Distribution des retards pointage**")
        _render_figure(global_charts(filtered).get("hist_duree"))
    with d2:
        st.markdown("**Analyse des délais**")
        delays = filtered[["or_id", "delai_premier_pointage", "delai_dernier_pointage"]].copy()
        st.dataframe(delays.sort_values("delai_dernier_pointage", ascending=False).head(30), use_container_width=True)

    monthly = build_monthly_metrics(filtered)
    st.markdown("**Évolution mensuelle (OR, % anomalies, score moyen)**")
    _render_figure(monthly_charts(monthly))


def render_anomaly_tab(frame: pd.DataFrame) -> None:
    anomaly_filter = st.multiselect(
        "Filtrer par type anomalie",
        ["technique", "process", "financier", "ML"],
    )

    anomalies = frame[frame["is_anomaly"]].copy()
    if anomaly_filter:
        pattern = "|".join(anomaly_filter)
        anomalies = anomalies[anomalies["anomaly_types"].str.contains(pattern, case=False, na=False)]

    st.markdown("**Table triable des OR anormaux**")
    st.dataframe(
        anomalies[
            [
                "or_id",
                "dim_type_or",
                "anomaly_types",
                "anomaly_score_global",
                "anomaly_score_rule",
                "anomaly_score_ml",
                "surconsommation",
                "perte_potentielle",
            ]
        ].sort_values("anomaly_score_global", ascending=False),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        _render_figure(anomaly_scatter_duration_efficiency(anomalies))
    with c2:
        _render_figure(anomaly_scatter_rule_ml(anomalies))

    st.markdown("**Top 20 OR critiques**")
    st.dataframe(
        anomalies.sort_values("anomaly_score_global", ascending=False).head(20),
        use_container_width=True,
    )

    impact_col1, impact_col2 = st.columns(2)
    impact_col1.metric(
        "Somme surconsommation",
        _format_metric(float(anomalies["surconsommation"].fillna(0).sum())),
    )
    impact_col2.metric(
        "Estimation perte potentielle",
        _format_metric(float(anomalies["perte_potentielle"].fillna(0).sum())),
    )


def render_individual_or_tab(frame: pd.DataFrame) -> None:
    or_values = frame["or_id"].astype(str).sort_values().unique().tolist()
    selected_or = st.selectbox("Choisir un OR", or_values)

    row = frame[frame["or_id"].astype(str) == selected_or]
    if row.empty:
        st.warning("OR introuvable.")
        return

    or_row = row.iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Temps prévu", _format_metric(or_row.get("temps_prevu")))
    m2.metric("Temps réel", _format_metric(or_row.get("temps_reel")))
    m3.metric("Efficience", _format_metric(or_row.get("efficience_devis")))
    m4.metric("Surconsommation", _format_metric(or_row.get("surconsommation")))
    m5.metric("Marge estimée", _format_metric(or_row.get("marge_estimee")))

    d1, d2 = st.columns(2)
    d1.metric("Délai premier pointage", _format_metric(or_row.get("delai_premier_pointage")))
    d2.metric("Délai dernier pointage", _format_metric(or_row.get("delai_dernier_pointage")))

    st.markdown("**Scores détaillés**")
    score_df = pd.DataFrame(
        {
            "score": [
                "anomaly_score_technique",
                "anomaly_score_process",
                "anomaly_score_financier",
                "anomaly_score_ml",
                "anomaly_score_global",
            ],
            "valeur": [
                or_row.get("anomaly_score_technique"),
                or_row.get("anomaly_score_process"),
                or_row.get("anomaly_score_financier"),
                or_row.get("anomaly_score_ml"),
                or_row.get("anomaly_score_global"),
            ],
        }
    )
    st.dataframe(score_df, use_container_width=True)

    st.markdown("**Règles déclenchées**")
    triggered = []
    if bool(or_row.get("rule_negative_values", False)):
        triggered.append("Valeurs négatives détectées")
    if bool(or_row.get("rule_excessive_hours", False)):
        triggered.append("Durée excessive")
    if bool(or_row.get("rule_high_missing", False)):
        triggered.append("Taux élevé de valeurs manquantes")
    st.write(triggered if triggered else ["Aucune règle déclenchée"])

    percentile = (
        frame["anomaly_score_global"].rank(pct=True).loc[row.index].iloc[0] * 100
        if "anomaly_score_global" in frame.columns
        else None
    )
    st.metric("Position percentile (score global)", _format_metric(percentile, "%"))


def main() -> None:
    st.title("OR Performance Analyzer")
    st.caption("Dashboard EDA corporate multi-onglets pour pilotage OR")

    with st.sidebar:
        st.header("Données d'entrée")
        ie_file = st.file_uploader("Fichier IE (Excel)", type=["xlsx", "xls"], key="ie")
        pointage_file = st.file_uploader(
            "Fichier Pointage (Excel)", type=["xlsx", "xls"], key="pointage"
        )
        bo_file = st.file_uploader("Fichier BO (Excel)", type=["xlsx", "xls"], key="bo")
        run_btn = st.button("Lancer l'analyse", type="primary")

    if not run_btn:
        st.info("Chargez les trois fichiers Excel puis cliquez sur 'Lancer l'analyse'.")
        return

    if not all([ie_file, pointage_file, bo_file]):
        st.error("Veuillez charger les 3 fichiers Excel avant de lancer l'analyse.")
        return

    scorer = get_scorer()
    analytics = get_dashboard_analytics()

    with st.spinner("Construction du dataset OR-level et scoring..."):
        try:
            scored_df = scorer.run(ie_file, pointage_file, bo_file)
            dashboard_df = analytics.enrich(scored_df)
        except (ScoringError, DashboardAnalyticsError) as exc:
            st.error(str(exc))
            with st.expander("Détails techniques"):
                st.code(traceback.format_exc())
            return
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Erreur inattendue: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return

    tabs = st.tabs(["📊 Vue Globale", "🔎 EDA Avancé", "🚩 Analyse des Anomalies", "🔍 Analyse OR Individuel"])

    with tabs[0]:
        render_global_tab(dashboard_df)

    with tabs[1]:
        render_advanced_eda_tab(dashboard_df)

    with tabs[2]:
        render_anomaly_tab(dashboard_df)

    with tabs[3]:
        render_individual_or_tab(dashboard_df)

    st.divider()
    st.subheader("Export")
    csv_bytes = dashboard_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger le dataset enrichi (CSV)",
        data=csv_bytes,
        file_name="or_performance_enriched_dataset.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
