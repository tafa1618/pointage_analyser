from __future__ import annotations

import traceback

import streamlit as st

from scoring.scorer import ORPerformanceScorer, ScoringError


st.set_page_config(page_title="OR Performance Analyzer", layout="wide")


@st.cache_resource(show_spinner=False)
def get_scorer() -> ORPerformanceScorer:
    return ORPerformanceScorer()


def main() -> None:
    st.title("OR Performance Analyzer")
    st.caption("Analyse OR-level : fusion IE + Pointage + BO, détection d'anomalies")

    st.markdown(
        """
        Chargez les 3 fichiers Excel complémentaires :
        - **IE**
        - **Pointage**
        - **BO**
        """
    )

    with st.sidebar:
        st.header("Données d'entrée")
        ie_file = st.file_uploader("Fichier IE (Excel)", type=["xlsx", "xls"], key="ie")
        pointage_file = st.file_uploader(
            "Fichier Pointage (Excel)", type=["xlsx", "xls"], key="pointage"
        )
        bo_file = st.file_uploader("Fichier BO (Excel)", type=["xlsx", "xls"], key="bo")
        run_btn = st.button("Lancer l'analyse", type="primary")

    if run_btn:
        if not all([ie_file, pointage_file, bo_file]):
            st.error("Veuillez charger les 3 fichiers Excel avant de lancer l'analyse.")
            return

        scorer = get_scorer()
        with st.spinner("Scoring en cours..."):
            try:
                enriched_df = scorer.run(ie_file, pointage_file, bo_file)
            except ScoringError as exc:
                st.error(str(exc))
                with st.expander("Détails techniques"):
                    st.code(traceback.format_exc())
                return
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Erreur inattendue: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return

        st.success("Analyse terminée avec succès.")

        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Total enregistrements", f"{len(enriched_df):,}".replace(",", " "))
        kpi_col2.metric(
            "Anomalies détectées",
            f"{int(enriched_df['final_anomaly_flag'].sum()):,}".replace(",", " "),
        )
        kpi_col3.metric(
            "Score moyen d'anomalie",
            f"{enriched_df['final_anomaly_score'].mean():.3f}",
        )

        st.subheader("Top anomalies")
        st.dataframe(
            enriched_df[
                [
                    "final_anomaly_score",
                    "final_anomaly_flag",
                    "rule_anomaly_score",
                    "ml_anomaly_score",
                ]
            ].head(50),
            use_container_width=True,
        )

        st.subheader("Dataset final unifié (1 ligne = 1 OR)")
        st.dataframe(enriched_df.head(200), use_container_width=True)

        csv_bytes = enriched_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger le dataset enrichi (CSV)",
            data=csv_bytes,
            file_name="or_performance_enriched_dataset.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
