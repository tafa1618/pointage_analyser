"""Dashboard analytics — prépare les KPIs et métriques pour l'affichage Streamlit."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DashboardAnalyticsError(Exception):
    """Raised when dashboard analytics preparation fails."""


@dataclass(slots=True)
class GlobalKPIs:
    """KPIs globaux affichés en haut du dashboard."""
    nb_or_total: int
    nb_anomalies: int
    taux_anomalie: float
    nb_or_ouverts: int
    nb_or_clotures: int
    nb_techniciens: int
    nb_equipes: int
    heures_totales: float
    score_moyen: float


def compute_global_kpis(df_or: pd.DataFrame, df_presence: pd.DataFrame) -> GlobalKPIs:
    """Calcule les KPIs globaux depuis les deux DataFrames."""
    nb_total = len(df_or)
    nb_anomalies = int(df_or.get("anomaly_flag", pd.Series(False)).sum())
    taux = nb_anomalies / nb_total if nb_total > 0 else 0.0

    # EC = En Cours | TT = Travaux Terminés | AF = Attente Facturation
    # CP = Comptabilisé (seul statut réellement clôturé — irréversible)
    nb_ouverts = int(
    df_or["position"].isin(["EC", "TT", "AF"]).sum()
    ) if "position" in df_or.columns else 0
    nb_clotures = int(
    (df_or["position"] == "CP").sum()
) if "position" in df_or.columns else 0f "position" in df_or.columns else 0

    nb_tech = (
        df_presence["salarie_nom"].nunique()
        if "salarie_nom" in df_presence.columns else 0
    )
    nb_equipes = (
        df_presence["equipe_nom"].nunique()
        if "equipe_nom" in df_presence.columns else 0
    )

    heures = float(df_or.get("total_heures", pd.Series(0.0)).sum())
    score_moy = float(df_or.get("score_final", pd.Series(0.0)).mean())

    return GlobalKPIs(
        nb_or_total=nb_total,
        nb_anomalies=nb_anomalies,
        taux_anomalie=taux,
        nb_or_ouverts=nb_ouverts,
        nb_or_clotures=nb_clotures,
        nb_techniciens=nb_tech,
        nb_equipes=nb_equipes,
        heures_totales=heures,
        score_moyen=score_moy,
    )


def apply_filters(
    df_or: pd.DataFrame,
    position_filter: list[str] | None = None,
    anomaly_only: bool = False,
    severity_filter: list[str] | None = None,
    type_or_filter: list[str] | None = None,
    equipe_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Applique les filtres de l'UI sur le dataset OR."""
    df = df_or.copy()

    if anomaly_only and "anomaly_flag" in df.columns:
        df = df[df["anomaly_flag"]]

    if position_filter and "position" in df.columns:
        df = df[df["position"].isin(position_filter)]

    if severity_filter and "severity" in df.columns:
        df = df[df["severity"].isin(severity_filter)]

    if type_or_filter and "type_or" in df.columns:
        df = df[df["type_or"].isin(type_or_filter)]

    if equipe_filter and "equipe_principale" in df.columns:
        df = df[df["equipe_principale"].isin(equipe_filter)]

    return df


def build_technicien_stats(df_or: pd.DataFrame) -> pd.DataFrame:
    """Agrège les statistiques OR par technicien principal."""
    if "technicien_principal_nom" not in df_or.columns:
        return pd.DataFrame()

    stats = (
        df_or.groupby("technicien_principal_nom", as_index=False)
        .agg(
            nb_or=("or_id", "size"),
            nb_anomalies=("anomaly_flag", "sum") if "anomaly_flag" in df_or.columns else ("or_id", "size"),
            heures_totales=("total_heures", "sum") if "total_heures" in df_or.columns else ("or_id", "size"),
            score_moyen=("score_final", "mean") if "score_final" in df_or.columns else ("or_id", "first"),
        )
    )
    if "nb_anomalies" in stats.columns and "nb_or" in stats.columns:
        stats["taux_anomalie"] = stats["nb_anomalies"] / stats["nb_or"].replace(0, np.nan)
    if "equipe_principale" in df_or.columns:
        equipe_map = (
            df_or.groupby("technicien_principal_nom")["equipe_principale"].first()
        )
        stats["equipe"] = stats["technicien_principal_nom"].map(equipe_map)

    return stats.sort_values("nb_anomalies", ascending=False)


def build_equipe_stats(df_or: pd.DataFrame) -> pd.DataFrame:
    """Agrège les statistiques OR par équipe principale."""
    if "equipe_principale" not in df_or.columns:
        return pd.DataFrame()

    stats = (
        df_or.groupby("equipe_principale", as_index=False)
        .agg(
            nb_or=("or_id", "size"),
            nb_anomalies=("anomaly_flag", "sum") if "anomaly_flag" in df_or.columns else ("or_id", "size"),
            heures_totales=("total_heures", "sum") if "total_heures" in df_or.columns else ("or_id", "size"),
            score_moyen=("score_final", "mean") if "score_final" in df_or.columns else ("or_id", "first"),
            nb_techniciens=("technicien_principal_nom", "nunique") if "technicien_principal_nom" in df_or.columns else ("or_id", "size"),
        )
    )
    if "nb_anomalies" in stats.columns:
        stats["taux_anomalie"] = stats["nb_anomalies"] / stats["nb_or"].replace(0, np.nan)

    return stats.sort_values("taux_anomalie", ascending=False)


def build_monthly_metrics(df_or: pd.DataFrame) -> pd.DataFrame:
    """Métriques mensuelles basées sur la date du premier pointage."""
    date_col = next(
        (c for c in ["date_premier_pointage", "date_creation_min"] if c in df_or.columns),
        None,
    )
    if date_col is None:
        return pd.DataFrame()

    df = df_or.copy()
    df["_mois"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")
    df = df[df["_mois"].notna()]

    monthly = (
        df.groupby("_mois", as_index=False)
        .agg(
            nb_or=("or_id", "size"),
            nb_anomalies=("anomaly_flag", "sum") if "anomaly_flag" in df.columns else ("or_id", "size"),
            heures=("total_heures", "sum") if "total_heures" in df.columns else ("or_id", "size"),
            score_moyen=("score_final", "mean") if "score_final" in df.columns else ("or_id", "first"),
        )
        .rename(columns={"_mois": "mois"})
    )
    monthly["mois_label"] = monthly["mois"].astype(str)
    return monthly.sort_values("mois")
