from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class DashboardAnalyticsError(Exception):
    """Raised when dashboard analytics preparation fails."""


@dataclass(slots=True)
class ColumnResolver:
    """Resolve semantic fields from heterogeneous source schemas."""

    frame: pd.DataFrame

    def first(self, candidates: list[str]) -> str | None:
        existing = set(self.frame.columns)
        for col in candidates:
            if col in existing:
                return col
        return None


@dataclass(slots=True)
class DashboardAnalytics:
    """Prepare enriched KPIs and datasets for EDA dashboard tabs."""

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise DashboardAnalyticsError("Le dataset enrichi est vide, impossible de construire le dashboard.")

        df = frame.copy()
        resolver = ColumnResolver(df)

        self._ensure_numeric(df)
        self._attach_dimensions(df, resolver)
        self._attach_time_metrics(df, resolver)
        self._attach_efficiency_and_finance(df)
        self._attach_anomaly_scores(df)
        self._attach_anomaly_types(df)

        return df

    @staticmethod
    def _ensure_numeric(df: pd.DataFrame) -> None:
        for col in df.columns:
            if df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() >= max(5, int(0.3 * len(df))):
                    df[col] = converted

    @staticmethod
    def _normalize_label(series: pd.Series, fallback: str) -> pd.Series:
        normalized = series.astype(str).str.strip()
        normalized = normalized.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        return normalized.fillna(fallback)

    def _attach_dimensions(self, df: pd.DataFrame, resolver: ColumnResolver) -> None:
        type_col = resolver.first(["type_or", "ie_type_or", "bo_type_or", "nature", "type_intervention"])
        loc_col = resolver.first(["localisation", "ie_localisation", "bo_localisation", "succursale"])
        tech_col = resolver.first(["technicien", "salarie", "salarie_n", "pointage_salarie_n", "employee_id"])
        model_col = resolver.first(["modele_equipement", "modele", "type_materiel", "bo_modele_de_l_equipement"])
        segment_col = resolver.first(["segment", "service", "bo_service", "ie_service", "nom_client_or"])

        df["dim_type_or"] = (
            self._normalize_label(df[type_col], "N/A") if type_col else pd.Series("N/A", index=df.index)
        )
        df["dim_localisation"] = (
            self._normalize_label(df[loc_col], "N/A") if loc_col else pd.Series("N/A", index=df.index)
        )
        df["dim_technicien"] = (
            self._normalize_label(df[tech_col], "N/A") if tech_col else pd.Series("N/A", index=df.index)
        )
        df["dim_modele"] = (
            self._normalize_label(df[model_col], "N/A") if model_col else pd.Series("N/A", index=df.index)
        )
        df["dim_segment"] = (
            self._normalize_label(df[segment_col], "N/A") if segment_col else pd.Series("N/A", index=df.index)
        )

    def _attach_time_metrics(self, df: pd.DataFrame, resolver: ColumnResolver) -> None:
        real_col = resolver.first(["total_heures", "duree_pointage", "temps_reel", "hr_totale"])
        planned_col = resolver.first(["temps_prevu", "temps_vendu", "bo_qte_demandee", "bo_qte_facturee"])

        if real_col:
            df["temps_reel"] = pd.to_numeric(df[real_col], errors="coerce")
        else:
            df["temps_reel"] = np.nan

        if planned_col:
            df["temps_prevu"] = pd.to_numeric(df[planned_col], errors="coerce")
        else:
            df["temps_prevu"] = np.nan

        date_ref_col = resolver.first(["ie_date_reference", "date_creation", "date"])
        if date_ref_col:
            date_ref = pd.to_datetime(df[date_ref_col], errors="coerce")
        else:
            date_ref = pd.Series(pd.NaT, index=df.index)

        close_col = resolver.first(["bo_date_cloture_max", "bo_date_cloture_min", "date_dernier_pointage"])
        close_date = pd.to_datetime(df[close_col], errors="coerce") if close_col else pd.Series(pd.NaT, index=df.index)

        df["date_reference"] = date_ref
        df["date_cloture"] = close_date

        df["mois_reference"] = df["date_reference"].dt.to_period("M").astype(str)
        df.loc[df["date_reference"].isna(), "mois_reference"] = "N/A"

    @staticmethod
    def _attach_efficiency_and_finance(df: pd.DataFrame) -> None:
        valid_mask = (df["temps_prevu"] > 0) & (df["temps_reel"] > 0)
        df["efficience_devis"] = np.where(valid_mask, df["temps_prevu"] / df["temps_reel"], np.nan)
        df["surconsommation"] = np.where(valid_mask, np.maximum(df["temps_reel"] - df["temps_prevu"], 0), np.nan)

        if "bo_montant_total" in df.columns:
            ca = pd.to_numeric(df["bo_montant_total"], errors="coerce")
            df["marge_estimee"] = np.where(
                valid_mask,
                ca * (1 - np.where(df["temps_prevu"] > 0, df["surconsommation"] / df["temps_prevu"], 0)),
                np.nan,
            )
            df["marge_estimee"] = df["marge_estimee"].clip(lower=0)
            df["perte_potentielle"] = np.where(valid_mask, ca - df["marge_estimee"], np.nan)
        else:
            df["marge_estimee"] = np.nan
            df["perte_potentielle"] = np.nan

    @staticmethod
    def _attach_anomaly_scores(df: pd.DataFrame) -> None:
        ml_score = pd.to_numeric(df.get("ml_anomaly_score"), errors="coerce")
        if ml_score.notna().any() and ml_score.max(skipna=True) <= 1.0:
            ml_score = ml_score * 100

        df["anomaly_score_ml"] = ml_score
        df["anomaly_score_global"] = pd.to_numeric(df.get("final_anomaly_score"), errors="coerce") * 100
        df["anomaly_score_rule"] = pd.to_numeric(df.get("rule_anomaly_score"), errors="coerce") * 100
        df["is_anomaly"] = df.get("final_anomaly_flag", False).fillna(False).astype(bool)

    @staticmethod
    def _attach_anomaly_types(df: pd.DataFrame) -> None:
        rule_negative = df.get("rule_negative_values", False).fillna(False).astype(bool)
        rule_hours = df.get("rule_excessive_hours", False).fillna(False).astype(bool)
        rule_missing = df.get("rule_high_missing", False).fillna(False).astype(bool)

        technique = np.where(rule_negative | (df["efficience_devis"] < 0.7), 100, 0)
        process = np.where(rule_hours | rule_missing | (df["delai_premier_pointage"].fillna(0) > 3), 100, 0)
        financier = np.where(
            (df.get("or_cloture_sans_finance", False).fillna(False).astype(bool))
            | (df["surconsommation"].fillna(0) > 2),
            100,
            0,
        )
        ml = df["anomaly_score_ml"].fillna(0)

        df["anomaly_score_technique"] = technique.astype(float)
        df["anomaly_score_process"] = process.astype(float)
        df["anomaly_score_financier"] = financier.astype(float)
        df["anomaly_score_ml"] = ml.astype(float)

        labels = []
        for idx in df.index:
            local_labels: list[str] = []
            if df.at[idx, "anomaly_score_technique"] > 0:
                local_labels.append("technique")
            if df.at[idx, "anomaly_score_process"] > 0:
                local_labels.append("process")
            if df.at[idx, "anomaly_score_financier"] > 0:
                local_labels.append("financier")
            if df.at[idx, "anomaly_score_ml"] >= 60:
                local_labels.append("ML")
            labels.append(", ".join(local_labels) if local_labels else "aucune")

        df["anomaly_types"] = labels


def apply_filters(
    frame: pd.DataFrame,
    type_or: list[str] | None,
    localisation: list[str] | None,
    technicien: list[str] | None,
    modele: list[str] | None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    """Apply dynamic EDA filters."""
    df = frame.copy()

    if type_or:
        df = df[df["dim_type_or"].isin(type_or)]
    if localisation:
        df = df[df["dim_localisation"].isin(localisation)]
    if technicien:
        df = df[df["dim_technicien"].isin(technicien)]
    if modele:
        df = df[df["dim_modele"].isin(modele)]

    if date_range and "date_reference" in df.columns:
        start_date, end_date = date_range
        if pd.notna(start_date):
            df = df[df["date_reference"] >= pd.Timestamp(start_date)]
        if pd.notna(end_date):
            df = df[df["date_reference"] <= pd.Timestamp(end_date)]

    return df


def compute_global_kpis(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level dashboard KPIs with safe defaults."""
    total_or = int(len(frame))
    anomalies = int(frame["is_anomaly"].sum()) if "is_anomaly" in frame else 0
    anomaly_pct = (anomalies / total_or * 100.0) if total_or else 0.0

    return {
        "total_or": total_or,
        "anomalies": anomalies,
        "anomaly_pct": anomaly_pct,
        "score_moyen": float(frame["anomaly_score_global"].mean(skipna=True)) if total_or else 0.0,
        "efficience_moyenne": float(frame["efficience_devis"].mean(skipna=True)) if total_or else np.nan,
        "marge_moyenne": float(frame["marge_estimee"].mean(skipna=True)) if total_or else np.nan,
    }


def build_monthly_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if "date_reference" not in frame.columns:
        return pd.DataFrame()

    valid = frame[frame["date_reference"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["mois"] = valid["date_reference"].dt.to_period("M").astype(str)
    monthly = (
        valid.groupby("mois", as_index=False)
        .agg(
            nb_or=("or_id", "size"),
            anomalies=("is_anomaly", "sum"),
            score_moyen=("anomaly_score_global", "mean"),
        )
        .sort_values("mois")
    )
    monthly["pct_anomalies"] = np.where(monthly["nb_or"] > 0, monthly["anomalies"] / monthly["nb_or"] * 100, 0)
    return monthly
