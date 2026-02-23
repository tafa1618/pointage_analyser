"""Feature engineering OR-level — features métier pour le moteur ML."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""


class FeatureEngineer:
    """
    Construit les features numériques pour le modèle Isolation Forest.

    Toutes les features sont calculées au niveau OR (1 ligne = 1 OR).
    Aucune feature employee-level : le dataset est agrégé.
    """

    def build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Point d'entrée principal : construit toutes les features."""
        if frame.empty:
            raise FeatureEngineeringError("Dataset vide — impossible de construire les features.")

        df = frame.copy()
        df = self._add_efficiency_features(df)
        df = self._add_duration_features(df)
        df = self._add_process_features(df)
        df = self._add_financial_features(df)
        df = self._clean_infinite(df)
        return df

    # ------------------------------------------------------------------
    # Features d'efficience
    # ------------------------------------------------------------------
    def _add_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratios entre temps réel et temps prévus/vendus."""
        has_total = "total_heures" in df.columns
        has_prevu = "temps_prevu_devis" in df.columns
        has_vendu = "temps_vendu" in df.columns

        if has_total and has_prevu:
            df["ratio_reel_prevu"] = df["total_heures"] / df["temps_prevu_devis"].replace(0, np.nan)
            df["delta_heures_prevu"] = df["total_heures"] - df["temps_prevu_devis"]

        if has_total and has_vendu:
            df["ratio_reel_vendu"] = df["total_heures"] / df["temps_vendu"].replace(0, np.nan)

        # Efficience financière MO
        if "montant_mo" in df.columns and has_total:
            df["montant_mo_par_heure"] = (
                df["montant_mo"] / df["total_heures"].replace(0, np.nan)
            )

        return df

    # ------------------------------------------------------------------
    # Features de durée / délais
    # ------------------------------------------------------------------
    def _add_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Durée de l'OR, délais de premier et dernier pointage."""
        has_premier = "date_premier_pointage" in df.columns
        has_dernier = "date_dernier_pointage" in df.columns

        if has_premier and has_dernier:
            df["date_premier_pointage"] = pd.to_datetime(
                df["date_premier_pointage"], errors="coerce"
            )
            df["date_dernier_pointage"] = pd.to_datetime(
                df["date_dernier_pointage"], errors="coerce"
            )
            df["duree_intervention_jours"] = (
                df["date_dernier_pointage"] - df["date_premier_pointage"]
            ).dt.days.clip(lower=0)

        if "total_heures" in df.columns:
            df["log_total_heures"] = np.log1p(df["total_heures"])

        if "nb_jours_travailles" in df.columns and "total_heures" in df.columns:
            df["heures_par_jour_moyen"] = (
                df["total_heures"] / df["nb_jours_travailles"].replace(0, np.nan)
            )

        return df

    # ------------------------------------------------------------------
    # Features process / qualité
    # ------------------------------------------------------------------
    def _add_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Caractéristiques du processus de pointage."""
        if "nb_lignes_pointage" in df.columns and "nb_jours_travailles" in df.columns:
            df["lignes_par_jour"] = (
                df["nb_lignes_pointage"] / df["nb_jours_travailles"].replace(0, np.nan)
            )

        if "nb_techniciens" in df.columns:
            df["or_multi_tech"] = (df["nb_techniciens"] > 1).astype(int)

        if "variance_journaliere" in df.columns:
            df["log_variance_jours"] = np.log1p(df["variance_journaliere"].fillna(0))

        return df

    # ------------------------------------------------------------------
    # Features financières
    # ------------------------------------------------------------------
    def _add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in ["qte_facturee", "qte_demandee"]):
            df["ratio_qte_facturee"] = (
                df["qte_facturee"] / df["qte_demandee"].replace(0, np.nan)
            )

        if "montant_total" in df.columns:
            df["log_montant_total"] = np.log1p(df["montant_total"].fillna(0))

        return df

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_infinite(df: pd.DataFrame) -> pd.DataFrame:
        """Remplace les infinis et NaN générés par les divisions."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df
