from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.preprocess import DataPreprocessor


class DatasetBuilderError(Exception):
    """Raised when OR-level unified dataset construction fails."""


@dataclass(slots=True)
class ORDatasetBuilder:
    """Build a single-row-per-OR dataset from IE, Pointage and BO."""

    preprocessor: DataPreprocessor

    def build(self, ie_df: pd.DataFrame, pointage_df: pd.DataFrame, bo_df: pd.DataFrame) -> pd.DataFrame:
        ie_or = self._prepare_ie_base(ie_df)
        pointage_agg = self._aggregate_pointage(pointage_df, ie_or)
        bo_agg = self._aggregate_bo(bo_df)

        final_df = ie_or.merge(pointage_agg, on="or_id", how="left", validate="1:1")
        final_df = final_df.merge(bo_agg, on="or_id", how="left", validate="1:1")

        final_df["has_pointage"] = final_df["nb_lignes_pointage"].fillna(0) > 0
        final_df["has_bo"] = final_df["bo_nb_lignes"].fillna(0) > 0
        final_df["or_cloture_sans_finance"] = (
            final_df["has_bo"] & final_df["bo_montant_total"].fillna(0).eq(0)
        )
        final_df["or_sans_pointage"] = ~final_df["has_pointage"]
        final_df["or_absent_bo"] = ~final_df["has_bo"]

        if final_df["or_id"].duplicated().any():
            raise DatasetBuilderError("Le dataset final n'est pas au niveau OR (doublons or_id).")

        return final_df

    def _prepare_ie_base(self, ie_df: pd.DataFrame) -> pd.DataFrame:
        if "or_id" not in ie_df.columns:
            raise DatasetBuilderError("IE doit contenir la clé harmonisée `or_id`.")

        ie = ie_df.copy()
        date_col = self._find_first(ie.columns, ["date_creation", "date", "date_ouverture"])
        if date_col:
            ie["ie_date_reference"] = self.preprocessor.to_datetime_safe(ie[date_col])
        else:
            ie["ie_date_reference"] = pd.NaT

        ie = ie.sort_values(["or_id", "ie_date_reference"], kind="stable")
        ie_one_row = ie.groupby("or_id", as_index=False).first()
        return ie_one_row

    def _aggregate_pointage(self, pointage_df: pd.DataFrame, ie_or: pd.DataFrame) -> pd.DataFrame:
        if "or_id" not in pointage_df.columns:
            raise DatasetBuilderError("Pointage doit contenir la clé harmonisée `or_id`.")

        pt = pointage_df.copy()

        date_col = self._find_first(
            pt.columns,
            ["date", "date_saisie_heure_salarie", "date_saisie", "jour", "date_pointage"],
        )
        hour_col = self._find_first(
            pt.columns,
            ["heure_realis_or", "hr_travaille", "hr_totale", "duree_pointage", "heures", "quantite"],
        )

        if date_col is None:
            raise DatasetBuilderError("Colonne de date introuvable dans le fichier Pointage.")
        if hour_col is None:
            raise DatasetBuilderError("Colonne d'heures introuvable dans le fichier Pointage.")

        pt["pointage_date"] = self.preprocessor.to_datetime_safe(pt[date_col])
        pt["pointage_heures"] = self.preprocessor.to_numeric_safe(pt[hour_col])
        pt["pointage_heures"] = pt["pointage_heures"].fillna(0.0)

        # granularité journalière OR
        daily = (
            pt.groupby(["or_id", "pointage_date"], as_index=False)
            .agg(heures_jour=("pointage_heures", "sum"), nb_lignes_jour=("or_id", "size"))
        )

        or_agg = (
            daily.groupby("or_id", as_index=False)
            .agg(
                total_heures=("heures_jour", "sum"),
                nb_lignes_pointage=("nb_lignes_jour", "sum"),
                date_premier_pointage=("pointage_date", "min"),
                date_dernier_pointage=("pointage_date", "max"),
                variance_journaliere=("heures_jour", "var"),
                moyenne_journaliere=("heures_jour", "mean"),
            )
        )

        or_agg["variance_journaliere"] = or_agg["variance_journaliere"].fillna(0.0)

        ref = ie_or[["or_id", "ie_date_reference"]]
        or_agg = or_agg.merge(ref, on="or_id", how="left")
        or_agg["delai_premier_pointage"] = (
            or_agg["date_premier_pointage"] - or_agg["ie_date_reference"]
        ).dt.days
        or_agg["delai_dernier_pointage"] = (
            or_agg["date_dernier_pointage"] - or_agg["ie_date_reference"]
        ).dt.days
        or_agg = or_agg.drop(columns=["ie_date_reference"])
        return or_agg

    def _aggregate_bo(self, bo_df: pd.DataFrame) -> pd.DataFrame:
        if "or_id" not in bo_df.columns:
            raise DatasetBuilderError("BO doit contenir la clé harmonisée `or_id`.")

        bo = bo_df.copy()
        money_col = self._find_first(
            bo.columns,
            ["montant_total_or", "montant_total", "montant_piece", "montant_mo", "montant"],
        )
        close_col = self._find_first(bo.columns, ["date_cloture", "date_facture", "date"])

        if money_col:
            bo["bo_montant_line"] = self.preprocessor.to_numeric_safe(bo[money_col]).fillna(0.0)
        else:
            bo["bo_montant_line"] = 0.0

        if close_col:
            bo["bo_date_cloture"] = self.preprocessor.to_datetime_safe(bo[close_col])
        else:
            bo["bo_date_cloture"] = pd.NaT

        bo_agg = (
            bo.groupby("or_id", as_index=False)
            .agg(
                bo_nb_lignes=("or_id", "size"),
                bo_montant_total=("bo_montant_line", "sum"),
                bo_date_cloture_min=("bo_date_cloture", "min"),
                bo_date_cloture_max=("bo_date_cloture", "max"),
            )
        )
        return bo_agg

    @staticmethod
    def _find_first(columns: pd.Index, candidates: list[str]) -> str | None:
        colset = set(columns)
        for c in candidates:
            if c in colset:
                return c
        return None
