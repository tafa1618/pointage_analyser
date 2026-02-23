"""Pipeline dataset_builder — construit le dataset OR-level (1 ligne = 1 OR)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pointage_analyzer.core.config import (
    BO_COLUMN_MAP,
    IE_COLUMN_MAP,
    POINTAGE_COLUMN_MAP,
    ScoringConfig,
)
from pointage_analyzer.ingestion.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class DatasetBuilderError(Exception):
    """Raised when OR-level dataset construction fails."""


@dataclass(slots=True)
class ORDatasetBuilder:
    """
    Construit un dataset OR-level depuis IE, Pointage et BO.

    Logique de jointure :
        - Base principale = Pointage (agrégé par OR, sans les OR=0)
        - IE enrichit en LEFT JOIN (dimensions métier)
        - BO enrichit en LEFT JOIN (données financières)
        - 1 ligne = 1 OR, garanti par assertion finale

    Attribution technicien :
        - Technicien principal = salarié avec le plus de Hr_Totale sur l'OR
        - Équipe principale    = équipe du technicien principal
        - Les autres techniciens sont stockés dans `techniciens_secondaires`
    """

    preprocessor: DataPreprocessor
    config: ScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ScoringConfig()

    def build(
        self,
        ie_df: pd.DataFrame,
        pointage_df: pd.DataFrame,
        bo_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construit le dataset final OR-level.

        Args:
            ie_df: DataFrame IE (colonnes harmonisées)
            pointage_df: DataFrame Pointage complet (colonnes harmonisées)
            bo_df: DataFrame BO (colonnes harmonisées)

        Returns:
            DataFrame avec 1 ligne par OR, toutes les dimensions enrichies
        """
        logger.info("=== Construction dataset OR-level ===")

        # 1. Agréger Pointage (base principale)
        pt_agg = self._aggregate_pointage(pointage_df)
        logger.info(f"Pointage agrégé: {len(pt_agg)} OR uniques")

        # 2. Préparer IE (dimensions métier)
        ie_base = self._prepare_ie_base(ie_df) if ie_df is not None else pd.DataFrame()
        logger.info(f"IE préparé: {len(ie_base)} OR")

        # 3. Préparer BO (données financières)
        bo_base = self._aggregate_bo(bo_df) if bo_df is not None else pd.DataFrame()
        logger.info(f"BO agrégé: {len(bo_base)} OR")

        # 4. Jointures LEFT depuis Pointage
        final = pt_agg.copy()

        if not ie_base.empty and "or_id" in ie_base.columns:
            before = len(final)
            final = final.merge(
                ie_base, on="or_id", how="left", suffixes=("", "_ie")
            )
            logger.info(
                f"Après LEFT JOIN IE: {final['or_id'].nunique()} OR "
                f"({len(final[final['type_or'].notna()])} enrichis)"
                if "type_or" in final.columns else f"JOIN IE: {len(final)} lignes"
            )

        if not bo_base.empty and "or_id" in bo_base.columns:
            final = final.merge(
                bo_base, on="or_id", how="left", suffixes=("", "_bo")
            )
            bo_col = "montant_total" if "montant_total" in final.columns else None
            enriched_bo = final[bo_col].notna().sum() if bo_col else 0
            logger.info(f"Après LEFT JOIN BO: {enriched_bo} OR enrichis financièrement")

        # 5. Flags de complétude
        final["has_pointage"] = True  # Toujours True (base = Pointage)
        final["has_bo"] = (
            final["montant_total"].notna() if "montant_total" in final.columns
            else pd.Series(False, index=final.index)
        )
        final["has_ie"] = (
            final["type_or"].notna() if "type_or" in final.columns
            else pd.Series(False, index=final.index)
        )
        final["or_cloture_sans_finance"] = (
            (final.get("position", pd.Series("", index=final.index)) == "CP")
            & ~final["has_bo"]
        )

        # 6. Assertion finale : 1 ligne = 1 OR
        if final["or_id"].duplicated().any():
            dupes = final["or_id"][final["or_id"].duplicated()].tolist()
            raise DatasetBuilderError(
                f"Dataset final non-unique: {len(dupes)} doublons or_id. "
                f"Exemples: {dupes[:5]}"
            )

        logger.info(
            f"=== Dataset final: {len(final)} OR | "
            f"avec IE: {final.get('has_ie', pd.Series(False)).sum()} | "
            f"avec BO: {final['has_bo'].sum()} ==="
        )
        return final

    def _aggregate_pointage(self, pointage_df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège le Pointage à 1 ligne par OR.

        Étapes :
        1. Filtre OR valides (or_id != None, correspond aux vraies interventions)
        2. Agrège métriques d'heures et dates
        3. Identifie le technicien principal (max Hr_Totale)
        4. Identifie l'équipe principale (équipe du tech principal)
        5. Calcule délais depuis IE date (si disponible)
        """
        pt = pointage_df.copy()

        # Filtre lignes avec OR valide (OR=0 / sans OR = hors-OR)
        pt_or = pt[pt["or_id"].notna()].copy()
        logger.info(
            f"Pointage: {len(pt)} lignes total, "
            f"{len(pt_or)} avec OR valide ({len(pt) - len(pt_or)} hors-OR exclus)"
        )

        if pt_or.empty:
            raise DatasetBuilderError(
                "Aucune ligne Pointage avec OR valide après filtrage."
            )

        # Colonnes heures
        hr_col = "hr_totale" if "hr_totale" in pt_or.columns else "heure_realisee"
        date_col = "date_saisie" if "date_saisie" in pt_or.columns else None

        if hr_col not in pt_or.columns:
            raise DatasetBuilderError(
                f"Colonne d'heures introuvable (cherché: hr_totale, heure_realisee). "
                f"Colonnes disponibles: {list(pt_or.columns)}"
            )

        pt_or[hr_col] = pd.to_numeric(pt_or[hr_col], errors="coerce").fillna(0.0)

        # --- Granularité journalière OR × date ---
        if date_col and date_col in pt_or.columns:
            pt_or["_date_parsed"] = pd.to_datetime(
                pt_or[date_col], errors="coerce", dayfirst=True
            )
            daily = (
                pt_or.groupby(["or_id", "_date_parsed"], as_index=False)
                .agg(
                    heures_jour=(hr_col, "sum"),
                    nb_lignes_jour=("or_id", "size"),
                )
            )
            or_dates = (
                daily.groupby("or_id", as_index=False)
                .agg(
                    date_premier_pointage=("_date_parsed", "min"),
                    date_dernier_pointage=("_date_parsed", "max"),
                    nb_jours_travailles=("_date_parsed", "nunique"),
                    variance_journaliere=("heures_jour", "var"),
                    heures_jour_max=("heures_jour", "max"),
                    heures_jour_min=("heures_jour", "min"),
                )
            )
            or_dates["variance_journaliere"] = or_dates["variance_journaliere"].fillna(0.0)
        else:
            or_dates = pd.DataFrame({"or_id": pt_or["or_id"].unique()})

        # --- Agrégation totale par OR ---
        or_agg = (
            pt_or.groupby("or_id", as_index=False)
            .agg(
                total_heures=(hr_col, "sum"),
                nb_lignes_pointage=("or_id", "size"),
            )
        )

        # --- Attribution technicien principal ---
        tech_agg = self._identify_technicien_principal(pt_or, hr_col)

        # --- Merge toutes les agrégations ---
        result = or_agg.merge(tech_agg, on="or_id", how="left")
        if not or_dates.empty and "or_id" in or_dates.columns:
            result = result.merge(or_dates, on="or_id", how="left")

        return result

    @staticmethod
    def _identify_technicien_principal(
        pt_or: pd.DataFrame, hr_col: str
    ) -> pd.DataFrame:
        """
        Identifie, pour chaque OR, le technicien principal (max heures).

        Returns:
            DataFrame avec colonnes:
                - or_id
                - technicien_principal_nom
                - technicien_principal_num
                - equipe_principale
                - nb_techniciens
                - techniciens_liste (tous, séparés par |)
        """
        has_nom = "salarie_nom" in pt_or.columns
        has_num = "salarie_num" in pt_or.columns
        has_equipe = "equipe_nom" in pt_or.columns

        if not has_nom and not has_num:
            # Pas d'info technicien dans ce fichier
            return pd.DataFrame({"or_id": pt_or["or_id"].unique()})

        group_cols = ["or_id"]
        if has_nom:
            group_cols.append("salarie_nom")
        if has_num:
            group_cols.append("salarie_num")
        if has_equipe:
            group_cols.append("equipe_nom")

        # Heures par (OR, technicien)
        tech_hours = (
            pt_or.groupby(group_cols, as_index=False, dropna=False)
            [hr_col]
            .sum()
            .rename(columns={hr_col: "_tech_heures"})
        )

        # Technicien principal = max heures par OR
        idx_max = tech_hours.groupby("or_id")["_tech_heures"].idxmax()
        principal = tech_hours.loc[idx_max].copy()

        rename_cols = {"or_id": "or_id"}
        if has_nom:
            rename_cols["salarie_nom"] = "technicien_principal_nom"
        if has_num:
            rename_cols["salarie_num"] = "technicien_principal_num"
        if has_equipe:
            rename_cols["equipe_nom"] = "equipe_principale"

        principal = principal.rename(columns=rename_cols).drop(
            columns=["_tech_heures"], errors="ignore"
        )

        # Nb techniciens par OR
        nb_tech = (
            pt_or.groupby("or_id")["salarie_nom" if has_nom else "salarie_num"]
            .nunique()
            .reset_index()
            .rename(
                columns={
                    "salarie_nom": "nb_techniciens",
                    "salarie_num": "nb_techniciens",
                }
            )
        )

        # Liste complète des techniciens (pour traçabilité)
        if has_nom:
            tech_liste = (
                pt_or.groupby("or_id")["salarie_nom"]
                .apply(lambda x: " | ".join(sorted(set(x.dropna().astype(str)))))
                .reset_index()
                .rename(columns={"salarie_nom": "techniciens_liste"})
            )
            principal = principal.merge(tech_liste, on="or_id", how="left")

        result = principal.merge(nb_tech, on="or_id", how="left")
        return result

    def _prepare_ie_base(self, ie_df: pd.DataFrame) -> pd.DataFrame:
        """Déduplique IE à 1 ligne par OR (premier par date si doublons)."""
        if "or_id" not in ie_df.columns:
            logger.warning("IE sans colonne or_id — enrichissement désactivé.")
            return pd.DataFrame()

        ie = ie_df.copy()

        # Déduplique par OR (garde la ligne la plus récente si doublons)
        date_col = next(
            (c for c in ["date_extraction", "date_creation", "date"] if c in ie.columns),
            None,
        )
        if date_col:
            ie["_sort_date"] = pd.to_datetime(ie[date_col], errors="coerce")
            ie = ie.sort_values(["or_id", "_sort_date"], ascending=[True, False])
            ie = ie.drop(columns=["_sort_date"])

        ie_dedup = ie.groupby("or_id", as_index=False).first()

        # Normalise Position
        if "position" in ie_dedup.columns:
            ie_dedup["position"] = ie_dedup["position"].astype(str).str.strip().str.upper()

        logger.info(
            f"IE dédupliqué: {len(ie_dedup)} OR uniques "
            f"(depuis {len(ie_df)} lignes brutes)"
        )
        return ie_dedup

    def _aggregate_bo(self, bo_df: pd.DataFrame) -> pd.DataFrame:
        """Agrège BO à 1 ligne par OR (somme des montants, min/max dates)."""
        if "or_id" not in bo_df.columns:
            logger.warning("BO sans colonne or_id — enrichissement désactivé.")
            return pd.DataFrame()

        bo = bo_df.copy()

        agg_dict: dict[str, Any] = {"or_id": ("or_id", "size")}

        numeric_bo_cols = [
            "montant_total", "montant_mo", "montant_pieces", "montant_frais",
            "temps_prevu_devis", "temps_vendu",
            "qte_demandee", "qte_facturee",
        ]
        for col in numeric_bo_cols:
            if col in bo.columns:
                bo[col] = pd.to_numeric(bo[col], errors="coerce").fillna(0.0)
                agg_dict[col] = (col, "sum")

        date_bo_cols = ["date_cloture", "date_facture", "date_creation"]
        for col in date_bo_cols:
            if col in bo.columns:
                bo[col] = pd.to_datetime(bo[col], errors="coerce")
                agg_dict[f"{col}_min"] = (col, "min")
                agg_dict[f"{col}_max"] = (col, "max")

        # Colonnes texte (prendre la première valeur non-nulle)
        for col in ["modele_equipement", "type_materiel", "localisation", "service"]:
            if col in bo.columns:
                agg_dict[col] = (col, "first")

        bo_agg = bo.groupby("or_id").agg(**{
            k: v for k, v in agg_dict.items() if k != "or_id"
        }).reset_index()

        return bo_agg


# Type hint helper
from typing import Any  # noqa: E402 (kept at bottom to avoid circular issues)
