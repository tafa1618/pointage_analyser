"""
Pipeline Productivité — analyse des heures facturables vs totales.

Formule officielle (alignée sur l'outil HTML existant) :
    Productivité = Σ Facturable / Σ Hr_Totale

Trois niveaux de granularité :
  - Global YTD
  - Par mois (rolling)
  - Par technicien × mois
  - Par équipe × mois

Aucune dépendance UI — ce module est consommé par dashboard/productivite.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Colonnes attendues dans le DataFrame Pointage harmonisé ───────────────
COL_DATE      = "date_saisie"           # après harmonisation DataPreprocessor
COL_TECHNOM   = "salarie_nom"
COL_EQUIPE    = "equipe_nom"
COL_FACTURABLE  = "facturable"
COL_HR_TOTALE   = "hr_totale"

# Colonnes brutes (fallback si le DataFrame n'est pas encore harmonisé)
COL_DATE_RAW    = "Saisie heures - Date"
COL_TECHNOM_RAW = "Salarié - Nom"
COL_EQUIPE_RAW  = "Salarié - Equipe(Nom)"
COL_FACTURABLE_RAW = "Facturable"
COL_HR_TOTALE_RAW  = "Hr_Totale"

# Seuils de performance (alignés sur l'outil HTML)
SEUIL_EXCELLENT = 0.60   # ≥ 60 %
SEUIL_BON       = 0.40   # ≥ 40 %
SEUIL_FAIBLE    = 0.20   # < 20 %


@dataclass
class ProductiviteResult:
    """Conteneur des résultats du pipeline productivité."""

    # KPIs globaux YTD
    ytd_productivite: float = 0.0
    ytd_facturable:   float = 0.0
    ytd_hr_totale:    float = 0.0

    # DataFrames résultats
    par_mois:        pd.DataFrame = field(default_factory=pd.DataFrame)
    par_technicien:  pd.DataFrame = field(default_factory=pd.DataFrame)
    par_equipe:      pd.DataFrame = field(default_factory=pd.DataFrame)
    par_tech_mois:   pd.DataFrame = field(default_factory=pd.DataFrame)
    par_equipe_mois: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Métadonnées
    nb_techniciens:  int = 0
    nb_equipes:      int = 0
    periode_debut:   str = ""
    periode_fin:     str = ""


def _label_perf(ratio: float) -> str:
    """Convertit un ratio en label de performance."""
    if ratio >= SEUIL_EXCELLENT:
        return "Excellent"
    elif ratio >= SEUIL_BON:
        return "Bon"
    elif ratio >= SEUIL_FAIBLE:
        return "Faible"
    else:
        return "Critique"


@dataclass
class ProductiviteBuilder:
    """
    Construit les agrégats de productivité à partir du DataFrame Pointage.

    Accepte indifféremment :
      - Un DataFrame déjà harmonisé (colonnes snake_case via DataPreprocessor)
      - Un DataFrame brut issu de l'Excel (colonnes françaises originales)
    """

    def build(self, pointage: pd.DataFrame) -> ProductiviteResult:
        """
        Point d'entrée principal.

        Args:
            pointage: DataFrame Pointage (brut ou harmonisé)

        Returns:
            ProductiviteResult avec tous les niveaux d'agrégation
        """
        if pointage is None or pointage.empty:
            logger.warning("Pointage vide — ProductiviteResult vide retourné")
            return ProductiviteResult()

        df = self._normalize_columns(pointage)

        if df.empty:
            return ProductiviteResult()

        logger.info(
            f"Productivité: {len(df)} lignes | "
            f"{df['_technom'].nunique()} techniciens | "
            f"{df['_equipe'].nunique()} équipes"
        )

        result = ProductiviteResult()

        # ── KPIs globaux YTD ──────────────────────────────────────────────
        result.ytd_facturable = df["_facturable"].sum()
        result.ytd_hr_totale  = df["_hr_totale"].sum()
        result.ytd_productivite = (
            result.ytd_facturable / result.ytd_hr_totale
            if result.ytd_hr_totale > 0 else 0.0
        )

        # ── Métadonnées période ───────────────────────────────────────────
        result.nb_techniciens = df["_technom"].nunique()
        result.nb_equipes     = df["_equipe"].nunique()
        result.periode_debut  = str(df["_date"].min().date())
        result.periode_fin    = str(df["_date"].max().date())

        # ── Agrégats ──────────────────────────────────────────────────────
        result.par_mois        = self._agg_par_mois(df)
        result.par_technicien  = self._agg_par_technicien(df)
        result.par_equipe      = self._agg_par_equipe(df)
        result.par_tech_mois   = self._agg_par_tech_mois(df)
        result.par_equipe_mois = self._agg_par_equipe_mois(df)

        logger.info(
            f"Productivité YTD: {result.ytd_productivite:.1%} | "
            f"{result.ytd_facturable:.0f}h fact / {result.ytd_hr_totale:.0f}h tot"
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Normalisation colonnes
    # ──────────────────────────────────────────────────────────────────────

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mappe les colonnes (harmonisées ou brutes) vers des noms internes
        préfixés _ pour éviter toute collision.
        """
        out = df.copy()

        def _pick(harmonized: str, raw: str) -> pd.Series:
            if harmonized in out.columns:
                return out[harmonized]
            elif raw in out.columns:
                return out[raw]
            else:
                logger.warning(f"Colonne manquante: {harmonized!r} / {raw!r}")
                return pd.Series(0.0, index=out.index)

        # Date
        date_col = _pick(COL_DATE, COL_DATE_RAW)
        out["_date"] = pd.to_datetime(date_col, errors="coerce")

        # Technicien
        out["_technom"] = _pick(COL_TECHNOM, COL_TECHNOM_RAW).fillna("Inconnu")

        # Équipe
        out["_equipe"] = _pick(COL_EQUIPE, COL_EQUIPE_RAW).fillna("Inconnu")

        # Heures
        out["_facturable"] = pd.to_numeric(
            _pick(COL_FACTURABLE, COL_FACTURABLE_RAW), errors="coerce"
        ).fillna(0.0)

        out["_hr_totale"] = pd.to_numeric(
            _pick(COL_HR_TOTALE, COL_HR_TOTALE_RAW), errors="coerce"
        ).fillna(0.0)

        # Mois (Period pour tri correct)
        out["_mois"] = out["_date"].dt.to_period("M")
        out["_mois_str"] = out["_mois"].astype(str)  # ex: "2026-01"

        # Filtrer les lignes sans date valide
        before = len(out)
        out = out[out["_date"].notna()].copy()
        if before != len(out):
            logger.warning(f"{before - len(out)} lignes sans date valide ignorées")

        return out

    # ──────────────────────────────────────────────────────────────────────
    # Agrégats
    # ──────────────────────────────────────────────────────────────────────

    def _prod_agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        """Agrégation standard: facturable, hr_totale, productivite, label."""
        agg = group.agg(
            facturable=("_facturable", "sum"),
            hr_totale=("_hr_totale", "sum"),
        ).reset_index()
        agg["productivite"] = (
            agg["facturable"] / agg["hr_totale"].replace(0, float("nan"))
        ).fillna(0.0)
        agg["perf_label"] = agg["productivite"].apply(_label_perf)
        return agg

    def _agg_par_mois(self, df: pd.DataFrame) -> pd.DataFrame:
        """Productivité globale par mois (toutes équipes confondues)."""
        agg = self._prod_agg(df.groupby("_mois_str"))
        agg = agg.rename(columns={"_mois_str": "mois"})
        agg = agg.sort_values("mois").reset_index(drop=True)
        return agg

    def _agg_par_technicien(self, df: pd.DataFrame) -> pd.DataFrame:
        """Productivité YTD par technicien."""
        agg = self._prod_agg(df.groupby(["_technom", "_equipe"]))
        agg = agg.rename(columns={"_technom": "technicien", "_equipe": "equipe"})

        # Nb jours travaillés
        jours = (
            df.groupby("_technom")["_date"]
            .nunique()
            .reset_index()
            .rename(columns={"_technom": "technicien", "_date": "nb_jours"})
        )
        agg = agg.merge(jours, on="technicien", how="left")
        agg = agg.sort_values("productivite", ascending=False).reset_index(drop=True)
        return agg

    def _agg_par_equipe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Productivité YTD par équipe."""
        agg = self._prod_agg(df.groupby("_equipe"))
        agg = agg.rename(columns={"_equipe": "equipe"})

        nb_tech = (
            df.groupby("_equipe")["_technom"]
            .nunique()
            .reset_index()
            .rename(columns={"_equipe": "equipe", "_technom": "nb_techniciens"})
        )
        agg = agg.merge(nb_tech, on="equipe", how="left")
        agg = agg.sort_values("productivite", ascending=False).reset_index(drop=True)
        return agg

    def _agg_par_tech_mois(self, df: pd.DataFrame) -> pd.DataFrame:
        """Productivité par technicien × mois (matrice)."""
        agg = self._prod_agg(df.groupby(["_technom", "_equipe", "_mois_str"]))
        agg = agg.rename(columns={
            "_technom": "technicien",
            "_equipe": "equipe",
            "_mois_str": "mois",
        })
        agg = agg.sort_values(["technicien", "mois"]).reset_index(drop=True)
        return agg

    def _agg_par_equipe_mois(self, df: pd.DataFrame) -> pd.DataFrame:
        """Productivité par équipe × mois."""
        agg = self._prod_agg(df.groupby(["_equipe", "_mois_str"]))
        agg = agg.rename(columns={"_equipe": "equipe", "_mois_str": "mois"})
        agg = agg.sort_values(["equipe", "mois"]).reset_index(drop=True)
        return agg
