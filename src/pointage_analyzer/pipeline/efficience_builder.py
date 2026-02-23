"""
pipeline/efficience_builder.py

Module d'analyse d'efficience OR basé sur BO.

Responsabilités :
  - Agréger BO par OR pour extraire temps_vendu, temps_prevu_devis, duree_pointage_tot
  - Calculer le ratio d'efficience (pointé / référence)
  - Enrichir avec les données Pointage (équipe, technicien, total_heures)
  - Fournir des agrégats par équipe et technicien pour le coaching
  - Supporter un filtre temporel (année / mois)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pointage_analyzer.core.config import ScoringConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Catégories d'efficience
CAT_SOUS_PRODUCTIF    = "🔴 Sous-productif"
CAT_NORMAL            = "🟢 Normal"
CAT_DEPASSE           = "🟡 Dépassement"
CAT_NON_EVALUABLE     = "⚪ Non évaluable"


@dataclass
class EfficienceResult:
    """Résultat du module efficience."""
    df_or: pd.DataFrame          # OR enrichis avec ratios d'efficience
    df_equipe: pd.DataFrame      # Agrégat par équipe
    df_technicien: pd.DataFrame  # Agrégat par technicien
    nb_or_evaluables: int
    nb_or_non_evaluables: int
    ratio_moyen_global: float | None


class EfficienceBuilder:
    """
    Construit les métriques d'efficience par OR, équipe, et technicien.

    Pipeline :
      1. Nettoyage et agrégation BO par or_id
      2. Calcul temps_reference (vendu → devis → NaN)
      3. Calcul ratio d'efficience
      4. Enrichissement avec df_or (équipe, technicien, date)
      5. Agrégation par équipe / technicien
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def build(
        self,
        bo: pd.DataFrame,
        df_or: pd.DataFrame,
    ) -> EfficienceResult:
        """
        Construit le résultat d'efficience complet.

        Args:
            bo: DataFrame BO harmonisé (colonnes normalisées)
            df_or: Dataset OR-level (issu du pipeline Pointage)
        """
        bo_agg = self._aggregate_bo(bo)
        df = self._compute_efficience(bo_agg)
        df = self._enrich_with_pointage(df, df_or)
        df = self._flag_ecart_sources(df)

        nb_eval = int((df["efficience_categorie"] != CAT_NON_EVALUABLE).sum())
        nb_non  = int((df["efficience_categorie"] == CAT_NON_EVALUABLE).sum())

        eval_rows = df[df["efficience_ratio"].notna()]
        ratio_moy = float(eval_rows["efficience_ratio"].mean()) if not eval_rows.empty else None

        df_equipe      = self._aggregate_by_equipe(df)
        df_technicien  = self._aggregate_by_technicien(df)

        logger.info(
            f"Efficience: {nb_eval} OR évaluables, {nb_non} non évaluables, "
            f"ratio moyen={ratio_moy:.2f}" if ratio_moy else f"Efficience: {nb_eval} OR évaluables"
        )

        return EfficienceResult(
            df_or=df,
            df_equipe=df_equipe,
            df_technicien=df_technicien,
            nb_or_evaluables=nb_eval,
            nb_or_non_evaluables=nb_non,
            ratio_moyen_global=ratio_moy,
        )

    def filter_by_period(
        self,
        df: pd.DataFrame,
        annee: int | None = None,
        mois: int | None = None,
    ) -> pd.DataFrame:
        """
        Filtre temporel sur la colonne date_reference (date_creation BO ou
        date_premier_pointage du Pointage).

        Args:
            df:    df_or enrichi (sortie de build())
            annee: Année à filtrer (None = toutes)
            mois:  Mois à filtrer, 1-12 (None = tous)
        """
        if df.empty:
            return df

        date_col = None
        for col in ("date_creation", "date_premier_pointage"):
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            logger.warning("Aucune colonne de date trouvée pour le filtre temporel")
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if annee is not None:
            df = df[df[date_col].dt.year == annee]
        if mois is not None:
            df = df[df[date_col].dt.month == mois]

        return df

    def get_annees_disponibles(self, df: pd.DataFrame) -> list[int]:
        """Retourne les années disponibles dans le dataset."""
        for col in ("date_creation", "date_premier_pointage"):
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                return sorted(dates.dt.year.unique().tolist())
        return []

    def get_mois_disponibles(self, df: pd.DataFrame, annee: int | None = None) -> list[int]:
        """Retourne les mois disponibles (optionnellement filtrés par année)."""
        for col in ("date_creation", "date_premier_pointage"):
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if annee is not None:
                    dates = dates[dates.dt.year == annee]
                return sorted(dates.dt.month.unique().tolist())
        return []

    # ------------------------------------------------------------------
    # Étapes internes
    # ------------------------------------------------------------------

    def _aggregate_bo(self, bo: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège BO par or_id — un OR peut avoir plusieurs lignes (segments).
        On prend les valeurs max (elles sont répétées par ligne dans BO).
        """
        if bo.empty or "or_id" not in bo.columns:
            logger.warning("BO vide ou sans colonne or_id")
            return pd.DataFrame()

        numeric_cols = [
            "temps_vendu", "temps_prevu_devis",
            "duree_pointage_prod", "duree_pointage_tot",
            "montant_mo", "montant_total",
        ]
        date_cols = ["date_creation", "date_cloture", "date_facture"]
        text_cols = ["nom_client", "modele_equipement", "type_materiel", "localisation"]

        agg_dict: dict = {}

        for col in numeric_cols:
            if col in bo.columns:
                agg_dict[col] = "max"

        for col in date_cols:
            if col in bo.columns:
                bo[col] = pd.to_datetime(bo[col], errors="coerce")
                agg_dict[col] = "min"  # date la plus ancienne (création)

        for col in text_cols:
            if col in bo.columns:
                agg_dict[col] = "first"

        if not agg_dict:
            logger.warning("Aucune colonne connue trouvée dans BO pour l'agrégation")
            return pd.DataFrame()

        bo_agg = (
            bo[["or_id"] + list(agg_dict.keys())]
            .groupby("or_id", as_index=False)
            .agg(agg_dict)
        )

        # Convertit en numérique
        for col in numeric_cols:
            if col in bo_agg.columns:
                bo_agg[col] = pd.to_numeric(bo_agg[col], errors="coerce")

        logger.info(f"BO agrégé: {len(bo_agg)} OR distincts")
        return bo_agg

    def _compute_efficience(self, bo_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le ratio d'efficience et catégorise chaque OR.

        Logique :
          1. temps_reference = temps_vendu si > 0
          2.                 = temps_prevu_devis si temps_vendu manquant/nul
          3.                 = NaN si les deux manquent → non évaluable

          ratio = duree_pointage_tot / temps_reference
        """
        if bo_agg.empty:
            return bo_agg

        df = bo_agg.copy()

        # Référence temps : vendu → devis fallback
        vendu = df.get("temps_vendu", pd.Series(np.nan, index=df.index))
        devis = df.get("temps_prevu_devis", pd.Series(np.nan, index=df.index))
        ptg   = df.get("duree_pointage_tot", pd.Series(np.nan, index=df.index))

        # Nettoyage : 0 = absent (pas renseigné)
        vendu = vendu.where(vendu > 0, np.nan)
        devis = devis.where(devis > 0, np.nan)

        df["temps_reference"]      = vendu.fillna(devis)
        df["source_reference"]     = np.where(
            vendu.notna(),  "Temps vendu",
            np.where(devis.notna(), "Temps prévu devis", "Aucune")
        )

        # Ratio
        df["efficience_ratio"] = np.where(
            df["temps_reference"].notna() & (df["temps_reference"] > 0),
            ptg / df["temps_reference"],
            np.nan,
        )

        # Catégorisation
        low  = self.config.efficience_low
        high = self.config.efficience_high

        conditions = [
            df["efficience_ratio"].isna(),
            df["efficience_ratio"] < low,
            df["efficience_ratio"] > high,
        ]
        choices = [CAT_NON_EVALUABLE, CAT_SOUS_PRODUCTIF, CAT_DEPASSE]
        df["efficience_categorie"] = np.select(conditions, choices, default=CAT_NORMAL)

        # Label lisible manager
        df["efficience_label"] = df.apply(self._make_label, axis=1)

        return df

    def _make_label(self, row: pd.Series) -> str:
        cat   = row.get("efficience_categorie", CAT_NON_EVALUABLE)
        ratio = row.get("efficience_ratio")
        ref   = row.get("temps_reference")
        src   = row.get("source_reference", "")

        if cat == CAT_NON_EVALUABLE:
            return (
                "⚪ NON ÉVALUABLE — Cet OR ne dispose ni de temps vendu ni de temps prévu devis. "
                "L'efficience ne peut pas être calculée. Vérifier que l'OR est bien configuré dans l'ERP."
            )
        if cat == CAT_SOUS_PRODUCTIF:
            pct = ratio * 100 if ratio is not None else 0
            return (
                f"🔴 SOUS-PRODUCTIF — Seulement {pct:.0f}% des heures de référence ont été consommées "
                f"(référence : {ref:.1f}h issue de « {src} »). "
                f"Des heures facturées ou prévues n'ont pas été pointées. "
                f"Vérifier si le pointage est complet ou si l'OR a été clôturé prématurément."
            )
        if cat == CAT_DEPASSE:
            pct = ratio * 100 if ratio is not None else 0
            surplus = (ratio - 1.0) * ref if ratio is not None and ref is not None else 0
            return (
                f"🟡 DÉPASSEMENT BUDGET — {pct:.0f}% des heures de référence consommées, "
                f"soit +{surplus:.1f}h de plus que prévu "
                f"(référence : {ref:.1f}h issue de « {src} »). "
                f"Le coût réel de cet OR dépasse ce qui avait été budgété."
            )
        # NORMAL
        pct = ratio * 100 if ratio is not None else 0
        return f"🟢 NORMAL — {pct:.0f}% des heures de référence consommées."

    def _enrich_with_pointage(
        self, df: pd.DataFrame, df_or: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enrichit le dataset BO avec les colonnes Pointage utiles :
        équipe_principale, technicien_principal_nom, total_heures Pointage,
        date_premier_pointage.
        """
        if df.empty or df_or.empty:
            return df

        cols_to_join = [c for c in [
            "or_id",
            "equipe_principale",
            "technicien_principal_nom",
            "total_heures",
            "date_premier_pointage",
            "date_dernier_pointage",
            "nb_jours_travailles",
            "nb_techniciens",
        ] if c in df_or.columns]

        df_join = df_or[cols_to_join].copy()
        df_join["or_id"] = df_join["or_id"].astype(str).str.strip()
        df["or_id"]      = df["or_id"].astype(str).str.strip()

        merged = df.merge(df_join, on="or_id", how="left")

        # Si date_creation BO absente, utiliser date_premier_pointage
        if "date_creation" not in merged.columns or merged["date_creation"].isna().all():
            merged["date_creation"] = merged.get("date_premier_pointage")

        logger.info(
            f"Enrichissement Pointage: {merged['equipe_principale'].notna().sum()} "
            f"OR appariés sur {len(merged)}"
        )
        return merged

    def _flag_ecart_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare duree_pointage_tot (BO) avec total_heures (Pointage).
        Un écart > seuil indique une incohérence entre les deux systèmes.
        """
        if "duree_pointage_tot" not in df.columns or "total_heures" not in df.columns:
            return df

        df = df.copy()
        bo_h  = pd.to_numeric(df["duree_pointage_tot"], errors="coerce")
        pt_h  = pd.to_numeric(df["total_heures"], errors="coerce")

        both_valid = bo_h.notna() & pt_h.notna() & (pt_h > 0)
        ecart_rel  = np.where(both_valid, np.abs(bo_h - pt_h) / pt_h, np.nan)

        df["ecart_bo_pointage"]      = np.where(both_valid, bo_h - pt_h, np.nan)
        df["ecart_bo_pointage_rel"]  = ecart_rel
        df["alerte_incoherence"]     = np.where(
            pd.notna(ecart_rel) & (ecart_rel > self.config.ecart_sources_seuil),
            (
                f"⚠️ INCOHÉRENCE DONNÉES — Les heures dans Business Objects et dans le "
                f"Pointage diffèrent de plus de "
                f"{self.config.ecart_sources_seuil:.0%}. "
                f"Vérifier la synchronisation entre les deux systèmes."
            ),
            "",
        )
        return df

    # ------------------------------------------------------------------
    # Agrégats pour les graphiques dashboard
    # ------------------------------------------------------------------

    def _aggregate_by_equipe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrégat d'efficience par équipe — pour les graphiques de coaching."""
        if "equipe_principale" not in df.columns or df.empty:
            return pd.DataFrame()

        evaluables = df[df["efficience_ratio"].notna()]
        if evaluables.empty:
            return pd.DataFrame()

        agg = (
            evaluables.groupby("equipe_principale")
            .agg(
                nb_or=("or_id", "count"),
                ratio_moyen=("efficience_ratio", "mean"),
                ratio_median=("efficience_ratio", "median"),
                nb_sous_productif=("efficience_categorie", lambda x: (x == CAT_SOUS_PRODUCTIF).sum()),
                nb_depasse=("efficience_categorie", lambda x: (x == CAT_DEPASSE).sum()),
                nb_normal=("efficience_categorie", lambda x: (x == CAT_NORMAL).sum()),
                heures_pointees_total=("total_heures", "sum"),
                heures_reference_total=("temps_reference", "sum"),
            )
            .reset_index()
            .sort_values("ratio_moyen")
        )

        agg["pct_sous_productif"] = (agg["nb_sous_productif"] / agg["nb_or"] * 100).round(1)
        agg["ratio_moyen_pct"]    = (agg["ratio_moyen"] * 100).round(1)
        return agg

    def _aggregate_by_technicien(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrégat d'efficience par technicien — pour le coaching individuel."""
        if "technicien_principal_nom" not in df.columns or df.empty:
            return pd.DataFrame()

        evaluables = df[df["efficience_ratio"].notna()]
        if evaluables.empty:
            return pd.DataFrame()

        agg = (
            evaluables.groupby(["technicien_principal_nom", "equipe_principale"], dropna=False)
            .agg(
                nb_or=("or_id", "count"),
                ratio_moyen=("efficience_ratio", "mean"),
                nb_sous_productif=("efficience_categorie", lambda x: (x == CAT_SOUS_PRODUCTIF).sum()),
                nb_depasse=("efficience_categorie", lambda x: (x == CAT_DEPASSE).sum()),
                heures_pointees=("total_heures", "sum"),
                heures_reference=("temps_reference", "sum"),
            )
            .reset_index()
            .sort_values("ratio_moyen")
        )

        agg["ratio_moyen_pct"] = (agg["ratio_moyen"] * 100).round(1)
        return agg
