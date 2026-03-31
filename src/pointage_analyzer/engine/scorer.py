
"""
Orchestrateur principal — point d'entrée unique du pipeline complet.

Quatre pipelines indépendants :
  1. Pipeline OR-level      (df_or)        → scoring anomalies
  2. Pipeline Présence      (df_presence)  → contrôle d'exhaustivité
  3. Pipeline Productivité  (productivite) → ratios heures facturables/totales
  4. Pipeline Efficience    (efficience)   → ratios pointé/référence

Aucune logique métier dans l'UI (dashboard/app.py appellera uniquement ce module).
"""

from __future__ import annotations
import traceback
import numpy as np

import logging
from dataclasses import dataclass, field

import pandas as pd

from pointage_analyzer.core.config import (
    BO_COLUMN_MAP,
    IE_COLUMN_MAP,
    POINTAGE_COLUMN_MAP,
    ScoringConfig,
)
from pointage_analyzer.engine.ml_model import IsolationForestModel
from pointage_analyzer.engine.rule_engine import RuleEngine
from pointage_analyzer.ingestion.preprocessor import DataPreprocessor
from pointage_analyzer.pipeline.dataset_builder import ORDatasetBuilder
from pointage_analyzer.pipeline.exhaustivite_builder import ExhaustiviteBuilder
from pointage_analyzer.pipeline.efficience_builder import EfficienceBuilder, EfficienceResult
from pointage_analyzer.pipeline.feature_engineering import FeatureEngineer
from pointage_analyzer.pipeline.productivite_builder import ProductiviteBuilder, ProductiviteResult

logger = logging.getLogger(__name__)


class ScoringError(Exception):
    """Raised when the end-to-end pipeline fails."""


@dataclass
class PipelineResult:
    """Résultats des trois pipelines."""

    df_or: pd.DataFrame
    df_presence: pd.DataFrame
    efficience: EfficienceResult | None = None
    productivite: ProductiviteResult | None = None  # ← ajout
    metadata: dict = field(default_factory=dict)  # Statistiques de trace
    TEST_ERREUR_VOLONTAIRE: str = "je suis le bon scorer"  
    


@dataclass
class ORPerformanceScorer:
    """
    Orchestrateur principal du pipeline.

    Utilisation:
        scorer = ORPerformanceScorer()
        result = scorer.run(ie_file, pointage_file, bo_file)
        df_or       = result.df_or       # pour les onglets OR / Anomalies
        df_presence = result.df_presence # pour l'onglet Exhaustivité
    """

    config: ScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ScoringConfig()

    def run(
        self,
        ie_file,
        pointage_file,
        bo_file,
        ie_sheet: int | str = "IE",
        pointage_sheet: int | str = "Pointage",
        bo_sheet: int | str = "BO",
    ) -> PipelineResult:
        """
        Exécute les deux pipelines et retourne un PipelineResult.

        Args:
            ie_file:       fichier IE (chemin, BytesIO, UploadedFile)
            pointage_file: fichier Pointage
            bo_file:       fichier BO
            ie_sheet:      nom/index feuille IE
            pointage_sheet: nom/index feuille Pointage
            bo_sheet:      nom/index feuille BO

        Returns:
            PipelineResult avec df_or et df_presence
        """
        np.random.seed(self.config.random_state) #fixer un seed global
        preprocessor = DataPreprocessor(config=self.config)

        # ==============================================================
        # 1. INGESTION — lecture et harmonisation
        # ==============================================================
        logger.info("=== INGESTION ===")
        try:
            # --- Pointage (obligatoire) ---
            pt_raw = preprocessor.read_excel(pointage_file, "Pointage", pointage_sheet)
            pt_harm = preprocessor.harmonize_columns(pt_raw, POINTAGE_COLUMN_MAP, "Pointage")

            # Normalise or_id dans Pointage (garde les 0 pour l'exhaustivité)
            if "or_id" in pt_harm.columns:
                pt_harm["_or_id_valid"] = pt_harm["or_id"].map(
                    lambda v: None if str(v).strip() in ("0", "nan", "") else str(v).strip()
                )
            else:
                pt_harm["_or_id_valid"] = None

            # Normalise les colonnes de date et heures
            if "date_saisie" in pt_harm.columns:
                pt_harm["date_saisie"] = preprocessor.to_datetime_safe(pt_harm["date_saisie"])
            for col in ["hr_totale", "heure_realisee", "facturable", "non_facturable"]:
                if col in pt_harm.columns:
                    pt_harm[col] = preprocessor.to_numeric_safe(pt_harm[col]).fillna(0.0)

            # Copie avec or_id normalisé pour le pipeline OR
            pt_for_or = pt_harm.copy()
            pt_for_or["or_id"] = pt_for_or["_or_id_valid"]
            pt_for_or = pt_for_or[pt_for_or["or_id"].notna()].copy()

            logger.info(
                f"Pointage: {len(pt_harm)} lignes total | "
                f"{len(pt_for_or)} avec OR valide"
            )

            # --- IE (optionnel) ---
            ie_df = None
            try:
                ie_raw = preprocessor.read_excel(ie_file, "IE", ie_sheet)
                ie_harm = preprocessor.harmonize_columns(ie_raw, IE_COLUMN_MAP, "IE")
                if "or_id" in ie_harm.columns:
                    ie_harm["or_id"] = ie_harm["or_id"].astype(str).str.strip()
                    ie_df = ie_harm
                    logger.info(f"IE: {len(ie_df)} lignes ({ie_df['or_id'].nunique()} OR)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"IE ignoré (non critique): {exc}")

            # --- BO (optionnel) ---
            bo_df = None
            try:
                bo_raw = preprocessor.read_excel(bo_file, "BO", bo_sheet)
                bo_harm = preprocessor.harmonize_columns(bo_raw, BO_COLUMN_MAP, "BO")
                if "or_id" in bo_harm.columns:
                    bo_harm["or_id"] = bo_harm["or_id"].astype(str).str.strip()
                    bo_df = bo_harm
                    logger.info(f"BO: {len(bo_df)} lignes ({bo_df['or_id'].nunique()} OR)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"BO ignoré (non critique): {exc}")

        except Exception as exc:
            raise ScoringError(f"Erreur ingestion: {exc}") from exc

        # ==============================================================
        # 2. PIPELINE PRÉSENCE (exhaustivité) — INDÉPENDANT
        # Utilise le Pointage BRUT complet (OR=0 inclus)
        # ==============================================================
        logger.info("=== PIPELINE PRÉSENCE ===")
        try:
            exhaustivite_builder = ExhaustiviteBuilder(config=self.config)
            df_presence = exhaustivite_builder.build_presence_dataframe(pt_harm)
        except Exception as exc:
            logger.error(f"Erreur pipeline présence: {exc}")
            df_presence = pd.DataFrame()

        # ==============================================================
        # 3. PIPELINE EFFICIENCE (BO)
        # ==============================================================
        logger.info("=== PIPELINE EFFICIENCE ===")
        efficience_result: EfficienceResult | None = None
        try:
            if bo_df is not None and not bo_df.empty:
                eff_builder = EfficienceBuilder(config=self.config)
                efficience_result = eff_builder.build(bo=bo_df, df_or=pd.DataFrame())
            else:
                logger.info("BO absent — pipeline efficience ignoré")
        except Exception as exc:
            logger.warning(f"Erreur pipeline efficience (non critique): {exc}")

        # ==============================================================
        # ==============================================================
        # 3. PIPELINE PRODUCTIVITÉ
        # ==============================================================
        logger.info("=== PIPELINE PRODUCTIVITÉ ===")
        productivite_result: ProductiviteResult | None = None
        try:
            prod_builder = ProductiviteBuilder()
            productivite_result = prod_builder.build(pt_harm)
        except Exception as exc:
            logger.warning(f"Erreur pipeline productivité (non critique): {exc}")
            
            traceback.print_exc()
        # 4. PIPELINE OR-LEVEL
        # ==============================================================
        logger.info("=== PIPELINE OR-LEVEL ===")
        try:
            # Dataset builder
            builder = ORDatasetBuilder(
                preprocessor=preprocessor, config=self.config
            )
            df_or = builder.build(ie_df, pt_for_or, bo_df)

            # Feature engineering
            fe = FeatureEngineer()
            df_or = fe.build_features(df_or)

            # Rule engine
            rule_engine = RuleEngine(config=self.config)
            df_or = rule_engine.apply(df_or)

            # ML scoring (train + score sur les mêmes données — inductive)
            ml_model = IsolationForestModel(config=self.config)
            ml_model.train(df_or)
            df_or["ml_score"] = ml_model.score(df_or)

            # Score final (unique, dans le scorer, PAS dans l'UI)
            df_or = self._compute_final_score(df_or)

            logger.info(
                f"=== Pipeline terminé: {len(df_or)} OR scorés | "
                f"{(df_or['anomaly_flag']).sum()} anomalies détectées ==="
            )

        except Exception as exc:
            raise ScoringError(f"Erreur pipeline OR: {exc}") from exc

        # Enrichissement efficience avec le df_or final
        if efficience_result is not None and bo_df is not None:
            try:
                eff_builder2 = EfficienceBuilder(config=self.config)
                efficience_result = eff_builder2.build(bo=bo_df, df_or=df_or)
            except Exception as exc:
                logger.warning(f"Enrichissement efficience échoué: {exc}")

        # ==============================================================
        # 5. Métadonnées de traçabilité
        # ==============================================================
        metadata = {
            "nb_or_total": len(df_or),
            "nb_or_anomalies": int(df_or.get("anomaly_flag", pd.Series(False)).sum()),
            "nb_techniciens": (
                df_presence["salarie_nom"].nunique()
                if "salarie_nom" in df_presence.columns else 0
            ),
            "nb_jours": (
                df_presence["date"].nunique()
                if "date" in df_presence.columns else 0
            ),
            "nb_or_avec_bo": int(df_or.get("has_bo", pd.Series(False)).sum()),
            "nb_or_avec_ie": int(df_or.get("has_ie", pd.Series(False)).sum()),
            "nb_or_efficience_evaluables": (
                efficience_result.nb_or_evaluables if efficience_result else 0
            ),
        }

        return PipelineResult(
            df_or=df_or,
            df_presence=df_presence,
            efficience=efficience_result,
            productivite=productivite_result,
            metadata=metadata,
        )

    def _compute_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le score final unique dans le scorer (pas dans l'UI).

        score_final = rule_weight * rule_score_total + ml_weight * ml_score
        anomaly_flag = score_final >= anomaly_threshold
        """
        rule_col = "rule_score_total" if "rule_score_total" in df.columns else None
        ml_col = "ml_score" if "ml_score" in df.columns else None

        if rule_col and ml_col:
            df["score_final"] = (
                self.config.rule_weight * df[rule_col]
                + self.config.ml_weight * df[ml_col]
            ).clip(0.0, 1.0)
        elif rule_col:
            df["score_final"] = df[rule_col]
        elif ml_col:
            df["score_final"] = df[ml_col]
        else:
            df["score_final"] = 0.0

        df["anomaly_flag"] = df["score_final"] >= self.config.anomaly_threshold

        # Sévérité lisible
        df["severity"] = pd.cut(
            df["score_final"],
            bins=[-0.001, 0.30, 0.60, 0.80, 1.001],
            labels=["Normal", "Faible", "Modéré", "Critique"],
        )

        return df
