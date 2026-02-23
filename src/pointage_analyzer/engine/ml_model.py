"""ML engine — Isolation Forest avec explainabilité feature-level."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pointage_analyzer.core.config import ScoringConfig

logger = logging.getLogger(__name__)


class MLModelError(Exception):
    """Raised when machine learning scoring fails."""


# Features numériques utilisées par le modèle
NUMERIC_FEATURES: list[str] = [
    "total_heures",
    "log_total_heures",
    "nb_lignes_pointage",
    "nb_jours_travailles",
    "heures_par_jour_moyen",
    "heures_jour_max",
    "variance_journaliere",
    "log_variance_jours",
    "ratio_reel_prevu",
    "ratio_reel_vendu",
    "delta_heures_prevu",
    "nb_techniciens",
    "lignes_par_jour",
]

CATEGORICAL_FEATURES: list[str] = [
    "type_or",
    "nature",
    "position",
    "localisation",
]


@dataclass
class IsolationForestModel:
    """Wrapper sklearn pipeline pour l'anomaly detection."""

    config: ScoringConfig = field(default_factory=ScoringConfig)
    _pipeline: Pipeline | None = field(default=None, init=False, repr=False)
    _numeric_cols: list[str] = field(default_factory=list, init=False, repr=False)
    _cat_cols: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def _build_pipeline(self, numeric_cols: list[str], cat_cols: list[str]) -> Pipeline:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="INCONNU")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="drop",
        )

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", IsolationForest(
                n_estimators=self.config.n_estimators,
                contamination=self.config.contamination,
                random_state=self.config.random_state,
                n_jobs=-1,
            )),
        ])

    def _select_features(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Sélectionne les features disponibles dans ce DataFrame."""
        numeric_cols = [c for c in NUMERIC_FEATURES if c in frame.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in frame.columns]

        if len(numeric_cols) < 2:
            raise MLModelError(
                f"Pas assez de features numériques disponibles. "
                f"Trouvé: {numeric_cols}. "
                f"Attendu: {NUMERIC_FEATURES[:5]}"
            )

        logger.info(
            f"Features ML: {len(numeric_cols)} numériques, {len(cat_cols)} catégorielles"
        )
        return numeric_cols, cat_cols

    def train(self, frame: pd.DataFrame) -> None:
        """Entraîne le modèle sur le dataset OR-level."""
        numeric_cols, cat_cols = self._select_features(frame)
        self._pipeline = self._build_pipeline(numeric_cols, cat_cols)
        # Stocke la liste des features pour score()
        self._numeric_cols = numeric_cols
        self._cat_cols = cat_cols
        self._pipeline.fit(frame)
        logger.info(f"Isolation Forest entraîné sur {len(frame)} OR")

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """
        Calcule le score d'anomalie ML pour chaque OR.

        Returns:
            pd.Series de valeurs entre 0 (normal) et 1 (très anormal)
        """
        if not self.is_trained:
            raise MLModelError("Modèle non entraîné. Appeler .train() d'abord.")

        # IsolationForest: decision_function → négatif = anomalie
        raw_scores = self._pipeline.decision_function(frame)
        # Normalise en [0, 1] : 0 = normal, 1 = anomalie
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s == min_s:
            normalized = np.zeros(len(raw_scores))
        else:
            normalized = (max_s - raw_scores) / (max_s - min_s)

        return pd.Series(normalized, index=frame.index, name="ml_score")

    def get_top_features(
        self, frame: pd.DataFrame, n_top: int = 3
    ) -> pd.DataFrame:
        """
        Approximation de l'importance des features par permutation.

        Pour chaque OR, identifie les features qui contribuent le plus
        à son score d'anomalie (explication simple non-probabiliste).

        Returns:
            DataFrame avec colonnes: or_id (index), top_feature_1, top_feature_2...
        """
        if not self.is_trained:
            return pd.DataFrame(index=frame.index)

        numeric_cols = getattr(self, "_numeric_cols", [])
        if not numeric_cols:
            return pd.DataFrame(index=frame.index)

        # Base score
        base_score = self.score(frame).values

        results: dict[str, list] = {f"top_feature_{i+1}": [] for i in range(n_top)}
        feature_impacts: list[tuple[str, float]] = []

        for col in numeric_cols[:10]:  # limiter pour performance
            frame_perm = frame.copy()
            frame_perm[col] = frame_perm[col].sample(frac=1, random_state=0).values
            perm_score = self.score(frame_perm).values
            impact = np.abs(perm_score - base_score).mean()
            feature_impacts.append((col, impact))

        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        top_feats = [name for name, _ in feature_impacts[:n_top]]

        for i, feat in enumerate(top_feats):
            results[f"top_feature_{i+1}"] = [feat] * len(frame)

        # Pad remaining
        for i in range(len(top_feats), n_top):
            results[f"top_feature_{i+1}"] = [""] * len(frame)

        return pd.DataFrame(results, index=frame.index)

    def save(self, path: Path | None = None) -> None:
        """Sérialise le pipeline entraîné."""
        dest = path or self.config.model_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, dest)
        logger.info(f"Modèle sauvegardé: {dest}")

    def load(self, path: Path | None = None) -> None:
        """Charge un pipeline sérialisé."""
        src = path or self.config.model_path
        if not src.exists():
            raise MLModelError(f"Fichier modèle introuvable: {src}")
        self._pipeline = joblib.load(src)
        logger.info(f"Modèle chargé: {src}")
