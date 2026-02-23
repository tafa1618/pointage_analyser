from __future__ import annotations

from dataclasses import dataclass
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


class MLModelError(Exception):
    """Raised when machine learning scoring fails."""


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Create a robust preprocessing graph for mixed-type tabular data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _extract_feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in frame.columns if col not in numeric_cols]
    if not numeric_cols and not categorical_cols:
        raise MLModelError("Aucune feature exploitable pour l'entraînement du modèle ML.")
    return numeric_cols, categorical_cols


def _normalize_inverse_to_0_100(scores: np.ndarray) -> np.ndarray:
    """Convert IsolationForest decision_function scores to anomaly score in [0, 100]."""
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if np.isclose(max_score, min_score):
        return np.zeros_like(scores, dtype=float)

    normalized = (scores - min_score) / (max_score - min_score)
    return (1.0 - normalized) * 100.0


def train_model(
    frame: pd.DataFrame,
    model_path: str | Path = "models/isolation_forest.joblib",
    contamination: float = 0.08,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and persist a production-ready Isolation Forest model."""
    if frame.empty:
        raise MLModelError("Dataset vide: entraînement impossible.")
    if frame.shape[0] < 5:
        raise MLModelError("Au moins 5 lignes sont requises pour entraîner Isolation Forest.")

    numeric_cols, categorical_cols = _extract_feature_groups(frame)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=300,
        n_jobs=-1,
    )

    try:
        transformed = preprocessor.fit_transform(frame)
        model.fit(transformed)
    except Exception as exc:  # pylint: disable=broad-except
        raise MLModelError(f"Erreur pendant l'entraînement du modèle: {exc}") from exc

    artifacts = {
        "preprocessor": preprocessor,
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "contamination": contamination,
        "random_state": random_state,
    }

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(artifacts, path)
    except Exception as exc:  # pylint: disable=broad-except
        raise MLModelError(f"Impossible de sauvegarder le modèle: {exc}") from exc

    return artifacts


def load_model(model_path: str | Path = "models/isolation_forest.joblib") -> dict[str, Any] | None:
    """Load a previously trained model; return None if not available."""
    path = Path(model_path)
    if not path.exists():
        return None

    try:
        artifacts = joblib.load(path)
    except Exception as exc:  # pylint: disable=broad-except
        raise MLModelError(f"Impossible de charger le modèle ML: {exc}") from exc

    expected_keys = {"preprocessor", "model", "numeric_cols", "categorical_cols"}
    if not isinstance(artifacts, dict) or not expected_keys.issubset(artifacts.keys()):
        raise MLModelError("Le fichier modèle est invalide ou incomplet.")

    return artifacts


def score_dataset(
    frame: pd.DataFrame,
    model_path: str | Path = "models/isolation_forest.joblib",
    contamination: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Score a dataset with automatic model loading.

    If model exists => load and score.
    Else => train + save + score.
    """
    if frame.empty:
        raise MLModelError("Dataset vide: scoring impossible.")

    artifacts = load_model(model_path)
    if artifacts is None:
        artifacts = train_model(
            frame=frame,
            model_path=model_path,
            contamination=contamination,
            random_state=random_state,
        )

    expected_cols = artifacts["numeric_cols"] + artifacts["categorical_cols"]
    missing_cols = [col for col in expected_cols if col not in frame.columns]
    if missing_cols:
        raise MLModelError(
            "Le dataset à scorer ne contient pas toutes les colonnes attendues par le modèle: "
            + ", ".join(missing_cols)
        )

    scoring_input = frame[expected_cols].copy()

    try:
        transformed = artifacts["preprocessor"].transform(scoring_input)
        raw_scores = artifacts["model"].decision_function(transformed)
        predictions = artifacts["model"].predict(transformed)
    except Exception as exc:  # pylint: disable=broad-except
        raise MLModelError(f"Erreur pendant le scoring Isolation Forest: {exc}") from exc

    scored = frame.copy()
    scored["ml_raw_decision_score"] = raw_scores
    scored["ml_anomaly_score"] = _normalize_inverse_to_0_100(raw_scores)
    scored["ml_anomaly_flag"] = predictions == -1
    return scored


@dataclass(slots=True)
class IsolationForestModel:
    """Thin orchestrator wrapper around train/load/score helpers."""

    contamination: float = 0.08
    random_state: int = 42
    model_path: str = "models/isolation_forest.joblib"

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        return score_dataset(
            frame=frame,
            model_path=self.model_path,
            contamination=self.contamination,
            random_state=self.random_state,
        )
