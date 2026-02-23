from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""


class FeatureEngineer:
    """Create robust numerical features for anomaly detection."""

    def build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise FeatureEngineeringError("Dataset vide après fusion.")

        engineered = frame.copy()
        engineered = self._add_datetime_features(engineered)
        engineered = self._convert_numeric_like_columns(engineered)
        engineered = self._add_group_consistency_features(engineered)
        engineered = self._add_row_statistics(engineered)

        return engineered

    @staticmethod
    def _add_datetime_features(frame: pd.DataFrame) -> pd.DataFrame:
        date_candidates = [
            column
            for column in frame.columns
            if any(token in column for token in ("date", "timestamp", "time"))
        ]

        for col in date_candidates:
            parsed = pd.to_datetime(frame[col], errors="coerce", utc=True)
            if parsed.notna().sum() == 0:
                continue
            frame[f"{col}_year"] = parsed.dt.year
            frame[f"{col}_month"] = parsed.dt.month
            frame[f"{col}_day"] = parsed.dt.day
            frame[f"{col}_weekday"] = parsed.dt.dayofweek
            frame[f"{col}_hour"] = parsed.dt.hour

        return frame

    @staticmethod
    def _convert_numeric_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
        object_cols = frame.select_dtypes(include=["object"]).columns
        for col in object_cols:
            cleaned = frame[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() >= max(3, int(0.2 * len(frame))):
                frame[col] = converted
        return frame

    @staticmethod
    def _add_group_consistency_features(frame: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        entity_key = next(
            (key for key in ["employee_id", "matricule", "id"] if key in frame.columns),
            None,
        )

        if not entity_key or not numeric_cols:
            return frame

        for col in numeric_cols[:30]:
            group_mean = frame.groupby(entity_key)[col].transform("mean")
            frame[f"{col}_delta_group_mean"] = frame[col] - group_mean

        return frame

    @staticmethod
    def _add_row_statistics(frame: pd.DataFrame) -> pd.DataFrame:
        numeric_frame = frame.select_dtypes(include=[np.number])
        if numeric_frame.empty:
            raise FeatureEngineeringError(
                "Aucune colonne numérique disponible pour le scoring ML."
            )

        frame["row_numeric_mean"] = numeric_frame.mean(axis=1)
        frame["row_numeric_std"] = numeric_frame.std(axis=1).fillna(0.0)
        frame["row_numeric_min"] = numeric_frame.min(axis=1)
        frame["row_numeric_max"] = numeric_frame.max(axis=1)
        frame["row_missing_ratio"] = frame.isna().mean(axis=1)

        return frame
