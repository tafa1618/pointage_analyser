from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RuleEngine:
    """Apply deterministic anomaly rules."""

    max_hours_threshold: float = 16.0
    high_missing_ratio_threshold: float = 0.4

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        rule_score = np.zeros(len(scored), dtype=float)

        negative_numeric_mask = self._detect_negative_numeric_values(scored)
        rule_score += negative_numeric_mask.astype(float) * 0.35

        excessive_hours_mask = self._detect_excessive_hours(scored)
        rule_score += excessive_hours_mask.astype(float) * 0.35

        missing_ratio_mask = scored.isna().mean(axis=1) > self.high_missing_ratio_threshold
        rule_score += missing_ratio_mask.astype(float) * 0.30

        scored["rule_negative_values"] = negative_numeric_mask
        scored["rule_excessive_hours"] = excessive_hours_mask
        scored["rule_high_missing"] = missing_ratio_mask
        scored["rule_anomaly_score"] = np.clip(rule_score, 0.0, 1.0)
        scored["rule_anomaly_flag"] = scored["rule_anomaly_score"] >= 0.5

        return scored

    @staticmethod
    def _detect_negative_numeric_values(frame: pd.DataFrame) -> pd.Series:
        numeric_df = frame.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.Series(False, index=frame.index)

        protected_keywords = ("id", "year", "month", "day", "weekday", "hour")
        check_cols = [
            col for col in numeric_df.columns if not any(key in col.lower() for key in protected_keywords)
        ]
        if not check_cols:
            return pd.Series(False, index=frame.index)

        return (numeric_df[check_cols] < 0).any(axis=1)

    def _detect_excessive_hours(self, frame: pd.DataFrame) -> pd.Series:
        hour_cols = [
            col
            for col in frame.columns
            if any(token in col.lower() for token in ("hour", "heure", "duration", "duree"))
        ]
        if not hour_cols:
            return pd.Series(False, index=frame.index)

        numeric_hours = frame[hour_cols].apply(pd.to_numeric, errors="coerce")
        return (numeric_hours > self.max_hours_threshold).any(axis=1)
scoring/scorer.py
