from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from scoring.dataset_builder import ORDatasetBuilder
from scoring.feature_engineering import FeatureEngineer
from scoring.ml_model import IsolationForestModel
from scoring.preprocess import DataPreprocessor
from scoring.rule_engine import RuleEngine


class ScoringError(Exception):
    """Raised when the end-to-end scoring pipeline fails."""


@dataclass(slots=True)
class ORPerformanceScorer:
    """Orchestrates OR-level dataset build, feature engineering and anomaly scoring."""

    preprocessor: DataPreprocessor = field(default_factory=DataPreprocessor)
    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)
    rule_engine: RuleEngine = field(default_factory=RuleEngine)
    ml_model: IsolationForestModel = field(default_factory=IsolationForestModel)
    dataset_builder: ORDatasetBuilder = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_builder = ORDatasetBuilder(preprocessor=self.preprocessor)

    def run(
        self,
        ie_file: Any,
        pointage_file: Any,
        bo_file: Any,
    ) -> pd.DataFrame:
        try:
            ie_df = self.preprocessor.read_excel(ie_file, "IE")
            pointage_df = self.preprocessor.read_excel(pointage_file, "Pointage")
            bo_df = self.preprocessor.read_excel(bo_file, "BO")

            final_or_level = self.dataset_builder.build(ie_df, pointage_df, bo_df)
            featured = self.feature_engineer.build_features(final_or_level)
            ruled = self.rule_engine.apply(featured)
            scored = self.ml_model.score(ruled)

            scored["final_anomaly_score"] = (
                0.45 * scored["rule_anomaly_score"] + 0.55 * (scored["ml_anomaly_score"] / 100.0)
            ).clip(0.0, 1.0)
            scored["final_anomaly_flag"] = scored["final_anomaly_score"] >= 0.60

            return scored.sort_values("final_anomaly_score", ascending=False)
        except Exception as exc:  # pylint: disable=broad-except
            raise ScoringError(f"Pipeline de scoring interrompu: {exc}") from exc
