from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re
import unicodedata

import pandas as pd


class PreprocessingError(Exception):
    """Raised when ingestion or preprocessing fails."""


def _normalize_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


@dataclass(slots=True)
class DataPreprocessor:
    """Handle ingestion and schema harmonization for IE / Pointage / BO."""

    min_rows: int = 1

    def read_excel(self, file_obj: Any, dataset_name: str) -> pd.DataFrame:
        """Read uploaded Excel file with robust error handling."""
        if file_obj is None:
            raise PreprocessingError(f"Le fichier {dataset_name} est manquant.")

        try:
            frame = pd.read_excel(file_obj, engine="openpyxl")
        except ValueError as exc:
            raise PreprocessingError(f"Format Excel invalide pour {dataset_name}: {exc}") from exc
        except ImportError as exc:
            raise PreprocessingError(
                "Le package openpyxl est requis pour lire les fichiers Excel."
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise PreprocessingError(
                f"Erreur inattendue lors de la lecture du fichier Excel {dataset_name}: {exc}"
            ) from exc

        if frame.empty or frame.shape[0] < self.min_rows:
            raise PreprocessingError(
                f"Le fichier {dataset_name} doit contenir au moins {self.min_rows} ligne(s)."
            )

        normalized = self.normalize_columns(frame)
        normalized = self.ensure_or_key(normalized, dataset_name)
        return normalized

    @staticmethod
    def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [_normalize_token(column) for column in normalized.columns]
        return normalized

    def ensure_or_key(self, frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Find and harmonize OR identifier into canonical `or_id` column."""
        candidates = [
            "or",
            "n_or",
            "num_or",
            "numero_or",
            "or_numero_intervention",
            "numero_intervention",
            "n_intervention",
            "num_intervention",
            "intervention",
            "code_or",
            "id_or",
        ]

        matched = self._find_first_existing(frame.columns, candidates)
        if matched is None:
            raise PreprocessingError(
                f"Impossible de détecter la clé OR dans {dataset_name}. "
                "Colonnes attendues: N° OR / Numéro intervention / OR."
            )

        harmonized = frame.copy()
        harmonized["or_id"] = harmonized[matched].map(self._harmonize_or_key)
        harmonized = harmonized[harmonized["or_id"].notna()]

        if harmonized.empty:
            raise PreprocessingError(
                f"Aucune clé OR exploitable détectée dans {dataset_name} après harmonisation."
            )

        return harmonized

    @staticmethod
    def _find_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
        set_cols = set(columns)
        for candidate in candidates:
            if candidate in set_cols:
                return candidate
        # Fallback fuzzy contains
        for col in columns:
            if "or" in col and ("intervention" in col or col.endswith("_or") or col == "or"):
                return col
        return None

    @staticmethod
    def _harmonize_or_key(value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        # Remove spaces and separators but preserve alphanumerics
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        return text or None

    @staticmethod
    def to_datetime_safe(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

    @staticmethod
    def to_numeric_safe(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(",", ".", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")

