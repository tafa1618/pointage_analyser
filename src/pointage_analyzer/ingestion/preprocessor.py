"""Ingestion layer — lecture et harmonisation des fichiers Excel bruts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pointage_analyzer.core.config import ScoringConfig


class PreprocessingError(Exception):
    """Raised when ingestion or schema harmonization fails."""


def _normalize_token(value: str) -> str:
    """Lowercase + ASCII + underscores — stable column naming."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _find_first(columns: pd.Index | list[str], candidates: list[str]) -> str | None:
    """Return the first candidate present in columns (exact match after normalization)."""
    col_set = {_normalize_token(c): c for c in columns}
    for candidate in candidates:
        norm = _normalize_token(candidate)
        if norm in col_set:
            return col_set[norm]
        # Also try raw match
        if candidate in columns:
            return candidate
    return None


@dataclass(slots=True)
class DataPreprocessor:
    """
    Lit un fichier Excel et harmonise les colonnes vers un schéma canonique.

    Repose sur les column_map définis dans config.py pour chaque dataset.
    Garantit que la colonne `or_id` existe et est harmonisée.
    """

    config: ScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ScoringConfig()

    def read_excel(
        self,
        file_obj: Any,
        dataset_name: str,
        sheet_name: int | str = 0,
    ) -> pd.DataFrame:
        """
        Lit un fichier Excel avec gestion d'erreurs robuste.

        Args:
            file_obj: chemin fichier, BytesIO ou UploadedFile Streamlit
            dataset_name: 'IE', 'Pointage' ou 'BO' (pour messages d'erreur)
            sheet_name: index ou nom de la feuille

        Returns:
            DataFrame avec colonnes originales (pas encore harmonisées)
        """
        if file_obj is None:
            raise PreprocessingError(f"[{dataset_name}] Fichier manquant.")

        try:
            frame = pd.read_excel(file_obj, sheet_name=sheet_name, engine="openpyxl")
        except ValueError as exc:
            raise PreprocessingError(
                f"[{dataset_name}] Format Excel invalide: {exc}"
            ) from exc
        except ImportError as exc:
             raise PreprocessingError(
                f"[{dataset_name}] openpyxl requis: pip install openpyxl"
            ) from exc

        except Exception as exc:  # noqa: BLE001
            raise PreprocessingError(
                f"[{dataset_name}] Erreur lecture Excel: {exc}"
            ) from exc

        if frame.empty or len(frame) < self.config.min_rows:
            raise PreprocessingError(
                f"[{dataset_name}] Le fichier doit contenir au moins "
                f"{self.config.min_rows} ligne(s). Reçu: {len(frame)}."
            )

        return frame

    @staticmethod
    def harmonize_columns(
        frame: pd.DataFrame,
        column_map: dict[str, list[str]],
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Renomme les colonnes brutes vers leur nom canonique via column_map.

        Colonnes non reconnues sont conservées avec leur nom normalisé.
        """
        df = frame.copy()

        # Normalize all raw column names first
        raw_to_norm: dict[str, str] = {
            col: _normalize_token(col) for col in df.columns
        }
        df.columns = pd.Index([raw_to_norm[c] for c in df.columns])

        # Build reverse map: normalized_raw → canonical
        rename_map: dict[str, str] = {}
        for canonical, candidates in column_map.items():
            matched = _find_first(df.columns.tolist(), candidates)
            if matched is not None:
                rename_map[matched] = canonical

        df = df.rename(columns=rename_map)
        return df

    @staticmethod
    def ensure_or_id(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Garantit que `or_id` existe et est une chaîne harmonisée.
        Filtre les lignes sans OR valide.
        """
        if "or_id" not in frame.columns:
            raise PreprocessingError(
                f"[{dataset_name}] Impossible de détecter la clé OR. "
                f"Colonnes disponibles: {list(frame.columns[:15])}"
            )

        df = frame.copy()
        df["or_id"] = df["or_id"].map(_harmonize_or_key)

        before = len(df)
        df = df[df["or_id"].notna()].copy()
        after = len(df)

        if after == 0:
            raise PreprocessingError(
                f"[{dataset_name}] Aucune clé OR valide après harmonisation."
            )

        if before != after:
            print(
                f"[{dataset_name}] {before - after} lignes sans OR valide écartées "
                f"({after} conservées)."
            )

        return df

    @staticmethod
    def to_datetime_safe(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

    @staticmethod
    def to_numeric_safe(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(",", ".", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")


def _harmonize_or_key(value: Any) -> str | None:
    """Normalise un identifiant OR en chaîne alphanumérique uppercase."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "0":
        return None
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return text or None
