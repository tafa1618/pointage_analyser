"""
Contrôle d'exhaustivité — Pipeline INDEPENDANT du pipeline OR.

DataFrame de présence : 1 ligne = 1 (technicien × jour)
Construit depuis les 8 528 lignes brutes du Pointage, AVANT toute agrégation OR.
Aucun technicien ne peut être perdu.

Deux modes :
  - Matrice brute       : technicien × jour → heures
  - Matrice consolidée  : équipe × jour → moyenne d'heures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pointage_analyzer.core.config import FERIES_FIXES, ScoringConfig

logger = logging.getLogger(__name__)


class ExhaustiviteError(Exception):
    """Raised when presence matrix construction fails."""


# ---------------------------------------------------------------------------
# Couleurs sémantiques (retournées dans la matrice pour usage Streamlit/Plotly)
# ---------------------------------------------------------------------------
class PresenceStatus:
    PRESENT = "PRESENT"          # 0 < h <= 8  → vert
    ABSENT = "ABSENT"            # h == 0       → rouge
    EXCESSIF = "EXCESSIF"        # h > 8        → orange
    WEEKEND = "WEEKEND"          # sam/dim      → gris
    FERIE = "FERIE"              # jour férié   → bleu clair
    NON_CONCERNE = "N/A"         # technicien pas dans l'équipe ce mois-là


@dataclass(slots=True)
class ExhaustiviteBuilder:
    """
    Construit la matrice de présence technicien × jour.

    IMPORTANT : utilise le Pointage BRUT complet (OR=0 inclus).
    Les congés, maladies et autres hors-OR comptent comme heures normales
    (le technicien a pointé son absence, donc il n'est PAS absent au sens exhaustivité).
    """

    config: ScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ScoringConfig()

    def build_presence_dataframe(
        self, pointage_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construit le DataFrame brut de présence (technicien × jour → heures).

        Returns:
            DataFrame avec colonnes:
                salarie_num, salarie_nom, equipe_nom, date, heures_totales
        """
        pt = pointage_df.copy()

        # Colonnes requises
        date_col = "date_saisie" if "date_saisie" in pt.columns else None
        hr_col = "hr_totale" if "hr_totale" in pt.columns else "heure_realisee"
        nom_col = "salarie_nom" if "salarie_nom" in pt.columns else None
        num_col = "salarie_num" if "salarie_num" in pt.columns else None
        equipe_col = "equipe_nom" if "equipe_nom" in pt.columns else None

        if date_col is None:
            raise ExhaustiviteError(
                "Colonne de date introuvable dans le Pointage. "
                f"Colonnes disponibles: {list(pt.columns[:10])}"
            )
        if hr_col not in pt.columns:
            raise ExhaustiviteError(
                "Colonne d'heures introuvable dans le Pointage."
            )
        if nom_col is None and num_col is None:
            raise ExhaustiviteError(
                "Aucune colonne technicien (nom ou numéro) dans le Pointage."
            )

        # Parse dates
        pt["_date"] = pd.to_datetime(pt[date_col], errors="coerce", dayfirst=True)
        pt_valid = pt[pt["_date"].notna()].copy()

        if pt_valid.empty:
            raise ExhaustiviteError("Aucune date valide dans le Pointage.")

        # Parse heures
        pt_valid[hr_col] = pd.to_numeric(pt_valid[hr_col], errors="coerce").fillna(0.0)

        # Colonnes identité technicien
        group_cols = ["_date"]
        if num_col:
            group_cols.append(num_col)
        if nom_col:
            group_cols.append(nom_col)
        if equipe_col:
            group_cols.append(equipe_col)

        # Agrégation technicien × jour
        presence = (
            pt_valid.groupby(group_cols, as_index=False, dropna=False)
            .agg(heures_totales=(hr_col, "sum"))
        )

        # Renommage colonnes
        rename_map = {"_date": "date"}
        if num_col:
            rename_map[num_col] = "salarie_num"
        if nom_col:
            rename_map[nom_col] = "salarie_nom"
        if equipe_col:
            rename_map[equipe_col] = "equipe_nom"

        presence = presence.rename(columns=rename_map)
        presence["date"] = pd.to_datetime(presence["date"])

        # Colonnes temporelles
        presence["annee"] = presence["date"].dt.year
        presence["mois"] = presence["date"].dt.month
        presence["mois_label"] = presence["date"].dt.strftime("%Y-%m")
        presence["jour_semaine"] = presence["date"].dt.dayofweek  # 0=lundi, 6=dimanche
        presence["est_weekend"] = presence["jour_semaine"] >= 5

        logger.info(
            f"Matrice présence construite: {len(presence)} lignes "
            f"({presence['date'].nunique()} jours, "
            f"{presence.get('salarie_nom', presence.get('salarie_num', pd.Series())).nunique()} techniciens)"
        )
        return presence

    def get_filtered_matrix(
        self,
        presence_df: pd.DataFrame,
        equipe_filter: list[str] | None = None,
        mois_label: str | None = None,
        annee: int | None = None,
        mois: int | None = None,
    ) -> pd.DataFrame:
        """
        Filtre la matrice de présence par équipe et/ou mois.

        Args:
            presence_df: DataFrame brut issu de build_presence_dataframe()
            equipe_filter: liste d'équipes ('Atelier Machine', etc.) ou None = toutes
            mois_label: filtre ex: '2026-02' (prioritaire sur annee/mois)
            annee: filtre année
            mois: filtre mois numérique (1-12)

        Returns:
            DataFrame filtré
        """
        df = presence_df.copy()

        if mois_label is not None and mois_label != "Tous":
            df = df[df["mois_label"] == mois_label]
        elif annee is not None and mois is not None:
            df = df[(df["annee"] == annee) & (df["mois"] == mois)]

        if equipe_filter and "equipe_nom" in df.columns:
            df = df[df["equipe_nom"].isin(equipe_filter)]

        return df.copy()

    def build_pivot_calendar(
        self,
        presence_df_filtered: pd.DataFrame,
        country: str = "SN",
        use_nom: bool = True,
    ) -> pd.DataFrame:
        """
        Construit la matrice pivot technicien × jour avec statut coloré.

        Args:
            presence_df_filtered: DataFrame filtré (par équipe + mois)
            country: code pays ISO pour jours fériés (défaut: SN = Sénégal)
            use_nom: True = lignes = nom technicien, False = numéro

        Returns:
            DataFrame pivot avec valeurs numériques (heures) et
            DataFrame status (PRESENT/ABSENT/EXCESSIF/WEEKEND/FERIE)
        """
        df = presence_df_filtered.copy()

        if df.empty:
            raise ExhaustiviteError(
                "Aucune donnée après filtrage. Vérifier les filtres équipe/mois."
            )

        tech_col = "salarie_nom" if (use_nom and "salarie_nom" in df.columns) else "salarie_num"
        if tech_col not in df.columns:
            raise ExhaustiviteError(f"Colonne technicien '{tech_col}' introuvable.")

        # Pivot heures
        pivot_heures = df.pivot_table(
            index=tech_col,
            columns="date",
            values="heures_totales",
            aggfunc="sum",
            fill_value=0.0,
        )
        pivot_heures.columns = pd.DatetimeIndex(pivot_heures.columns)

        # Calcul des jours fériés pour la période
        dates = pivot_heures.columns
        feries = self._get_feries(dates, country=country)

        # Matrice de statuts
        status_matrix = self._compute_status_matrix(pivot_heures, feries)

        return pivot_heures, status_matrix

    def _compute_status_matrix(
        self,
        pivot_heures: pd.DataFrame,
        feries: set[date],
    ) -> pd.DataFrame:
        """Calcule le statut coloré pour chaque cellule (technicien × jour)."""
        status = pd.DataFrame(
            PresenceStatus.ABSENT,
            index=pivot_heures.index,
            columns=pivot_heures.columns,
        )

        for col_date in pivot_heures.columns:
            d = col_date.date() if hasattr(col_date, "date") else col_date
            is_weekend = d.weekday() >= 5
            is_ferie = d in feries

            col_heures = pivot_heures[col_date]

            if is_ferie:
                status[col_date] = np.where(
                    col_heures > self.config.hours_normal_max, PresenceStatus.EXCESSIF,
                    PresenceStatus.FERIE
                )
            elif is_weekend:
                status[col_date] = np.where(
                    col_heures > self.config.hours_normal_max, PresenceStatus.EXCESSIF,
                    PresenceStatus.WEEKEND
                )
            else:
                # Jour ouvrable
                status[col_date] = np.where(
                    col_heures > self.config.hours_normal_max, PresenceStatus.EXCESSIF,
                    np.where(col_heures > 0, PresenceStatus.PRESENT, PresenceStatus.ABSENT)
                )

        return status

    @staticmethod
    def _get_feries(dates: pd.DatetimeIndex, country: str = "SN") -> set[date]:
        """
        Retourne les dates fériées pour la période couverte par `dates`.

        Utilise la bibliothèque `holidays` si disponible, sinon fall-back
        sur FERIES_FIXES définis dans config.py.
        """
        if dates.empty:
            return set()

        years = set(dates.year.unique())
        feries: set[date] = set()

        try:
            import holidays as hols  # type: ignore[import]

            for year in years:
                country_hols = hols.country_holidays(country, years=year)
                feries.update(country_hols.keys())
        except (ImportError, NotImplementedError):
            # Fall-back sur les jours fixes Sénégal définis dans config
            for year in years:
                for month, day in FERIES_FIXES:
                    try:
                        feries.add(date(year, month, day))
                    except ValueError:
                        pass

        return feries

    def get_equipes_list(self, presence_df: pd.DataFrame) -> list[str]:
        """Retourne la liste triée des équipes présentes dans le fichier."""
        if "equipe_nom" not in presence_df.columns:
            return []
        return sorted(presence_df["equipe_nom"].dropna().unique().tolist())

    def get_mois_list(self, presence_df: pd.DataFrame) -> list[str]:
        """Retourne la liste triée des mois présents (format YYYY-MM)."""
        if "mois_label" not in presence_df.columns:
            return []
        return sorted(presence_df["mois_label"].dropna().unique().tolist())

    def compute_daily_stats(
        self, presence_df_filtered: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Statistiques journalières pour le mois filtré.

        Returns:
            DataFrame avec: date, nb_presents, nb_absents, nb_excessifs,
                            taux_presence, heures_moy
        """
        df = presence_df_filtered.copy()
        if df.empty:
            return pd.DataFrame()

        daily = (
            df.groupby("date")
            .agg(
                total_techniciens=("heures_totales", "size"),
                nb_presents=("heures_totales", lambda x: (x > 0).sum()),
                nb_absents=("heures_totales", lambda x: (x == 0).sum()),
                nb_excessifs=(
                    "heures_totales",
                    lambda x: (x > self.config.hours_normal_max).sum(),
                ),
                heures_moy=("heures_totales", "mean"),
                heures_total=("heures_totales", "sum"),
            )
            .reset_index()
        )
        daily["taux_presence"] = (
            daily["nb_presents"] / daily["total_techniciens"].replace(0, np.nan)
        ).fillna(0.0)
        daily["est_weekend"] = pd.to_datetime(daily["date"]).dt.dayofweek >= 5
        return daily
