"""Rule engine — Règles déterministes d'anomalie OR-level."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pointage_analyzer.core.config import POSITION_CLOSED, ScoringConfig

logger = logging.getLogger(__name__)


class RuleEngineError(Exception):
    """Raised when rule application fails."""


@dataclass(slots=True)
class RuleResult:
    """Résultat d'une règle individuelle."""
    rule_name: str
    category: str      # TECHNIQUE | PROCESS | FINANCIER
    description: str   # Explication lisible par un non data-scientist
    score: float       # 0.0 (normal) → 1.0 (anomalie certaine)


@dataclass(slots=True)
class RuleEngine:
    """
    Applique les règles métier déterministes sur le dataset OR-level.

    8 règles organisées en 3 catégories :
      - TECHNIQUE  : anomalies numériques dans les pointages
      - PROCESS    : anomalies de processus (délais, absence de pointage)
      - FINANCIER  : anomalies entre réalisé et prévu/facturé

    Chaque règle contribue un score continu [0, 1] au lieu d'un flag binaire.
    Score final = moyenne pondérée des règles (exposé pour chaque OR).
    """

    config: ScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ScoringConfig()

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Applique toutes les règles et retourne le DataFrame enrichi.

        Colonnes ajoutées :
          - rule_<name>           : score [0, 1] par règle
          - rule_<name>_label     : description lisible
          - rule_score_technique  : score agrégé catégorie
          - rule_score_process    : score agrégé catégorie
          - rule_score_financier  : score agrégé catégorie
          - rule_score_total      : score global règles [0, 1]
          - rule_anomaly_types    : liste des règles déclenchées
        """
        df = frame.copy()
        all_rules: list[tuple[str, pd.Series, str, str]] = []

        # ---- TECHNIQUE ----
        r = self._rule_heures_negatives(df)
        df["rule_heures_negatives"] = r
        df["rule_heures_negatives_label"] = np.where(
            r > 0,
            "🔴 ERREUR DE SAISIE — Des heures négatives ont été enregistrées sur cet OR. "
            "Ce n'est pas physiquement possible : il s'agit d'une erreur de saisie à corriger "
            "manuellement dans le système de pointage.",
            ""
        )
        all_rules.append(("rule_heures_negatives", r, "TECHNIQUE", "Erreur de saisie (heures négatives)"))

        r = self._rule_heures_excessives(df)
        df["rule_heures_excessives"] = r
        df["rule_heures_excessives_label"] = np.where(
            r > 0.5,
            f"🔴 JOURNÉE EXCESSIVE — Au moins une journée dépasse {self.config.max_hours_per_day:.0f}h "
            f"pointées sur cet OR. Vérifier si plusieurs techniciens ont pointé sur la même ligne "
            f"ou si la date de saisie est erronée.",
            np.where(r > 0,
                     "🟡 JOURNÉE LONGUE — Une journée dépasse 10h pointées sur cet OR. "
                     "Cela reste dans les limites mais mérite une vérification.",
                     "")
        )
        all_rules.append(("rule_heures_excessives", r, "TECHNIQUE", "Journée excessive (>12h)"))

        r = self._rule_variance_anormale(df)
        df["rule_variance_anormale"] = r
        df["rule_variance_anormale_label"] = np.where(
            r > 0.5,
            "🟡 POINTAGE IRRÉGULIER — Les heures pointées varient énormément d'un jour à l'autre "
            "sur cet OR (ex : 1h un jour, 11h le lendemain). Cela peut indiquer un regroupement "
            "de pointages en fin de semaine ou des oublis de saisie rattrapés.",
            ""
        )
        all_rules.append(("rule_variance_anormale", r, "TECHNIQUE", "Pointage irrégulier"))

        # ---- PROCESS ----
        r = self._rule_or_sans_pointage(df)
        df["rule_or_sans_pointage"] = r
        df["rule_or_sans_pointage_label"] = np.where(
            r > 0,
            "🔴 OR SANS POINTAGE — Cet OR existe dans le système mais aucun technicien "
            "n'a enregistré d'heures dessus. Si des travaux ont bien été réalisés, "
            "les heures doivent être saisies. Sinon, vérifier si l'OR doit être clôturé ou annulé.",
            ""
        )
        all_rules.append(("rule_or_sans_pointage", r, "PROCESS", "OR ouvert sans heure pointée"))

        r = self._rule_retard_premier_pointage(df)
        df["rule_retard_premier_pointage"] = r
        df["rule_retard_premier_pointage_label"] = np.where(
            r > 0,
            f"🟡 DÉMARRAGE TARDIF — Le premier pointage sur cet OR a été enregistré plus de "
            f"{self.config.max_delay_first_pointage_days} jours après son ouverture. "
            f"Les travaux ont-ils bien démarré à la date prévue ? Un retard de démarrage "
            f"peut impacter la date de livraison client.",
            ""
        )
        all_rules.append(("rule_retard_premier_pointage", r, "PROCESS", "Démarrage tardif des travaux"))

        # ---- FINANCIER ----
        r = self._rule_or_cloture_sans_finance(df)
        df["rule_or_cloture_sans_finance"] = r
        df["rule_or_cloture_sans_finance_label"] = np.where(
            r > 0,
            "🔴 CLÔTURE INCOMPLÈTE — Cet OR est marqué 'Clôturé' dans l'ERP mais "
            "aucune donnée financière correspondante n'existe dans Business Objects. "
            "Il peut s'agir d'un OR non facturé ou d'une erreur de statut à corriger.",
            ""
        )
        all_rules.append(("rule_or_cloture_sans_finance", r, "FINANCIER", "OR clôturé sans facturation"))

        r = self._rule_surconsommation(df)
        df["rule_surconsommation"] = r
        df["rule_surconsommation_label"] = np.where(
            r > 0,
            f"🟡 DÉPASSEMENT BUDGET HEURES — Les heures réellement pointées dépassent les heures "
            f"prévues au devis de plus de {self.config.overconsumption_threshold:.0f}h. "
            f"Le coût réel est supérieur à ce qui a été vendu au client. "
            f"Une révision du devis ou une discussion commerciale peut être nécessaire.",
            ""
        )
        all_rules.append(("rule_surconsommation", r, "FINANCIER", "Dépassement budget heures"))

        r = self._rule_efficience_faible(df)
        df["rule_efficience_faible"] = r
        df["rule_efficience_faible_label"] = np.where(
            r > 0.5,
            f"🔴 SUR-CONSOMMATION HEURES — Le ratio heures réalisées / heures vendues dépasse "
            f"{1 / self.config.efficiency_low_threshold:.0%}. L'équipe a travaillé bien plus que "
            f"prévu sans que cela soit compensé commercialement. Vérifier si un avenant est nécessaire.",
            np.where(
                r > 0,
                f"🟡 SOUS-EFFICIENCE — Le ratio heures réalisées / heures vendues est inférieur à "
                f"{self.config.efficiency_low_threshold:.0%}. L'équipe a travaillé significativement "
                f"moins que prévu dans le contrat. Vérifier si des heures ont été oubliées dans "
                f"le pointage ou si l'OR a été réalisé partiellement.",
                ""
            )
        )
        all_rules.append(("rule_efficience_faible", r, "FINANCIER", "Efficience hors norme (sous ou sur)"))

        # ---- Scores agrégés par catégorie ----
        tech_rules  = [r for name, r, cat, _ in all_rules if cat == "TECHNIQUE"]
        proc_rules  = [r for name, r, cat, _ in all_rules if cat == "PROCESS"]
        fin_rules   = [r for name, r, cat, _ in all_rules if cat == "FINANCIER"]

        df["rule_score_technique"] = (
            pd.concat(tech_rules, axis=1).max(axis=1) if tech_rules else 0.0
        )
        df["rule_score_process"] = (
            pd.concat(proc_rules, axis=1).max(axis=1) if proc_rules else 0.0
        )
        df["rule_score_financier"] = (
            pd.concat(fin_rules, axis=1).max(axis=1) if fin_rules else 0.0
        )

        all_scores = pd.concat([r for _, r, _, _ in all_rules], axis=1)
        df["rule_score_total"] = all_scores.mean(axis=1).clip(0.0, 1.0)

        # ---- Texte explicatif des anomalies déclenchées ----
        label_cols = [c for c in df.columns if c.endswith("_label")]
        df["rule_anomaly_types"] = df[label_cols].apply(
            lambda row: " | ".join(v for v in row if v and str(v) != "nan"),
            axis=1,
        )

        nb_anomalies = (df["rule_score_total"] > 0.2).sum()
        logger.info(
            f"Rule engine: {nb_anomalies}/{len(df)} OR avec score règle > 0.2"
        )

        return df

    # ------------------------------------------------------------------
    # RÈGLES TECHNIQUE
    # ------------------------------------------------------------------
    def _rule_heures_negatives(self, df: pd.DataFrame) -> pd.Series:
        if "total_heures" not in df.columns:
            return pd.Series(0.0, index=df.index)
        return (df["total_heures"] < 0).astype(float)

    def _rule_heures_excessives(self, df: pd.DataFrame) -> pd.Series:
        if "heures_jour_max" not in df.columns:
            return pd.Series(0.0, index=df.index)
        max_h = df["heures_jour_max"].fillna(0.0)
        # Score graduel : 0 si ≤ 10h, 0.5 si 10-12h, 1.0 si > 12h
        return pd.Series(
            np.where(
                max_h > self.config.max_hours_per_day, 1.0,
                np.where(max_h > 10.0, 0.5, 0.0)
            ),
            index=df.index,
        )

    def _rule_variance_anormale(self, df: pd.DataFrame) -> pd.Series:
        if "variance_journaliere" not in df.columns:
            return pd.Series(0.0, index=df.index)
        var = df["variance_journaliere"].fillna(0.0)
        p95 = var.quantile(0.95)
        if p95 == 0:
            return pd.Series(0.0, index=df.index)
        return (var / p95).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # RÈGLES PROCESS
    # ------------------------------------------------------------------
    def _rule_or_sans_pointage(self, df: pd.DataFrame) -> pd.Series:
        if "has_pointage" in df.columns:
            return (~df["has_pointage"]).astype(float)
        return pd.Series(0.0, index=df.index)

    def _rule_retard_premier_pointage(self, df: pd.DataFrame) -> pd.Series:
        if not all(c in df.columns for c in ["date_premier_pointage", "date_creation_min"]):
            return pd.Series(0.0, index=df.index)
        d1 = pd.to_datetime(df["date_premier_pointage"], errors="coerce")
        d0 = pd.to_datetime(df["date_creation_min"], errors="coerce")
        delay = (d1 - d0).dt.days.fillna(0).clip(lower=0)
        max_d = self.config.max_delay_first_pointage_days
        return (delay > max_d).astype(float)

    # ------------------------------------------------------------------
    # RÈGLES FINANCIER
    # ------------------------------------------------------------------
    def _rule_or_cloture_sans_finance(self, df: pd.DataFrame) -> pd.Series:
        if "or_cloture_sans_finance" in df.columns:
            return df["or_cloture_sans_finance"].astype(float)
        return pd.Series(0.0, index=df.index)

    def _rule_surconsommation(self, df: pd.DataFrame) -> pd.Series:
        if not all(c in df.columns for c in ["total_heures", "temps_prevu_devis"]):
            return pd.Series(0.0, index=df.index)
        delta = (
            df["total_heures"].fillna(0.0) - df["temps_prevu_devis"].fillna(0.0)
        )
        return (delta > self.config.overconsumption_threshold).astype(float)

    def _rule_efficience_faible(self, df: pd.DataFrame) -> pd.Series:
        if "ratio_reel_vendu" not in df.columns:
            return pd.Series(0.0, index=df.index)
        ratio = df["ratio_reel_vendu"].fillna(1.0)
        threshold = self.config.efficiency_low_threshold
        over_threshold = 1.0 / threshold  # ex: 1/0.70 ≈ 1.43 → sur-consommation
        return pd.Series(
            np.where(
                ratio > over_threshold, 1.0,   # sur-consommation (score max)
                np.where(ratio < threshold, 0.5, 0.0)  # sous-efficience (score moyen)
            ),
            index=df.index,
        )
