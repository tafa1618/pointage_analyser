"""Configuration centralisée — toutes les constantes métier en un seul endroit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Codes Position OR (colonne IE)
# ---------------------------------------------------------------------------
POSITION_CODES: dict[str, str] = {
    "EC": "En Cours",
    "CP": "Clôturé",
    "AF": "Attente Facture",
    "TT": "Inconnu",  # TODO: à préciser avec le métier
}

# Codes considérés comme "clôturé" pour les règles métier
POSITION_CLOSED: frozenset[str] = frozenset({"CP"})
POSITION_OPEN: frozenset[str] = frozenset({"EC"})
POSITION_PENDING: frozenset[str] = frozenset({"AF"})

# ---------------------------------------------------------------------------
# Jours fériés Sénégal (dates fixes)
# Format: (mois, jour)
# Les dates variables (Tabaski, Aïd, etc.) sont à alimenter annuellement
# ---------------------------------------------------------------------------
FERIES_FIXES: list[tuple[int, int]] = [
    (1, 1),   # Jour de l'An
    (4, 4),   # Journée du Sénégal (historique)
    (5, 1),   # Fête du Travail
    (8, 15),  # Assomption
    (11, 1),  # Toussaint
    (12, 25), # Noël
]

# ---------------------------------------------------------------------------
# Mapping colonnes brutes → colonnes normalisées
# Permet au préprocesseur de savoir quoi chercher dans chaque fichier
# ---------------------------------------------------------------------------
IE_COLUMN_MAP: dict[str, list[str]] = {
    "or_id":              ["OR", "or", "N° OR", "num_or", "numero_or"],
    "date_extraction":    ["xxx", "Date", "date_extraction"],
    "nom_client":         ["Nom client", "nom_client", "client"],
    "type_or":            ["Type OR", "type_or", "TypeOR"],
    "nature":             ["Nature", "nature"],
    "position":           ["Position", "position", "statut"],
    "num_serie":          ["N° de série", "numero_serie", "n_serie"],
    "designation":        ["Désignation matériel", "designation_materiel"],
    "nature_materiel":    ["Nature matériel", "nature_materiel"],
    "type_intervention":  ["Type intervention", "type_intervention"],
    "localisation":       ["Localisation", "localisation"],
}

POINTAGE_COLUMN_MAP: dict[str, list[str]] = {
    "or_id":          ["OR (Numéro)", "OR", "or", "N° OR"],
    "date_saisie":    ["Saisie heures - Date", "date_saisie", "date"],
    "salarie_num":    ["Salarié - Numéro", "salarie_numero", "matricule"],
    "salarie_nom":    ["Salarié - Nom", "salarie_nom", "nom"],
    "equipe_code":    ["Salarié - Equipe", "equipe_code"],
    "equipe_nom":     ["Salarié - Equipe(Nom)", "equipe_nom", "equipe"],
    "type_heure":     ["Type heure (Libellé)", "type_heure", "categorie_heure"],
    "heure_realisee": ["Heure realisee", "heure_realisee"],
    "hr_travaillee":  ["Hr_travaillée", "hr_travaillee", "Hr_travaillée"],
    "hr_totale":      ["Hr_Totale", "hr_totale", "heures_totales"],
    "facturable":     ["Facturable", "facturable"],
    "non_facturable": ["Non Facturable", "non_facturable"],
}

BO_COLUMN_MAP: dict[str, list[str]] = {
    "or_id":              ["N° OR (Segment)", "N° OR", "or", "num_or"],
    "date_creation":      ["Date Création OR (Segment)", "date_creation"],
    "date_cloture":       ["Date Clôture (Segment)", "date_cloture"],
    "date_facture":       ["Date Facture (Lignes)", "date_facture"],
    "nom_client":         ["Nom Client OR (or)", "nom_client"],
    "modele_equipement":  ["Modèle de l'équipement (Segment)", "modele"],
    "type_materiel":      ["Type matériel", "type_materiel"],
    "temps_vendu":        ["Temps vendu (OR)", "temps_vendu"],
    "temps_prevu_devis":  ["Temps prévu devis (OR)", "temps_prevu", "temps_prevu_devis"],
    "duree_pointage_prod":["Durée pointage agents productifs (OR)", "duree_ptg_prod"],
    "duree_pointage_tot": ["Durée pointage total (OR)", "duree_ptg_tot"],
    "qte_demandee":       ["Qté demandée client  (OR)", "qte_demandee"],
    "qte_facturee":       ["Qté Facturée (OR)", "qte_facturee"],
    "montant_mo":         ["Montant Mo (OR)", "montant_mo"],
    "montant_pieces":     ["Montant Pieces net (OR)", "montant_pieces"],
    "montant_frais":      ["Montant Frais net (OR)", "montant_frais"],
    "montant_total":      ["Montant Total (OR)", "montant_total"],
    "localisation":       ["Localisation (Segment)", "localisation"],
    "service":            ["Service (Lignes)", "service"],
    "num_intervention":   ["N° intervention (Segment)", "num_intervention"],
}

# Types d'heures hors-OR (lignes à exclure du pipeline OR, mais garder pour exhaustivité)
HORS_OR_TYPES: frozenset[str] = frozenset({
    "PAS DE TRAVAIL",
    "CONGES",
    "ABSENCE AUTORISEE",
    "REUNION",
    "PAUSE",
    "TÂCHES ADMINISTRATIVES",
    "NETTOYAGE ET RANGEMENT",
    "RECUPERATION",
    "MALADIE",
    "ATTENTE OUTILLAGES",
    "FORMATION",
    "VISITE MEDICALE",
})


@dataclass(frozen=True)
class ScoringConfig:
    """Tous les hyperparamètres et seuils métier — source unique de vérité."""

    # -- ML Isolation Forest --
    contamination: float = 0.08
    random_state: int = 42
    n_estimators: int = 300
    model_path: Path = field(default_factory=lambda: Path("models/isolation_forest.joblib"))

    # -- Score blending --
    rule_weight: float = 0.45
    ml_weight: float = 0.55
    anomaly_threshold: float = 0.60   # seuil de déclaration d'anomalie

    # -- Règles métier OR-level --
    max_hours_per_day: float = 12.0           # validé métier
    max_delay_first_pointage_days: int = 3    # délai max avant premier pointage
    high_missing_ratio: float = 0.40          # taux valeurs manquantes
    efficiency_low_threshold: float = 0.70   # efficience < 70% = anomalie
    overconsumption_threshold: float = 2.0   # surconsommation > 2h = anomalie

    # -- Exhaustivité calendrier --
    hours_normal_max: float = 8.0     # seuil normal/excessif (>8h = orange)
    hours_absent: float = 0.0         # seuil absent (=0 = rouge)

    # -- Préprocesseur --
    min_rows: int = 1
