"""
senegal_holidays.py
-------------------
Jours fériés officiels du Sénégal.
Couvre 2024, 2025, 2026 — les fêtes islamiques sont des estimations
basées sur le calendrier hégirien ; à ajuster si l'annonce officielle diffère.

Usage :
    from senegal_holidays import get_holidays_sn, nb_jours_ouvres

    feries = get_holidays_sn(2026)
    jours  = nb_jours_ouvres(date_debut, date_fin, annee=2026)
"""

from __future__ import annotations
import pandas as pd
from datetime import date


# ══════════════════════════════════════════════════════════════════════
# JOURS FÉRIÉS PAR ANNÉE
# Fêtes fixes  → dates certaines
# Fêtes islamiques → estimations (dépendent de l'observation de la lune)
# ══════════════════════════════════════════════════════════════════════

_FIXED: list[tuple[int, int, str]] = [
    # (mois, jour, libellé)
    (1,  1,  "Nouvel An"),
    (4,  4,  "Fête de l'Indépendance"),
    (5,  1,  "Fête du Travail"),
    (8,  15, "Assomption"),
    (11, 1,  "Toussaint"),
    (12, 25, "Noël"),
]

# Fêtes chrétiennes mobiles (Pâques + dépendantes)
# Calculées via l'algorithme de Meeus/Jones/Butcher
def _paques(annee: int) -> date:
    a = annee % 19
    b, c = divmod(annee, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(annee, month, day + 1)


# Fêtes islamiques estimées (±1 jour possible selon annonce officielle)
# Source : calcul hégirien approximatif
_ISLAMIC: dict[int, list[tuple[str, date]]] = {
    2024: [
        ("Tamkharit (Achoura)",    date(2024,  7, 17)),
        ("Maouloud (Naissance du Prophète)", date(2024, 9, 15)),
        ("Korité (Aïd el-Fitr)",   date(2024,  4, 10)),
        ("Tabaski (Aïd el-Adha)", date(2024,  6, 17)),
        ("Jour de l'an islamique", date(2024,  7,  7)),
    ],
    2025: [
        ("Tamkharit (Achoura)",    date(2025,  7,  5)),
        ("Maouloud (Naissance du Prophète)", date(2025, 9,  4)),
        ("Korité (Aïd el-Fitr)",   date(2025,  3, 30)),
        ("Tabaski (Aïd el-Adha)", date(2025,  6,  6)),
        ("Jour de l'an islamique", date(2025,  6, 26)),
    ],
    2026: [
        ("Tamkharit (Achoura)",    date(2026,  6, 25)),
        ("Maouloud (Naissance du Prophète)", date(2026, 8, 25)),
        ("Korité (Aïd el-Fitr)",   date(2026,  3, 20)),
        ("Tabaski (Aïd el-Adha)", date(2026,  5, 27)),
        ("Jour de l'an islamique", date(2026,  6, 16)),
    ],
}


def get_holidays_sn(annee: int) -> set[date]:
    """
    Retourne l'ensemble des jours fériés officiels du Sénégal pour une année.
    Inclut : fêtes fixes, fêtes chrétiennes mobiles, fêtes islamiques estimées.
    """
    feries: set[date] = set()

    # Fêtes fixes
    for mois, jour, _ in _FIXED:
        feries.add(date(annee, mois, jour))

    # Pâques + lundi de Pâques + Ascension + Pentecôte
    paques = _paques(annee)
    from datetime import timedelta
    feries.add(paques)
    feries.add(paques + timedelta(days=1))   # Lundi de Pâques
    feries.add(paques + timedelta(days=39))  # Ascension
    feries.add(paques + timedelta(days=49))  # Lundi de Pentecôte

    # Fêtes islamiques
    for _, d in _ISLAMIC.get(annee, []):
        feries.add(d)

    return feries


def get_holidays_range(date_debut: "pd.Timestamp | date", date_fin: "pd.Timestamp | date") -> set[date]:
    """
    Retourne tous les jours fériés entre date_debut et date_fin (inclus),
    en couvrant toutes les années concernées.
    """
    if hasattr(date_debut, "date"):
        date_debut = date_debut.date()
    if hasattr(date_fin, "date"):
        date_fin = date_fin.date()

    feries: set[date] = set()
    for annee in range(date_debut.year, date_fin.year + 1):
        feries |= get_holidays_sn(annee)

    return {d for d in feries if date_debut <= d <= date_fin}


def nb_jours_ouvres(
    date_debut: "pd.Timestamp | date",
    date_fin: "pd.Timestamp | date",
) -> int:
    """
    Calcule le nombre de jours ouvrés réels dans la plage [date_debut, date_fin].
    Exclut : samedis, dimanches, jours fériés Sénégal.

    Utilisé dans l'export exhaustivité pour le dénominateur du taux de présence.
    """
    if hasattr(date_debut, "date"):
        date_debut = date_debut.date()
    if hasattr(date_fin, "date"):
        date_fin = date_fin.date()

    feries = get_holidays_range(date_debut, date_fin)

    tous_jours = pd.date_range(str(date_debut), str(date_fin), freq="D")
    return sum(
        1 for d in tous_jours
        if d.weekday() < 5 and d.date() not in feries
    )


def is_jour_ouvre(d: "pd.Timestamp | date", feries: set[date] | None = None) -> bool:
    """Retourne True si le jour est un jour ouvré (lun-ven, hors férié)."""
    if hasattr(d, "date"):
        d = d.date()
    if feries is None:
        feries = get_holidays_sn(d.year)
    return d.weekday() < 5 and d not in feries


# ── Accès rapide pour le heatmap (PresenceStatus.FERIE) ───────────────
def build_ferie_set(df: pd.DataFrame) -> set[date]:
    """
    Construit le set de fériés pour toutes les années présentes dans df['date'].
    Pratique pour marquer les cellules du calendrier.
    """
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return set()
    feries: set[date] = set()
    for annee in dates.dt.year.unique():
        feries |= get_holidays_sn(int(annee))
    return feries
