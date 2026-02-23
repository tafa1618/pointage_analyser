"""
Smoke test end-to-end du pipeline avec le fichier Classeur3.xlsx réel.
Test la logique de données sans les libs ML (sklearn/joblib/plotly non installés).
"""
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")

print("=" * 60)
print("SMOKE TEST — pipeline pointage_analyzer")
print("=" * 60)

# -----------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------
from pointage_analyzer.core.config import ScoringConfig, POSITION_CODES
cfg = ScoringConfig()
print(f"\n[1] Config OK — contamination={cfg.contamination}, "
      f"rule_weight={cfg.rule_weight}, anomaly_threshold={cfg.anomaly_threshold}")
print(f"    Position codes: {POSITION_CODES}")

# -----------------------------------------------------------------------
# 2. Preprocessor
# -----------------------------------------------------------------------
from pointage_analyzer.ingestion.preprocessor import DataPreprocessor
from pointage_analyzer.core.config import IE_COLUMN_MAP, POINTAGE_COLUMN_MAP, BO_COLUMN_MAP
import pandas as pd

preprocessor = DataPreprocessor(config=cfg)

xl = pd.ExcelFile("data/Classeur3.xlsx", engine="openpyxl")

ie_raw  = preprocessor.read_excel("data/Classeur3.xlsx", "IE",       sheet_name="IE")
pt_raw  = preprocessor.read_excel("data/Classeur3.xlsx", "Pointage", sheet_name="Pointage")
bo_raw  = preprocessor.read_excel("data/Classeur3.xlsx", "BO",       sheet_name="BO")

ie  = preprocessor.harmonize_columns(ie_raw,  IE_COLUMN_MAP,       "IE")
pt  = preprocessor.harmonize_columns(pt_raw,  POINTAGE_COLUMN_MAP, "Pointage")
bo  = preprocessor.harmonize_columns(bo_raw,  BO_COLUMN_MAP,       "BO")

print(f"\n[2] Preprocessor OK")
print(f"    IE:       {ie.shape}  — colonnes or_id: {'or_id' in ie.columns}")
print(f"    Pointage: {pt.shape}  — colonnes or_id: {'or_id' in pt.columns}")
print(f"    BO:       {bo.shape}  — colonnes or_id: {'or_id' in bo.columns}")
print(f"    IE cols:  {list(ie.columns[:6])}")
print(f"    PT cols:  {list(pt.columns[:6])}")
print(f"    BO cols:  {list(bo.columns[:6])}")

# -----------------------------------------------------------------------
# 3. Exhaustivité Builder (PIPELINE INDÉPENDANT — priorité)
# -----------------------------------------------------------------------
from pointage_analyzer.pipeline.exhaustivite_builder import ExhaustiviteBuilder, PresenceStatus

# Normalise dates et heures pour le pointage complet
if "date_saisie" in pt.columns:
    pt["date_saisie"] = pd.to_datetime(pt["date_saisie"], errors="coerce", dayfirst=True)
for col in ["hr_totale", "heure_realisee"]:
    if col in pt.columns:
        pt[col] = pd.to_numeric(pt[col], errors="coerce").fillna(0.0)

exh_builder = ExhaustiviteBuilder(config=cfg)
df_presence = exh_builder.build_presence_dataframe(pt)

print(f"\n[3] ExhaustiviteBuilder OK")
print(f"    Matrice présence: {df_presence.shape}")
print(f"    Techniciens: {df_presence['salarie_nom'].nunique() if 'salarie_nom' in df_presence.columns else '?'}")
print(f"    Jours couverts: {df_presence['date'].nunique()}")
print(f"    Équipes: {df_presence['equipe_nom'].nunique() if 'equipe_nom' in df_presence.columns else '?'}")
print(f"    Mois disponibles: {sorted(df_presence['mois_label'].unique().tolist())}")

# Filtre par équipe
equipes = exh_builder.get_equipes_list(df_presence)
print(f"    Équipes disponibles: {equipes}")

# Test filtre + pivot sur premier mois dispo
mois_dispo = exh_builder.get_mois_list(df_presence)
if mois_dispo:
    mois_test = mois_dispo[-1]
    df_filt = exh_builder.get_filtered_matrix(df_presence, mois_label=mois_test)
    print(f"\n    Test filtre mois={mois_test}: {df_filt.shape} lignes")
    try:
        pivot_h, status_m = exh_builder.build_pivot_calendar(df_filt)
        print(f"    Pivot construit: {pivot_h.shape} (techniciens × jours)")
        status_counts = status_m.values.flatten()
        from collections import Counter
        counts = Counter(status_counts)
        print(f"    Statuts: {dict(counts)}")
    except Exception as e:
        print(f"    Pivot ERREUR: {e}")

# Stats journalières
daily = exh_builder.compute_daily_stats(df_filt if mois_dispo else df_presence)
print(f"    Stats journalières: {daily.shape}")
print(f"    {daily[['date', 'nb_presents', 'nb_absents', 'taux_presence']].head(5).to_string()}")

# -----------------------------------------------------------------------
# 4. Dataset OR-level Builder
# -----------------------------------------------------------------------
from pointage_analyzer.pipeline.dataset_builder import ORDatasetBuilder

# Préparer pt avec or_id normalisé (sans les OR=0)
pt_with_orid = pt.copy()
pt_with_orid["or_id"] = pt_with_orid["or_id"].map(
    lambda v: None if str(v).strip() in ("0", "nan", "") else str(v).strip()
)
pt_or = pt_with_orid[pt_with_orid["or_id"].notna()].copy()

# Normalise or_id dans IE et BO
ie["or_id"] = ie["or_id"].astype(str).str.strip()
bo["or_id"] = bo["or_id"].astype(str).str.strip()

builder = ORDatasetBuilder(preprocessor=preprocessor, config=cfg)
df_or = builder.build(ie, pt_or, bo)

print(f"\n[4] ORDatasetBuilder OK")
print(f"    Dataset OR-level: {df_or.shape}")
print(f"    Colonnes: {list(df_or.columns[:12])}")
print(f"    OR avec IE: {df_or.get('has_ie', pd.Series(False)).sum()}")
print(f"    OR avec BO: {df_or.get('has_bo', pd.Series(False)).sum()}")
if "technicien_principal_nom" in df_or.columns:
    print(f"    Tech principal (top 5): {df_or['technicien_principal_nom'].value_counts().head(5).to_dict()}")
if "equipe_principale" in df_or.columns:
    print(f"    Équipes (top 5): {df_or['equipe_principale'].value_counts().head(5).to_dict()}")
if "nb_techniciens" in df_or.columns:
    print(f"    OR multi-tech (>1 tech): {(df_or['nb_techniciens'] > 1).sum()}")

# -----------------------------------------------------------------------
# 5. Feature Engineering
# -----------------------------------------------------------------------
from pointage_analyzer.pipeline.feature_engineering import FeatureEngineer
fe = FeatureEngineer()
df_or_fe = fe.build_features(df_or)
new_cols = [c for c in df_or_fe.columns if c not in df_or.columns]
print(f"\n[5] FeatureEngineer OK — {len(new_cols)} nouvelles features: {new_cols[:8]}")

# -----------------------------------------------------------------------
# 6. Rule Engine
# -----------------------------------------------------------------------
from pointage_analyzer.engine.rule_engine import RuleEngine
rule_engine = RuleEngine(config=cfg)
df_or_rules = rule_engine.apply(df_or_fe)

rule_score_cols = [c for c in df_or_rules.columns if c.startswith("rule_score")]
print(f"\n[6] RuleEngine OK")
print(f"    Colonnes scores: {rule_score_cols}")
if "rule_score_total" in df_or_rules.columns:
    print(f"    Score règle moyen: {df_or_rules['rule_score_total'].mean():.3f}")
    print(f"    OR avec anomalie règle (>0.2): {(df_or_rules['rule_score_total'] > 0.2).sum()}")
if "rule_anomaly_types" in df_or_rules.columns:
    non_empty = df_or_rules["rule_anomaly_types"][df_or_rules["rule_anomaly_types"] != ""]
    print(f"    Exemples anomalies: {list(non_empty.head(3))}")

# -----------------------------------------------------------------------
# 7. Unicité finale
# -----------------------------------------------------------------------
assert not df_or_rules["or_id"].duplicated().any(), "ERREUR: doublons or_id!"
print(f"\n[7] Assertion 1 ligne = 1 OR: OK ({len(df_or_rules)} OR uniques)")

print()
print("=" * 60)
print("SMOKE TEST PASSED — tous les composants fonctionnels")
print("=" * 60)
