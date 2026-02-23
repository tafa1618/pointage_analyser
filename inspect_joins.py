import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

xl = pd.ExcelFile("data/Classeur3.xlsx", engine="openpyxl")
ie = pd.read_excel(xl, sheet_name="IE", engine="openpyxl")
pt = pd.read_excel(xl, sheet_name="Pointage", engine="openpyxl")
bo = pd.read_excel(xl, sheet_name="BO", engine="openpyxl")

ie_or  = ie.iloc[:, 0].dropna().astype(str).str.strip()
pt_or  = pt.iloc[:, 7].dropna().astype(str).str.strip()
bo_or  = bo.iloc[:, 6].dropna().astype(str).str.strip()

ie_set = set(ie_or)
pt_set = set(pt_or)
bo_set = set(bo_or)

print("=== VOLUMES ===")
print(f"IE  unique OR: {len(ie_set)}  (total lignes: {len(ie)})")
print(f"Pointage unique OR: {len(pt_set)}  (total lignes: {len(pt)})")
print(f"BO  unique OR: {len(bo_set)}  (total lignes: {len(bo)})")

print()
print("=== INTERSECTIONS ===")
ie_pt  = ie_set & pt_set
ie_bo  = ie_set & bo_set
pt_bo  = pt_set & bo_set
all3   = ie_set & pt_set & bo_set
print(f"IE inter Pointage : {len(ie_pt)} OR en commun")
print(f"IE inter BO       : {len(ie_bo)} OR en commun")
print(f"Pointage inter BO : {len(pt_bo)} OR en commun")
print(f"IE inter Pt et BO : {len(all3)} OR dans les 3")

print()
print("=== ORPHELINS ===")
pt_only = pt_set - ie_set - {"0"}
bo_only = bo_set - ie_set
ie_no_pt_bo = ie_set - pt_set - bo_set
print(f"Pointage OR absents de IE (hors OR=0): {len(pt_only)}")
if pt_only:
    print(f"  Samples: {list(pt_only)[:5]}")
print(f"BO OR absents de IE:                  {len(bo_only)}")
if bo_only:
    print(f"  Samples: {list(bo_only)[:5]}")
print(f"IE OR sans Pointage ni BO:            {len(ie_no_pt_bo)}")

print()
print("=== LIGNES OR=0 dans POINTAGE ===")
zero_lines = (pt.iloc[:, 7] == 0).sum()
print(f"Lignes avec OR=0: {zero_lines} / {len(pt)} ({100*zero_lines/len(pt):.1f}%)")
print("Type heure pour OR=0 (top 10):")
mask_zero = pt.iloc[:, 7] == 0
print(pt.loc[mask_zero, pt.columns[5]].value_counts().head(10).to_string())

print()
print("=== POSITION dans IE ===")
print(ie["Position"].value_counts(dropna=False).to_string())

print()
print("=== EQUIPES Pointage ===")
print(pt.iloc[:, 4].value_counts().to_string())

print()
print("=== BOX: Heures Pointage resume ===")
hr = pt["Hr_Totale"] if "Hr_Totale" in pt.columns else pt.iloc[:, 21]
print(hr.describe().to_string())
print(f"Heures > 12h: {(hr > 12).sum()} lignes")
print(f"Heures > 10h: {(hr > 10).sum()} lignes")

print()
print("=== BO: colonnes financieres cles ===")
bo_fin = bo.iloc[:, [31, 32, 33, 42, 44]]  # Temps prevu, Duree pt, Duree tot, Montant Mo, Total
print(bo_fin.describe().to_string())
