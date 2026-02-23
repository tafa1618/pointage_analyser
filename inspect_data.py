import pandas as pd
import warnings
warnings.filterwarnings("ignore")

xl = pd.ExcelFile("data/Classeur3.xlsx", engine="openpyxl")
ie = pd.read_excel(xl, sheet_name="IE", engine="openpyxl")
pt = pd.read_excel(xl, sheet_name="Pointage", engine="openpyxl")
bo = pd.read_excel(xl, sheet_name="BO", engine="openpyxl")

print("=" * 60)
print("IE — colonnes completes")
print("=" * 60)
for i, c in enumerate(ie.columns):
    print(f"  [{i}] {c!r}")
print(f"  Shape: {ie.shape}")
print()

print("=" * 60)
print("IE — col [0] (xxx) = OR key ?")
print("=" * 60)
print(ie.iloc[:, 0].head(10).tolist())
print()

print("=" * 60)
print("IE — Position valeurs uniques")
print("=" * 60)
pos = ie["Position"]
print(pos.value_counts(dropna=False).to_string())
print()

print("=" * 60)
print("POINTAGE — colonnes completes")
print("=" * 60)
for i, c in enumerate(pt.columns):
    print(f"  [{i}] {c!r}")
print(f"  Shape: {pt.shape}")
print()

print("=" * 60)
print("POINTAGE — col OR (index 7) : OR key")
print("=" * 60)
or_col = pt.iloc[:, 7]
print(f"  Nom: {or_col.name!r}")
print(f"  Uniques: {or_col.nunique()}  |  NaN: {or_col.isna().sum()}")
print(f"  Samples: {or_col.dropna().head(10).tolist()}")
print()

print("=" * 60)
print("POINTAGE — Technicien / Equipe / Heures sample")
print("=" * 60)
tech_cols = [pt.columns[0], pt.columns[2], pt.columns[3], pt.columns[4],
             pt.columns[6], pt.columns[20], pt.columns[21]]
print(pt[tech_cols].head(8).to_string())
print()

print("=" * 60)
print("BO — colonnes completes")
print("=" * 60)
for i, c in enumerate(bo.columns):
    print(f"  [{i}] {c!r}")
print(f"  Shape: {bo.shape}")
print()

print("=" * 60)
print("BO — col N° OR (index 6)")
print("=" * 60)
bo_or = bo.iloc[:, 6]
print(f"  Nom: {bo_or.name!r}")
print(f"  Uniques: {bo_or.nunique()}  |  NaN: {bo_or.isna().sum()}")
print(f"  Samples: {bo_or.dropna().head(10).tolist()}")
print()

print("=" * 60)
print("BO — colonnes financieres (index 30..44)")
print("=" * 60)
fin_cols = list(bo.columns[30:])
print(bo[fin_cols].describe().to_string())
