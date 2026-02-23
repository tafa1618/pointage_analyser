import pandas as pd
import warnings
warnings.filterwarnings("ignore")

xl = pd.ExcelFile("data/Classeur3.xlsx", engine="openpyxl")
ie = pd.read_excel(xl, sheet_name="IE", engine="openpyxl")
pt = pd.read_excel(xl, sheet_name="Pointage", engine="openpyxl")
bo = pd.read_excel(xl, sheet_name="BO", engine="openpyxl")

# Is there a numeric OR-looking column in IE?
print("=== IE - toutes colonnes (sample values) ===")
for col in ie.columns:
    sample = ie[col].dropna().head(3).tolist()
    print(f"  {col!r}: {sample}")

print()
print("=== IE - col xxx dtype ===")
print(ie.iloc[:, 0].dtype, ie.iloc[:, 0].head(3).tolist())

# Check all IE columns for OR pattern (large int ~ 8 digits)
print()
print("=== IE - cherche OR (int 8 chiffres) ===")
for col in ie.columns:
    s = ie[col].dropna()
    try:
        nums = pd.to_numeric(s, errors="coerce").dropna()
        if len(nums) > 0 and nums.min() > 10_000_000 and nums.max() < 99_999_999:
            print(f"  CANDIDAT OR: {col!r}  uniques={nums.nunique()}  sample={nums.head(3).tolist()}")
    except Exception:
        pass

# Verify OR (Numéro) in Pointage - is it really 0 for all?
pt_or = pt.iloc[:, 7]
print()
print(f"=== POINTAGE OR key value_counts (top 20) ===")
print(pt_or.value_counts().head(20).to_string())

# BO cross-check
bo_or = bo.iloc[:, 6]
print()
print(f"=== BO N° OR (Segment) value_counts ===")
print(bo_or.value_counts().head(20).to_string())

# Can we join Pointage OR to BO OR?
pt_or_set = set(pt_or.dropna().astype(str).str.strip())
bo_or_set = set(bo_or.dropna().astype(str).str.strip())
intersection = pt_or_set & bo_or_set
print()
print(f"=== JOIN CHECK: Pointage OR ∩ BO OR ===")
print(f"  Pointage unique OR: {len(pt_or_set)}")
print(f"  BO unique OR: {len(bo_or_set)}")
print(f"  Intersection: {len(intersection)}")
print(f"  Samples matching: {list(intersection)[:5]}")
