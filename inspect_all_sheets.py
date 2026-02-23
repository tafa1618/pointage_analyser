import pandas as pd
import warnings
warnings.filterwarnings("ignore")

xl = pd.ExcelFile("data/Classeur3.xlsx", engine="openpyxl")

print("=== TOUTES LES FEUILLES ===")
print(xl.sheet_names)
print()

# Re-read IE avec header auto-detect
for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet, engine="openpyxl")
    print(f"=== FEUILLE: {sheet!r} | Shape: {df.shape} ===")
    for i, c in enumerate(df.columns):
        sample = df.iloc[:, i].dropna().head(2).tolist()
        print(f"  [{i:02d}] {str(c)!r:50s}  => {sample}")
    print()
