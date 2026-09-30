from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------- Dementia CSV ----------
csv_path = DATA_DIR / "dementia" / "dementia_patients_health_data.csv"
df = pd.read_csv(r"C:\Users\kriss\OneDrive\Bureau\HHCRSP\data\dementia\dementia_patients_health_data.csv")

print("=== DEMENTIA ===")
print("Shape:", df.shape)  # expected (1000, 24)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nFirst 3 rows:\n", df.head(3).to_string())
for col in df.select_dtypes(include="object").columns:
    if df[col].nunique() <= 15:
        print(f"\n{col}:\n", df[col].value_counts())

# ---------- Solomon / G&H ----------
def show_head(path='', n=12):
    print(f"\n=== {path.parent.name}/{path.name} ===")
    lines = path.read_text().splitlines()
    print("Total lines:", len(lines))
    print("\n".join(lines[:n]))

show_head(DATA_DIR / "solomon" / "C101.txt")
show_head(DATA_DIR / "gehring_homberger" / "C1_2_1.txt")