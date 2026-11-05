from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


# ---------- Dementia CSV ----------
def _print_dementia_summary() -> None:
    csv_path = DATA_DIR / "dementia" / "dementia_patients_health_data.csv"
    df = pd.read_csv(csv_path)
    print("=== DEMENTIA ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nFirst 3 rows:\n", df.head(3).to_string())
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() <= 15:
            print(f"\n{col}:\n", df[col].value_counts())


# ---------- Solomon / G&H ----------
def show_head(path: Path, n: int = 12) -> None:
    print(f"\n=== {path.parent.name}/{path.name} ===")
    lines = path.read_text(encoding="utf-8").splitlines()
    print("Total lines:", len(lines))
    print("\n".join(lines[:n]))


if __name__ == "__main__":
    _print_dementia_summary()
    show_head(DATA_DIR / "Solomon" / "C101.txt")
    show_head(DATA_DIR / "Gehing&Homberger" / "C1_2_1.txt")