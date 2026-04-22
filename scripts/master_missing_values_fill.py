
import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------------
# PATH SETUP
# -------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "master_event_level_features.csv"


OUTPUT_REDUCED_FILE = DATA_DIR / "master_event_level_features_monthly_imputed.csv"

# --- Load data ---
df = pd.read_csv(INPUT_FILE)

# --- Ensure date is parsed ---
df["center_date"] = pd.to_datetime(df["center_date"], errors="coerce")

# --- Create month column ---
df["month"] = df["center_date"].dt.month

# --- Create NEW columns (do NOT overwrite originals) ---
df["RH2M_max_monthly_med"] = df.groupby("month")["RH2M_max"].transform(
    lambda x: x.fillna(x.median())
)

df["WS2M_max_monthly_med"] = df.groupby("month")["WS2M_max"].transform(
    lambda x: x.fillna(x.median())
)

# --- Optional: sanity check ---
print(df[["RH2M_max", "RH2M_max_monthly_med", "WS2M_max", "WS2M_max_monthly_med"]].isna().sum())

# --- Save new file ---
df.to_csv(OUTPUT_REDUCED_FILE, index=False)

print(f"Saved to: {OUTPUT_REDUCED_FILE}")
