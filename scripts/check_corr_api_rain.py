import pandas as pd


#   - saves a reduced event-level dataset
#
# Notes:
#   - impact_curve_continuous.csv is a lookup/curve file, not an event-level
#     file keyed by center_date, so it should NOT be merged by date here.
#   - default proxy below is flow_pct_max because it is continuous and present
#     in the event-level dataset.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------
# PATH SETUP (REPO-BASED)
# -------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# USER SETTINGS
# -------------------------------
# Change this if you want to point to a different event-level file.
INPUT_FILE = DATA_DIR / "combined_data_1975_2025_all.csv"

OUTPUT_REDUCED_FILE = DATA_DIR / "event_level_features_reduced.csv"
OUTPUT_SUMMARY_FILE = DATA_DIR / "selected_rain_api_summary.csv"

# Minimum non-missing observations required to compute correlation
MIN_NON_MISSING = 10

# If True, use absolute correlation to select strongest variable
USE_ABS_CORR = True

# Manually define impact proxy.
# Recommended defaults for your current event-level dataset:
#   "flow_pct_max"   -> continuous severity-style proxy
#   "flow_raw_max"   -> raw peak flow proxy
#   "flood_label_in_window" -> binary flood label
IMPACT_COL_MANUAL = "flow_pct_max"


# load your data
df = pd.read_csv(INPUT_FILE)

# make sure numeric
df["rain_3day"] = pd.to_numeric(df["rain_3day"], errors="coerce")
df["api_k_0_90"] = pd.to_numeric(df["api_k_0_90"], errors="coerce")  # adjust if your api_0.7 column name differs

# drop missing pairs
pair = df[["rain_3day", "api_k_0_90"]].dropna()

# correlation
corr = pair["rain_3day"].corr(pair["api_k_0_90"])

print("Correlation (rain_3day vs api_k_0_90):", corr)
print("Number of observations:", len(pair))


df["api_k_0_70"] = pd.to_numeric(df["api_k_0_70"], errors="coerce")  # adjust if your api_0.7 column name differs

# drop missing pairs
pair = df[["rain_3day", "api_k_0_70"]].dropna()

# correlation
corr = pair["rain_3day"].corr(pair["api_k_0_70"])

print("Correlation (rain_3day vs api_k_0_70):", corr)
print("Number of observations:", len(pair))

df["rain_5day"] = pd.to_numeric(df["rain_5day"], errors="coerce")
corr = pair["rain_5day"].corr(pair["api_k_0_70"])

print("Correlation (rain_5day vs api_k_0_70):", corr)
print("Number of observations:", len(pair))

