import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------------
# PATH SETUP
# -------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "master_event_level_features_monthly_imputed.csv"

OUTPUT_SUMMARY_FILE = DATA_DIR / "master_event_level_filled.csv"

df = pd.read_csv(INPUT_FILE)

max_dist = df['storm_min_distance_km'].max()
df['storm_min_distance_km'] = df['storm_min_distance_km'].fillna(max_dist + 1000)

df = df.drop(columns=['RH2M_max', 'WS2M_max'])

df.to_csv(OUTPUT_SUMMARY_FILE, index=False)


