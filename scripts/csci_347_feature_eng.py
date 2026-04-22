# ============================================================
# FEATURE ENGINEERING PIPELINE FOR DATA MINING PROJECT
# Input: combined_data_1975_2025_all.csv
# Output:
#   1) event_level_features.csv
#   2) event_level_features_modeling_only.csv
#
# What this script does:
#   - reads your combined daily file
#   - parses dates and sorts
#   - finds a usable flood label
#   - creates flow_percentile from predicted flow
#   - defines Q* = 90th percentile in percentile space
#   - builds event windows around flood and non-flood dates
#   - engineers flow / rain / API / storm features
#   - keeps evaluation-only variables separate
#
# Notes:
#   - this is written to be robust to slightly different column names
#   - read the "USER SETTINGS" section first
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

# repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# input + output both in /data
INPUT_FILE = DATA_DIR / "combined_data_1975_2025_all.csv"
OUTPUT_DIR = DATA_DIR   # no subfolder


print("INPUT_FILE =", INPUT_FILE)
print("Exists?", INPUT_FILE.exists())
print("Size in bytes:", INPUT_FILE.stat().st_size)


# parameters
WINDOW_BEFORE = 2
WINDOW_AFTER = 2
QSTAR = 0.90
NON_FLOOD_MULTIPLIER = 1.0
MIN_EVENT_GAP = 5

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_existing(df, candidates, required=True):
    """
    Return the first candidate column that exists in df.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


def empirical_percentile(series):
    """
    Empirical percentile in [0,1], robust to ties and missing values.
    """
    s = series.copy()
    out = s.rank(method="average", pct=True)
    return out


def clean_numeric(df, cols):
    """
    Force selected columns to numeric if present.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def choose_flood_label(df):
    """
    Pick one flood label automatically if possible.
    Preference order is based on your prior work.
    """
    preferred = [
        "flood_m2_p2",
        "flood_m1_p1",
        "flood_m3_p3",
        "flood",
        "is_flood",
        "flood_event"
    ]
    existing = [c for c in preferred if c in df.columns]
    if existing:
        return existing[0]

    # fallback: any column containing "flood"
    fuzzy = [c for c in df.columns if "flood" in c.lower()]
    if fuzzy:
        return fuzzy[0]

    raise ValueError("No flood label column found.")


def choose_flow_col(df):
    """
    Pick a predicted flow column.
    """
    preferred = [
        "flow",
        "predicted_flow",
        "forecasted_flow",
        "lisflood_flow",
        "flow_pred",
        "streamflow",
        "discharge"
    ]
    existing = [c for c in preferred if c in df.columns]
    if existing:
        return existing[0]

    fuzzy = [c for c in df.columns if "flow" in c.lower() or "discharge" in c.lower()]
    if fuzzy:
        return fuzzy[0]

    raise ValueError("No flow column found.")


def pick_cols(df, candidates):
    """
    Return existing candidates in the same order.
    """
    return [c for c in candidates if c in df.columns]


def collapse_event_dates(dates, min_gap_days=5):
    """
    Collapse nearby event dates so you do not create many overlapping flood windows.
    Keeps earliest date in each cluster.
    """
    if len(dates) == 0:
        return []

    dates = pd.to_datetime(pd.Series(dates)).sort_values().tolist()
    kept = [dates[0]]
    for d in dates[1:]:
        if (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def sample_non_flood_dates(df, date_col, flood_dates, n_needed, min_gap_days=5, seed=42):
    """
    Sample non-flood dates that are not too close to flood dates.
    """
    rng = np.random.default_rng(seed)
    all_dates = pd.to_datetime(df[date_col]).sort_values().unique()

    flood_dates = pd.to_datetime(pd.Series(flood_dates)).sort_values().tolist()

    eligible = []
    for d in all_dates:
        too_close = any(abs((pd.Timestamp(d) - fd).days) < min_gap_days for fd in flood_dates)
        if not too_close:
            eligible.append(pd.Timestamp(d))

    if len(eligible) == 0:
        return []

    n_needed = min(n_needed, len(eligible))
    chosen = rng.choice(np.array(eligible, dtype="datetime64[ns]"), size=n_needed, replace=False)
    chosen = pd.to_datetime(pd.Series(chosen)).sort_values().tolist()
    return collapse_event_dates(chosen, min_gap_days=min_gap_days)


def window_slice(df, date_col, center_date, before=2, after=2):
    """
    Return the event window around a center date.
    """
    start = pd.Timestamp(center_date) - pd.Timedelta(days=before)
    end   = pd.Timestamp(center_date) + pd.Timedelta(days=after)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


def safe_max(s):
    return np.nan if s.dropna().empty else s.max()

def safe_mean(s):
    return np.nan if s.dropna().empty else s.mean()

def safe_sum(s):
    return np.nan if s.dropna().empty else s.sum()

def safe_min(s):
    return np.nan if s.dropna().empty else s.min()


def standardize_columns(df, cols):
    """
    z-score standardization for modeling columns only.
    """
    out = df.copy()
    for c in cols:
        mu = out[c].mean(skipna=True)
        sd = out[c].std(skipna=True)
        if pd.isna(sd) or sd == 0:
            out[c] = 0.0
        else:
            out[c] = (out[c] - mu) / sd
    return out


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Identify date column
date_col = first_existing(df, ["date", "Date", "datetime", "day"])
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

# Identify main columns
flood_col = choose_flood_label(df)
flow_col = choose_flow_col(df)

# Force core columns numeric if present
candidate_numeric = [
    flow_col, flood_col,
    "rain_1day", "rain_2day", "rain_3day", "rain_4day", "rain_5day", "rain_7day", "rain_14day",
    "tp_1h_intensity", "tp_3h_intensity", "tp_6h_intensity", "tp_12h_intensity", "tp_24h_intensity",
    "tp_1h_max", "tp_3h_max", "tp_6h_max", "tp_12h_max", "tp_24h_max",
    "api", "api_k_0_90", "api_k_0_95", "api_k_0_98", "api_k_0_99",
    "storm_distance", "min_storm_distance", "storm_wind", "max_storm_wind",
    "impact_proxy", "severity_proxy"
]
df = clean_numeric(df, candidate_numeric)

# Make flood binary if not already
df[flood_col] = pd.to_numeric(df[flood_col], errors="coerce").fillna(0)
df[flood_col] = (df[flood_col] > 0).astype(int)

# ============================================================
# CREATE FLOW PERCENTILE + Q* FEATURES
# ============================================================

df["flow_percentile"] = empirical_percentile(df[flow_col])
df["above_qstar"] = (df["flow_percentile"] > QSTAR).astype(int)
df["flow_exceedance_qstar"] = np.maximum(df["flow_percentile"] - QSTAR, 0)

# Optional interpretation variable (not ideal for clustering)
# Avoid infinite values at percentile = 1
eps = 1e-6
df["flow_return_period_approx"] = 1.0 / np.maximum(1.0 - np.minimum(df["flow_percentile"], 1 - eps), eps)

# ============================================================
# CHOOSE SUPPORTING FEATURE COLUMNS
# ============================================================

# Rain: keep only a small set
rain_total_cols = pick_cols(df, ["rain_3day", "rain_5day", "rain_7day", "rain_14day"])
rain_daily_cols = pick_cols(df, ["rain_1day"])
rain_intensity_cols = pick_cols(df, ["tp_6h_intensity", "tp_3h_intensity", "tp_24h_intensity"])
rain_max_cols = pick_cols(df, ["tp_6h_max", "tp_3h_max", "tp_24h_max"])

# API: keep 1-2
api_cols = pick_cols(df, ["api_k_0_90", "api_k_0_95", "api", "api_k_0_98", "api_k_0_99"])

# Storm
storm_distance_col = first_existing(
    df,
    ["min_storm_distance", "storm_distance", "dist_to_storm"],
    required=False
)
storm_wind_col = first_existing(
    df,
    ["max_storm_wind", "storm_wind", "wind", "WS2M"],
    required=False
)

# Evaluation-only
impact_col = first_existing(df, ["impact_proxy", "building_exposure", "exposure_proxy"], required=False)
severity_col = first_existing(df, ["severity_proxy", "severity", "impact_severity"], required=False)

# ============================================================
# IDENTIFY FLOOD EVENTS + NON-FLOOD EVENTS
# ============================================================

flood_dates_raw = df.loc[df[flood_col] == 1, date_col].tolist()
flood_dates = collapse_event_dates(flood_dates_raw, min_gap_days=MIN_EVENT_GAP)

n_flood = len(flood_dates)
n_nonflood = int(round(n_flood * NON_FLOOD_MULTIPLIER))

non_flood_dates = sample_non_flood_dates(
    df=df,
    date_col=date_col,
    flood_dates=flood_dates,
    n_needed=n_nonflood,
    min_gap_days=MIN_EVENT_GAP,
    seed=42
)

# ============================================================
# BUILD EVENT-LEVEL FEATURES
# ============================================================

def engineer_event_features(window_df, center_date, event_id, is_flood):
    row = {
        "event_id": event_id,
        "center_date": pd.Timestamp(center_date),
        "is_flood_event": int(is_flood),
        "window_n_days": len(window_df)
    }

    # ---- flow features ----
    row["flow_raw_max"] = safe_max(window_df[flow_col])
    row["flow_raw_mean"] = safe_mean(window_df[flow_col])
    row["flow_pct_max"] = safe_max(window_df["flow_percentile"])
    row["flow_pct_mean"] = safe_mean(window_df["flow_percentile"])
    row["duration_above_90"] = safe_sum(window_df["above_qstar"])
    row["max_exceedance_90"] = safe_max(window_df["flow_exceedance_qstar"])
    row["area_above_90"] = safe_sum(window_df["flow_exceedance_qstar"])

    # peak timing
    if window_df["flow_percentile"].notna().any():
        peak_idx = window_df["flow_percentile"].idxmax()
        peak_date = window_df.loc[peak_idx, date_col]
        row["days_to_peak_flow"] = (pd.Timestamp(peak_date) - pd.Timestamp(center_date)).days
    else:
        row["days_to_peak_flow"] = np.nan

    # ---- rain features ----
    for c in rain_total_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    for c in rain_daily_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_sum"] = safe_sum(window_df[c])

    for c in rain_intensity_cols:
        row[f"{c}_max"] = safe_max(window_df[c])

    for c in rain_max_cols:
        row[f"{c}_max"] = safe_max(window_df[c])

    # simple shape feature: total daily rain before flow peak
    if "rain_1day" in window_df.columns and window_df["flow_percentile"].notna().any():
        peak_idx = window_df["flow_percentile"].idxmax()
        peak_date = window_df.loc[peak_idx, date_col]
        before_peak = window_df[window_df[date_col] < peak_date]
        row["rain_before_peak"] = safe_sum(before_peak["rain_1day"])
    else:
        row["rain_before_peak"] = np.nan

    # ---- API features ----
    for c in api_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    # ---- storm features ----
    if storm_distance_col is not None:
        row["storm_distance_min"] = safe_min(window_df[storm_distance_col])
        row["storm_within_300km"] = int((window_df[storm_distance_col] <= 300).fillna(False).any())
        row["storm_within_500km"] = int((window_df[storm_distance_col] <= 500).fillna(False).any())

    if storm_wind_col is not None:
        row["storm_wind_max"] = safe_max(window_df[storm_wind_col])
        row["storm_wind_mean"] = safe_mean(window_df[storm_wind_col])

    # ---- event duration proxy ----
    row["event_duration_proxy"] = safe_sum(window_df["above_qstar"])

    # ---- evaluation-only variables ----
    if impact_col is not None:
        row["impact_proxy"] = safe_max(window_df[impact_col])

    if severity_col is not None:
        row["severity_proxy"] = safe_max(window_df[severity_col])

    # keep original label info for evaluation
    row["flood_label_in_window"] = int(window_df[flood_col].max())

    return row


event_rows = []
event_id_counter = 1

# Flood events
for d in flood_dates:
    w = window_slice(df, date_col, d, before=WINDOW_BEFORE, after=WINDOW_AFTER)
    event_rows.append(engineer_event_features(w, d, f"F{event_id_counter:03d}", is_flood=1))
    event_id_counter += 1

# Non-flood events
nonflood_counter = 1
for d in non_flood_dates:
    w = window_slice(df, date_col, d, before=WINDOW_BEFORE, after=WINDOW_AFTER)
    event_rows.append(engineer_event_features(w, d, f"N{nonflood_counter:03d}", is_flood=0))
    nonflood_counter += 1

event_df = pd.DataFrame(event_rows).sort_values("center_date").reset_index(drop=True)

# ============================================================
# FINAL FEATURE SELECTION
# Keep modeling features separate from evaluation-only columns
# ============================================================

# columns not for modeling
non_model_cols = [
    "event_id",
    "center_date",
    "is_flood_event",
    "flood_label_in_window",
    "impact_proxy",
    "severity_proxy"
]
non_model_cols = [c for c in non_model_cols if c in event_df.columns]

# Start from all numeric engineered columns
numeric_cols = event_df.select_dtypes(include=[np.number]).columns.tolist()

# remove evaluation-only and identifiers
model_cols = [c for c in numeric_cols if c not in [
    "is_flood_event",
    "flood_label_in_window",
    "impact_proxy",
    "severity_proxy"
]]

# Optional pruning: drop columns with too many missing values
missing_frac = event_df[model_cols].isna().mean()
model_cols = [c for c in model_cols if missing_frac[c] <= 0.40]

# Optional pruning: drop near-constant columns
keep_model_cols = []
for c in model_cols:
    nunique = event_df[c].nunique(dropna=True)
    if nunique > 1:
        keep_model_cols.append(c)
model_cols = keep_model_cols

# Optional pruning: remove highly correlated duplicates
corr_df = event_df[model_cols].copy()
corr_df = corr_df.fillna(corr_df.median(numeric_only=True))

corr = corr_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = []
for c in upper.columns:
    if any(upper[c] > 0.95):
        to_drop.append(c)

model_cols = [c for c in model_cols if c not in to_drop]

# ============================================================
# IMPUTE + STANDARDIZE MODELING FEATURES
# ============================================================

model_df = event_df[["event_id", "center_date"] + [c for c in non_model_cols if c not in ["event_id", "center_date"]] + model_cols].copy()

# Median impute modeling columns only
for c in model_cols:
    med = model_df[c].median(skipna=True)
    if pd.isna(med):
        med = 0
    model_df[c] = model_df[c].fillna(med)

# Standardized version for PCA / clustering
model_df_std = model_df.copy()
model_df_std[model_cols] = standardize_columns(model_df_std[model_cols], model_cols)

# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

daily_out = os.path.join(OUTPUT_DIR, "daily_with_flow_percentile.csv")
event_out = os.path.join(OUTPUT_DIR, "event_level_features.csv")
model_out = os.path.join(OUTPUT_DIR, "event_level_features_modeling_only.csv")
metadata_out = os.path.join(OUTPUT_DIR, "feature_metadata.txt")

df.to_csv(daily_out, index=False)
event_df.to_csv(event_out, index=False)
model_df_std.to_csv(model_out, index=False)

with open(metadata_out, "w") as f:
    f.write("INPUT FILE\n")
    f.write(f"{INPUT_FILE}\n\n")

    f.write("DATE COLUMN\n")
    f.write(f"{date_col}\n\n")

    f.write("FLOOD LABEL USED\n")
    f.write(f"{flood_col}\n\n")

    f.write("FLOW COLUMN USED\n")
    f.write(f"{flow_col}\n\n")

    f.write("QSTAR\n")
    f.write(f"{QSTAR}\n\n")

    f.write("RAIN TOTAL COLS\n")
    f.write(f"{rain_total_cols}\n\n")

    f.write("RAIN DAILY COLS\n")
    f.write(f"{rain_daily_cols}\n\n")

    f.write("RAIN INTENSITY COLS\n")
    f.write(f"{rain_intensity_cols}\n\n")

    f.write("API COLS\n")
    f.write(f"{api_cols}\n\n")

    f.write("STORM DISTANCE COL\n")
    f.write(f"{storm_distance_col}\n\n")

    f.write("STORM WIND COL\n")
    f.write(f"{storm_wind_col}\n\n")

    f.write("IMPACT COL\n")
    f.write(f"{impact_col}\n\n")

    f.write("SEVERITY COL\n")
    f.write(f"{severity_col}\n\n")

    f.write("MODELING COLS\n")
    for c in model_cols:
        f.write(f"{c}\n")

print("Done.")
print(f"Daily file saved to: {daily_out}")
print(f"Event-level feature file saved to: {event_out}")
print(f"Model-ready standardized file saved to: {model_out}")
print(f"Metadata saved to: {metadata_out}")
print(f"\nFlood events used: {len(flood_dates)}")
print(f"Non-flood events used: {len(non_flood_dates)}")
print(f"Modeling feature count: {len(model_cols)}")
