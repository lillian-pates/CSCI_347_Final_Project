# ============================================================
# BUILD MASTER FULL ROLLING 5-DAY WINDOW DATASET
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "combined_data_1975_2025_all.csv"
IMPACT_CURVE_FILE = DATA_DIR / "impact_curve_continuous.csv"

OUTPUT_FULL_FILE = DATA_DIR / "full_rolling_5day_features.csv"
OUTPUT_MODEL_FILE = DATA_DIR / "full_rolling_5day_features_modeling_only.csv"
OUTPUT_METADATA_FILE = DATA_DIR / "full_rolling_5day_feature_metadata.txt"

OUTPUT_MASTER_FULL_FILE = DATA_DIR / "master_full_rolling_5day_features.csv"
OUTPUT_MASTER_MODEL_FILE = DATA_DIR / "master_full_rolling_5day_features_modeling_only.csv"

WINDOW_BEFORE = 2
WINDOW_AFTER = 2
WINDOW_N_DAYS = WINDOW_BEFORE + WINDOW_AFTER + 1

QSTAR = 0.90
PERCENTILE_CAP = 0.995
RP_CAP = 300

# ============================================================
# HELPERS
# ============================================================

def first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


def pick_cols(df, candidates):
    return [c for c in candidates if c in df.columns]


def make_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def empirical_percentile_safe(series):
    """
    Safer empirical percentile.
    Uses rank / (n + 1), so percentile never equals 1.0.
    """
    s = pd.to_numeric(series, errors="coerce")
    n = s.notna().sum()
    return s.rank(method="average") / (n + 1)


def safe_max(s):
    s = s.dropna()
    return np.nan if s.empty else s.max()


def safe_mean(s):
    s = s.dropna()
    return np.nan if s.empty else s.mean()


def safe_sum(s):
    s = s.dropna()
    return np.nan if s.empty else s.sum()


def safe_min(s):
    s = s.dropna()
    return np.nan if s.empty else s.min()


def find_cols_by_keywords(columns, keywords, exclude_keywords=None):
    if exclude_keywords is None:
        exclude_keywords = []

    keep = []
    for c in columns:
        cl = c.lower()
        if any(k.lower() in cl for k in keywords):
            if not any(x.lower() in cl for x in exclude_keywords):
                keep.append(c)
    return keep


def window_slice(df, date_col, center_date, before=2, after=2):
    start = pd.Timestamp(center_date) - pd.Timedelta(days=before)
    end = pd.Timestamp(center_date) + pd.Timedelta(days=after)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


# ============================================================
# LOAD DAILY DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

date_col = first_existing(df, ["date", "Date", "datetime", "day"], required=True)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df = (
    df.dropna(subset=[date_col])
      .sort_values(date_col)
      .drop_duplicates(subset=[date_col])
      .reset_index(drop=True)
)

# ============================================================
# IDENTIFY CORE COLUMNS
# ============================================================

flood_col = first_existing(
    df,
    ["flood_m2_p2", "flood_m1_p1", "flood_m3_p3", "flood", "is_flood", "flood_event"],
    required=True
)

flow_col = first_existing(
    df,
    ["flow", "predicted_flow", "forecasted_flow", "lisflood_flow", "flow_pred", "streamflow", "discharge"],
    required=True
)

storm_distance_col = first_existing(
    df,
    ["storm_min_distance_km", "storm_distance_min", "min_storm_distance", "storm_distance", "dist_to_storm"],
    required=False
)

# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_candidates = [
    flood_col,
    flow_col,
    "rain_1day", "rain_2day", "rain_3day", "rain_4day", "rain_5day",
    "rain_7day", "rain_14day",
    "tp_6h_intensity_mm_hr", "tp_12h_intensity_mm_hr",
    "tp_6h_intensity", "tp_12h_intensity",
    "tp_6h_max", "tp_12h_max",
    "api_k_0_70", "api_k_0_80", "api_k_0_85", "api_k_0_90",
    "api_k_0_95", "api_k_0_98", "api_k_0_99", "api",
    "storm_min_distance_km", "storm_distance_min", "min_storm_distance",
    "storm_distance", "dist_to_storm",
    "RH2M", "RH2M_max", "WS2M", "WS2M_max",
    "wind", "storm_wind", "max_storm_wind",
    "storm_max_wind_kt",
    "rp", "aep", "total_impact", "impact_proxy", "severity_proxy", "severity"
]

df = make_numeric(df, numeric_candidates)

df[flood_col] = pd.to_numeric(df[flood_col], errors="coerce").fillna(0)
df[flood_col] = (df[flood_col] > 0).astype(int)

if storm_distance_col is not None:
    df["storm_min_distance_km"] = pd.to_numeric(df[storm_distance_col], errors="coerce")
    storm_distance_col = "storm_min_distance_km"

# ============================================================
# DAILY FLOW PERCENTILE / RETURN PERIOD FEATURES
# ============================================================

df["flow_percentile"] = empirical_percentile_safe(df[flow_col])

df["above_90_daily"] = (df["flow_percentile"] > QSTAR).astype(int)

df["flow_exceedance_90_daily"] = np.maximum(
    df["flow_percentile"] - QSTAR,
    0
)

# Cap percentile before converting to return period
df["flow_percentile_capped"] = np.minimum(df["flow_percentile"], PERCENTILE_CAP)

df["flow_return_period_approx"] = 1.0 / (1.0 - df["flow_percentile_capped"])

# Hard cap as extra protection
df["flow_return_period_approx"] = np.minimum(
    df["flow_return_period_approx"],
    RP_CAP
)

print("MAX DAILY RP AFTER CAP:", df["flow_return_period_approx"].max())

# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

rain_cols = pick_cols(
    df,
    [
        "rain_1day", "rain_2day", "rain_3day", "rain_4day", "rain_5day",
        "rain_7day", "rain_14day"
    ]
)

api_cols = pick_cols(
    df,
    [
        "api_k_0_70", "api_k_0_80", "api_k_0_85", "api_k_0_90",
        "api_k_0_95", "api_k_0_98", "api_k_0_99", "api"
    ]
)

rain_intensity_cols = pick_cols(
    df,
    [
        "tp_6h_intensity_mm_hr", "tp_12h_intensity_mm_hr",
        "tp_6h_intensity", "tp_12h_intensity",
        "tp_6h_max", "tp_12h_max"
    ]
)

humidity_cols = find_cols_by_keywords(
    df.columns,
    keywords=["rh", "humidity"],
    exclude_keywords=["flood", "impact", "severity"]
)

wind_cols = find_cols_by_keywords(
    df.columns,
    keywords=["wind", "ws2m"],
    exclude_keywords=["window", "flood", "impact", "severity"]
)

storm_wind_col = first_existing(
    df,
    ["storm_max_wind_kt", "max_storm_wind", "storm_wind", "wind", "WS2M", "WS2M_max"],
    required=False
)

existing_daily_impact_cols = pick_cols(
    df,
    ["impact_proxy", "total_impact", "severity_proxy", "severity", "rp", "aep"]
)

# ============================================================
# ENGINEER ONE 5-DAY WINDOW
# ============================================================

def engineer_window_features(window_df, center_date, window_id):
    row = {
        "window_id": window_id,
        "center_date": pd.Timestamp(center_date),
        "window_start": window_df[date_col].min(),
        "window_end": window_df[date_col].max(),
        "window_n_days": len(window_df),
        "is_flood_event": int(window_df[flood_col].max()),
        "n_flood_days_in_window": int(window_df[flood_col].sum()),
    }

    flood_dates = window_df.loc[window_df[flood_col] == 1, date_col]
    if len(flood_dates) > 0:
        row["days_to_nearest_flood_in_window"] = min(
            abs((pd.Timestamp(d) - pd.Timestamp(center_date)).days)
            for d in flood_dates
        )
    else:
        row["days_to_nearest_flood_in_window"] = np.nan

    row["flow_raw_max"] = safe_max(window_df[flow_col])
    row["flow_raw_mean"] = safe_mean(window_df[flow_col])

    row["flow_pct_max"] = safe_max(window_df["flow_percentile"])
    row["flow_pct_mean"] = safe_mean(window_df["flow_percentile"])

    row["duration_above_90"] = safe_sum(window_df["above_90_daily"])
    row["area_above_90"] = safe_sum(window_df["flow_exceedance_90_daily"])
    row["max_exceedance_90"] = safe_max(window_df["flow_exceedance_90_daily"])

    row["rp_approx_max"] = safe_max(window_df["flow_return_period_approx"])
    row["rp_approx_mean"] = safe_mean(window_df["flow_return_period_approx"])

    if window_df["flow_percentile"].notna().any():
        peak_idx = window_df["flow_percentile"].idxmax()
        peak_date = window_df.loc[peak_idx, date_col]
        row["days_to_peak_flow"] = (pd.Timestamp(peak_date) - pd.Timestamp(center_date)).days
    else:
        row["days_to_peak_flow"] = np.nan

    for c in rain_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    for c in rain_intensity_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    for c in api_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    if storm_distance_col is not None:
        row["storm_min_distance_km"] = safe_min(window_df[storm_distance_col])
        row["storm_within_300km"] = int((window_df[storm_distance_col] <= 300).fillna(False).any())
        row["storm_within_500km"] = int((window_df[storm_distance_col] <= 500).fillna(False).any())

    if storm_wind_col is not None:
        row["storm_wind_max"] = safe_max(window_df[storm_wind_col])
        row["storm_wind_mean"] = safe_mean(window_df[storm_wind_col])

    for c in humidity_cols:
        if c in window_df.columns:
            row[f"{c}_max"] = safe_max(window_df[c])
            row[f"{c}_mean"] = safe_mean(window_df[c])

    for c in wind_cols:
        if c in window_df.columns:
            row[f"{c}_max"] = safe_max(window_df[c])
            row[f"{c}_mean"] = safe_mean(window_df[c])

    for c in existing_daily_impact_cols:
        row[f"daily_{c}_max"] = safe_max(window_df[c])

    return row

# ============================================================
# BUILD FULL ROLLING CENTERED 5-DAY DATASET
# ============================================================

rows = []

valid_center_dates = df[date_col].iloc[WINDOW_BEFORE : len(df) - WINDOW_AFTER].tolist()

for i, center_date in enumerate(valid_center_dates, start=1):
    w = window_slice(df, date_col, center_date, before=WINDOW_BEFORE, after=WINDOW_AFTER)

    if len(w) != WINDOW_N_DAYS:
        continue

    rows.append(engineer_window_features(w, center_date, f"W{i:06d}"))

full_df = pd.DataFrame(rows).sort_values("center_date").reset_index(drop=True)

print("MAX WINDOW RP AFTER CAP:", full_df["rp_approx_max"].max())

# ============================================================
# ADD CONTINUOUS IMPACT PROXY FROM RETURN PERIOD CURVE
# ============================================================

if IMPACT_CURVE_FILE.exists():
    impact_curve = pd.read_csv(IMPACT_CURVE_FILE)

    impact_curve["rp"] = pd.to_numeric(impact_curve["rp"], errors="coerce")
    impact_curve["total_impact"] = pd.to_numeric(impact_curve["total_impact"], errors="coerce")

    impact_curve = (
        impact_curve
        .dropna(subset=["rp", "total_impact"])
        .sort_values("rp")
    )

    full_df["rp_for_impact"] = full_df["rp_approx_max"].clip(
        lower=impact_curve["rp"].min(),
        upper=impact_curve["rp"].max()
    )

    full_df["impact_proxy"] = np.interp(
        full_df["rp_for_impact"],
        impact_curve["rp"],
        impact_curve["total_impact"]
    )

    full_df["impact_proxy_log"] = np.log1p(full_df["impact_proxy"])

else:
    print(f"Warning: Impact curve file not found at {IMPACT_CURVE_FILE}")
    full_df["rp_for_impact"] = np.nan
    full_df["impact_proxy"] = np.nan
    full_df["impact_proxy_log"] = np.nan

# ============================================================
# MODELING / PCA / CLUSTERING FILE
# ============================================================

non_model_cols = [
    "window_id",
    "center_date",
    "window_start",
    "window_end",
    "is_flood_event",
    "n_flood_days_in_window",
    "days_to_nearest_flood_in_window",
    "impact_proxy",
    "impact_proxy_log",
    "rp_for_impact",
    "daily_impact_proxy_max",
    "daily_total_impact_max",
    "daily_severity_proxy_max",
    "daily_severity_max",
    "daily_rp_max",
    "daily_aep_max"
]

non_model_cols = [c for c in non_model_cols if c in full_df.columns]

numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()

model_cols = [
    c for c in numeric_cols
    if c not in non_model_cols
]

missing_frac = full_df[model_cols].isna().mean()
model_cols = [c for c in model_cols if missing_frac[c] <= 0.40]

model_cols = [
    c for c in model_cols
    if full_df[c].nunique(dropna=True) > 1
]

model_df = full_df[
    ["window_id", "center_date", "is_flood_event"] + model_cols
].copy()

for c in model_cols:
    med = model_df[c].median(skipna=True)
    if pd.isna(med):
        med = 0
    model_df[c] = model_df[c].fillna(med)

# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)

# Save both versions to avoid filename confusion
full_df.to_csv(OUTPUT_FULL_FILE, index=False)
model_df.to_csv(OUTPUT_MODEL_FILE, index=False)

full_df.to_csv(OUTPUT_MASTER_FULL_FILE, index=False)
model_df.to_csv(OUTPUT_MASTER_MODEL_FILE, index=False)

with open(OUTPUT_METADATA_FILE, "w") as f:
    f.write("FULL ROLLING 5-DAY WINDOW DATASET\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Impact curve file: {IMPACT_CURVE_FILE}\n")
    f.write(f"Flood column: {flood_col}\n")
    f.write(f"Flow column: {flow_col}\n")
    f.write(f"Window: centered {WINDOW_N_DAYS}-day window\n")
    f.write(f"QSTAR: {QSTAR}\n")
    f.write(f"Percentile cap: {PERCENTILE_CAP}\n")
    f.write(f"RP cap: {RP_CAP}\n\n")
    f.write("Modeling columns:\n")
    for c in model_cols:
        f.write(f"{c}\n")

print("\nDone.")
print(f"Full rolling-window file saved to: {OUTPUT_FULL_FILE}")
print(f"PCA/clustering-ready file saved to: {OUTPUT_MODEL_FILE}")
print(f"Master full file also saved to: {OUTPUT_MASTER_FULL_FILE}")
print(f"Master modeling file also saved to: {OUTPUT_MASTER_MODEL_FILE}")

print("\nDataset summary:")
print(f"Total complete 5-day windows: {len(full_df)}")
print(f"Flood windows: {int(full_df['is_flood_event'].sum())}")
print(f"Non-flood windows: {int((full_df['is_flood_event'] == 0).sum())}")
print(f"Modeling feature count: {len(model_cols)}")

print("\nRP summary:")
print(full_df["rp_approx_max"].describe())

if "impact_proxy" in full_df.columns:
    print("\nImpact proxy summary by flood label:")
    print(full_df.groupby("is_flood_event")["impact_proxy"].describe())