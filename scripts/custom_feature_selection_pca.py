# ============================================================
# CUSTOM FEATURE SELECTION FOR PCA / MODEL INPUT
#
# Input:
#   combined_data_1975_2025_all.csv
#   impact_curve_continuous.csv
#
# Output:
#   1) custom_feature_selection_summary.csv
#   2) event_level_features_custom_selected.csv
#
# What this does:
#   - reads the full dataset
#   - builds a daily return-period proxy from flow
#   - joins impact curve by nearest rp
#   - keeps the variables you requested:
#       * storm distance min
#       * storm within 300 km
#       * rain_3day
#       * api .9
#       * duration above .9
#       * area above .9
#       * humidity variables
#       * wind variables
#   - chooses ONE rain intensity variable from 6h or 12h family
#     based on the lowest absolute correlation with rain_3day
#   - saves a reduced output file
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

INPUT_FILE = DATA_DIR / "combined_data_1975_2025_all.csv"
IMPACT_CURVE_FILE = DATA_DIR / "impact_curve_continuous.csv"

OUTPUT_REDUCED_FILE = DATA_DIR / "event_level_features_custom_selected.csv"
OUTPUT_SUMMARY_FILE = DATA_DIR / "custom_feature_selection_summary.csv"

MIN_NON_MISSING = 10


# -----------------------------
# HELPERS
# -----------------------------
def first_existing(df, candidates, required=True):
    candidates = [c for c in candidates if c is not None]
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


def make_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE)
impact_df = pd.read_csv(IMPACT_CURVE_FILE)

# -----------------------------
# BUILD DAILY RETURN PERIOD PROXY
# -----------------------------
flow_col = first_existing(
    df,
    ["flow", "predicted_flow", "forecasted_flow", "lisflood_flow", "flow_pred", "streamflow", "discharge"],
    required=True
)

df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")

# empirical percentile over full dataset
df["flow_percentile"] = df[flow_col].rank(pct=True)
df["flow_percentile"] = df["flow_percentile"].clip(lower=1e-6, upper=0.999)
df["rp"] = 1 / (1 - df["flow_percentile"])

# -----------------------------
# JOIN IMPACT CURVE
# -----------------------------
required_impact_cols = ["rp", "total_impact"]
missing_impact_cols = [c for c in required_impact_cols if c not in impact_df.columns]
if missing_impact_cols:
    raise ValueError(
        f"impact_curve_continuous.csv is missing required columns: {missing_impact_cols}"
    )

impact_df["rp"] = pd.to_numeric(impact_df["rp"], errors="coerce")
impact_df["total_impact"] = pd.to_numeric(impact_df["total_impact"], errors="coerce")

if "aep" in impact_df.columns:
    impact_df["aep"] = pd.to_numeric(impact_df["aep"], errors="coerce")

impact_df = impact_df.dropna(subset=["rp", "total_impact"]).sort_values("rp").reset_index(drop=True)
df = df.dropna(subset=["rp"]).sort_values("rp").reset_index(drop=True)

df = pd.merge_asof(
    df,
    impact_df[[c for c in ["rp", "total_impact", "aep"] if c in impact_df.columns]],
    on="rp",
    direction="nearest"
)

# -----------------------------
# ENSURE CORE COLUMNS EXIST / CREATE IF NEEDED
# -----------------------------
# storm distance min
storm_distance_source = first_existing(
    df,
    ["storm_distance_min", "min_storm_distance", "storm_distance", "dist_to_storm"],
    required=False
)
if storm_distance_source is not None:
    df[storm_distance_source] = pd.to_numeric(df[storm_distance_source], errors="coerce")
    if storm_distance_source != "storm_distance_min":
        df["storm_distance_min"] = df[storm_distance_source]

# storm within 300 km
if "storm_within_300km" not in df.columns:
    if "storm_distance_min" in df.columns:
        df["storm_within_300km"] = (df["storm_distance_min"] <= 300).astype("Int64")
    else:
        df["storm_within_300km"] = pd.Series([pd.NA] * len(df), dtype="Int64")

# duration above .9
if "duration_above_90" not in df.columns:
    if "flow_percentile" in df.columns:
        df["duration_above_90"] = (df["flow_percentile"] > 0.90).astype(int)
    else:
        df["duration_above_90"] = np.nan

# area above .9
if "area_above_90" not in df.columns:
    if "flow_percentile" in df.columns:
        df["area_above_90"] = np.maximum(df["flow_percentile"] - 0.90, 0)
    else:
        df["area_above_90"] = np.nan

# rain_3day
rain_3day_col = first_existing(df, ["rain_3day"], required=True)

# api .9
api_09_col = first_existing(
    df,
    ["api_k_0_90", "api_0_90", "api_09", "api.9", "api9"],
    required=True
)

# numeric conversion
core_numeric_candidates = [
    "storm_distance_min", "storm_within_300km", "duration_above_90", "area_above_90",
    rain_3day_col, api_09_col
]
df = make_numeric(df, core_numeric_candidates)

# -----------------------------
# HUMIDITY + WIND COLUMNS
# -----------------------------
humidity_vars = find_cols_by_keywords(
    df.columns,
    keywords=["rh", "humidity"],
    exclude_keywords=["impact", "severity", "flood"]
)

wind_vars = find_cols_by_keywords(
    df.columns,
    keywords=["wind", "ws2m"],
    exclude_keywords=["window", "impact", "severity", "flood"]
)

# -----------------------------
# CHOOSE 6H OR 12H RAIN VARIABLE
# LOWEST ABS CORRELATION WITH rain_3day
# -----------------------------
rain_intensity_candidates = []
for c in df.columns:
    cl = c.lower()
    if ("6h" in cl or "12h" in cl) and ("rain" in cl or "tp_" in cl or "precip" in cl):
        rain_intensity_candidates.append(c)

df = make_numeric(df, rain_intensity_candidates)

best_intensity = None
lowest_abs_corr = np.inf
intensity_rows = []

for c in rain_intensity_candidates:
    pair = df[[rain_3day_col, c]].dropna()

    if len(pair) < MIN_NON_MISSING:
        intensity_rows.append({
            "candidate": c,
            "n_non_missing": len(pair),
            "corr_with_rain_3day": np.nan,
            "abs_corr_with_rain_3day": np.nan,
            "selected": False
        })
        continue

    corr = pair[rain_3day_col].corr(pair[c])
    abs_corr = abs(corr) if pd.notna(corr) else np.nan

    intensity_rows.append({
        "candidate": c,
        "n_non_missing": len(pair),
        "corr_with_rain_3day": corr,
        "abs_corr_with_rain_3day": abs_corr,
        "selected": False
    })

    if pd.notna(abs_corr) and abs_corr < lowest_abs_corr:
        lowest_abs_corr = abs_corr
        best_intensity = c

for row in intensity_rows:
    if row["candidate"] == best_intensity:
        row["selected"] = True

# -----------------------------
# FINAL KEEP LIST
# -----------------------------
id_vars = [c for c in ["event_id", "center_date", "is_flood_event"] if c in df.columns]

requested_core = [
    "storm_distance_min",
    "storm_within_300km",
    rain_3day_col,
    api_09_col,
    "duration_above_90",
    "area_above_90",
]

if best_intensity is not None:
    requested_core.append(best_intensity)

# unique, ordered keep
final_keep = []
for c in id_vars + requested_core + humidity_vars + wind_vars:
    if c in df.columns and c not in final_keep:
        final_keep.append(c)

reduced_df = df[final_keep].copy()
reduced_df.to_csv(OUTPUT_REDUCED_FILE, index=False)

# -----------------------------
# SAVE SUMMARY
# -----------------------------
summary_rows = [
    {"item": "flow_col_used", "value": flow_col},
    {"item": "rain_3day_col_used", "value": rain_3day_col},
    {"item": "api_09_col_used", "value": api_09_col},
    {"item": "selected_rain_intensity", "value": best_intensity},
    {"item": "selected_rain_intensity_abs_corr_with_rain_3day", "value": lowest_abs_corr},
    {"item": "humidity_vars", "value": ", ".join(humidity_vars) if humidity_vars else ""},
    {"item": "wind_vars", "value": ", ".join(wind_vars) if wind_vars else ""},
    {"item": "output_columns", "value": ", ".join(final_keep)},
]

summary_df = pd.DataFrame(summary_rows)
intensity_df = pd.DataFrame(intensity_rows)
excel_path = OUTPUT_SUMMARY_FILE.with_suffix(".xlsx")

summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

intensity_out = OUTPUT_SUMMARY_FILE.with_name("rain_intensity_check.csv")
intensity_df.to_csv(intensity_out, index=False)

# also save csv summary
summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("Saved reduced dataset to:", OUTPUT_REDUCED_FILE)
print("Saved summary csv to:", OUTPUT_SUMMARY_FILE)
print("\nSelected variables:")
for c in final_keep:
    print(" -", c)

print("\nChosen rain intensity variable:", best_intensity)
print("Absolute correlation with rain_3day:", lowest_abs_corr)
