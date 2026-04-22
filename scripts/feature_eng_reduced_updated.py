# ============================================================
# SELECT ONE RAIN VARIABLE + ONE API VARIABLE
# BASED ON CORRELATION WITH IMPACT PROXY
#
# Input:
#   combined_data_1975_2025_all.csv (or another event-level file)
#
# Output:
#   1) selected_rain_api_summary.csv
#   2) event_level_features_reduced.csv
#
# What this does:
#   - reads the event-level dataset
#   - uses a valid impact proxy already present in the dataset
#   - identifies rain-family variables
#   - identifies API-family variables
#   - computes correlation with impact proxy
#   - selects the strongest variable in each family
#   - drops the other rain/API variables
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



def get_family_candidates(columns, family):
    """
    Identify columns that belong to a variable family.
    family = 'rain' or 'api'
    """
    cols = list(columns)

    exclude_contains = [
        "impact", "severity", "flood", "event_id", "center_date"
    ]

    if family == "rain":
        keep = []
        for c in cols:
            cl = c.lower()
            if (
                "rain" in cl
                or "tp_" in cl
                or "precip" in cl
            ):
                keep.append(c)

        keep = [c for c in keep if not any(x in c.lower() for x in exclude_contains)]
        return keep

    if family == "api":
        keep = [c for c in cols if "api" in c.lower()]
        keep = [c for c in keep if not any(x in c.lower() for x in exclude_contains)]
        return keep

    raise ValueError("family must be 'rain' or 'api'")



def compute_corr_table(df, candidates, impact_col, family_name, min_non_missing=10, use_abs=True):
    """
    Compute correlations between each candidate and impact proxy.
    Returns a summary dataframe sorted by strongest correlation.
    """
    rows = []

    for c in candidates:
        pair = df[[c, impact_col]].copy().dropna()

        if len(pair) < min_non_missing:
            rows.append({
                "family": family_name,
                "variable": c,
                "n_non_missing": len(pair),
                "corr": np.nan,
                "abs_corr": np.nan
            })
            continue

        corr = pair[c].corr(pair[impact_col])

        rows.append({
            "family": family_name,
            "variable": c,
            "n_non_missing": len(pair),
            "corr": corr,
            "abs_corr": abs(corr) if pd.notna(corr) else np.nan
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    sort_col = "abs_corr" if use_abs else "corr"
    out = out.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    return out



def select_best_variable(summary_df, use_abs=True):
    """
    Return the strongest variable from a summary table.
    """
    if summary_df.empty:
        return None

    sort_col = "abs_corr" if use_abs else "corr"
    valid = summary_df.dropna(subset=[sort_col]).copy()

    if valid.empty:
        return None

    return valid.iloc[0]["variable"]


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE)

if "center_date" in df.columns:
    df["center_date"] = pd.to_datetime(df["center_date"], errors="coerce")

print("Columns in input file:")
print(df.columns.tolist())

# -----------------------------
# IDENTIFY IMPACT PROXY
# -----------------------------
if IMPACT_COL_MANUAL is not None:
    if IMPACT_COL_MANUAL not in df.columns:
        raise ValueError(
            f"IMPACT_COL_MANUAL='{IMPACT_COL_MANUAL}' was not found in the input file. "
            f"Available columns: {df.columns.tolist()}"
        )
    impact_col = IMPACT_COL_MANUAL
else:
    impact_col = first_existing(
        df,
        [
            "flow_pct_max",
            "flow_raw_max",
            "flood_label_in_window",
            "impact_proxy",
            "building_exposure",
            "exposure_proxy",
            "impact",
        ],
        required=True,
    )

# Ensure numeric
[df.__setitem__(col, pd.to_numeric(df[col], errors="coerce")) for col in [impact_col]]

# -----------------------------
# IDENTIFY CANDIDATES
# -----------------------------
rain_candidates = get_family_candidates(df.columns, family="rain")
api_candidates = get_family_candidates(df.columns, family="api")

if len(rain_candidates) == 0:
    raise ValueError("No rain-family variables were found in the input file.")

if len(api_candidates) == 0:
    raise ValueError("No API-family variables were found in the input file.")

# Force candidates numeric if possible
for c in rain_candidates + api_candidates:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# COMPUTE CORRELATION TABLES
# -----------------------------
rain_summary = compute_corr_table(
    df=df,
    candidates=rain_candidates,
    impact_col=impact_col,
    family_name="rain",
    min_non_missing=MIN_NON_MISSING,
    use_abs=USE_ABS_CORR,
)

api_summary = compute_corr_table(
    df=df,
    candidates=api_candidates,
    impact_col=impact_col,
    family_name="api",
    min_non_missing=MIN_NON_MISSING,
    use_abs=USE_ABS_CORR,
)

# -----------------------------
# SELECT BEST VARIABLE IN EACH FAMILY
# -----------------------------
best_rain = select_best_variable(rain_summary, use_abs=USE_ABS_CORR)
best_api = select_best_variable(api_summary, use_abs=USE_ABS_CORR)

print("Impact proxy used:", impact_col)
print("Best rain variable:", best_rain)
print("Best API variable:", best_api)

# -----------------------------
# SAVE SUMMARY TABLE
# -----------------------------
summary_df = pd.concat([rain_summary, api_summary], ignore_index=True)
summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

# -----------------------------
# BUILD REDUCED DATASET
# Keep:
#   - all non-rain/api columns
#   - the single best rain variable
#   - the single best api variable
# -----------------------------
drop_rain = [c for c in rain_candidates if c != best_rain]
drop_api = [c for c in api_candidates if c != best_api]

drop_cols = sorted(set(drop_rain + drop_api))

reduced_df = df.drop(columns=drop_cols, errors="ignore").copy()
reduced_df.to_csv(OUTPUT_REDUCED_FILE, index=False)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\nSaved:")
print("  Summary table ->", OUTPUT_SUMMARY_FILE)
print("  Reduced dataset ->", OUTPUT_REDUCED_FILE)

print("\nTop rain correlations:")
print(rain_summary.head(10))

print("\nTop API correlations:")
print(api_summary.head(10))
