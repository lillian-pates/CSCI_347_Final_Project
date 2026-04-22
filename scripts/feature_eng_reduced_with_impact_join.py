# ============================================================
# SELECT ONE RAIN VARIABLE + ONE API VARIABLE
# BASED ON CORRELATION WITH IMPACT PROXY
#
# Input:
#   combined_data_1975_2025_all.csv
#   impact_curve_continuous.csv
#
# Output:
#   1) selected_rain_api_summary.csv
#   2) event_level_features_reduced.csv
#
# What this does:
#   - reads the event-level dataset
#   - reads the impact curve lookup file
#   - creates an event-level return period proxy from flow_pct_max
#   - joins the impact curve at the beginning of the file using nearest rp
#   - uses total_impact as the impact proxy
#   - identifies rain-family variables
#   - identifies API-family variables
#   - computes correlation with impact proxy
#   - selects the strongest variable in each family
#   - drops the other rain/API variables
#   - saves a reduced event-level dataset
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
INPUT_FILE = DATA_DIR / "combined_data_1975_2025_all.csv"
IMPACT_CURVE_FILE = DATA_DIR / "impact_curve_continuous.csv"

OUTPUT_REDUCED_FILE = DATA_DIR / "event_level_features_reduced.csv"
OUTPUT_SUMMARY_FILE = DATA_DIR / "selected_rain_api_summary.csv"

# Minimum non-missing observations required to compute correlation
MIN_NON_MISSING = 10

# If True, use absolute correlation to select strongest variable
USE_ABS_CORR = True

# Optional override. Leave as None to use total_impact after join.
IMPACT_COL_MANUAL = None


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

    if family == "rain":
        keep = []
        for c in cols:
            cl = c.lower()
            if "rain" in cl or "tp_" in cl or "precip" in cl:
                keep.append(c)

        exclude_contains = [
            "impact", "severity", "flood", "event_id", "center_date", "rp", "aep"
        ]
        keep = [c for c in keep if not any(x in c.lower() for x in exclude_contains)]
        return keep

    if family == "api":
        keep = [c for c in cols if "api" in c.lower()]
        exclude_contains = [
            "impact", "severity", "flood", "event_id", "center_date", "rp", "aep"
        ]
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
# LOAD DATA + JOIN IMPACT CURVE AT THE BEGINNING
# -----------------------------
df = pd.read_csv(INPUT_FILE)
impact_df = pd.read_csv(IMPACT_CURVE_FILE)

# Build daily return period proxy from the full flow series
flow_col = first_existing(
    df,
    ["flow", "predicted_flow", "forecasted_flow", "lisflood_flow", "flow_pred", "streamflow", "discharge"],
    required=True
)

df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")

# empirical percentile on the full daily dataset
df["flow_percentile"] = df[flow_col].rank(pct=True)

# avoid infinite rp at percentile = 1
df["flow_percentile"] = df["flow_percentile"].clip(lower=1e-6, upper=0.999)

# return period proxy
df["rp"] = 1 / (1 - df["flow_percentile"])

# Prepare impact curve
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
df = df.sort_values("rp").reset_index(drop=True)

# Join on nearest rp instead of center_date.
df = pd.merge_asof(
    df,
    impact_df[[c for c in ["rp", "total_impact", "aep"] if c in impact_df.columns]],
    on="rp",
    direction="nearest"
)

# -----------------------------
# IDENTIFY IMPACT PROXY
# -----------------------------
if IMPACT_COL_MANUAL is not None:
    impact_col = IMPACT_COL_MANUAL
else:
    impact_col = first_existing(
        df,
        ["total_impact", "impact_proxy", "building_exposure", "exposure_proxy", "impact"],
        required=True
    )

if impact_col not in df.columns:
    raise ValueError(f"Impact proxy column '{impact_col}' was not found after the impact join.")

# Ensure numeric
df[impact_col] = pd.to_numeric(df[impact_col], errors="coerce")

# -----------------------------
# IDENTIFY CANDIDATES
# -----------------------------
rain_candidates = get_family_candidates(df.columns, family="rain")
api_candidates = get_family_candidates(df.columns, family="api")

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
    use_abs=USE_ABS_CORR
)

api_summary = compute_corr_table(
    df=df,
    candidates=api_candidates,
    impact_col=impact_col,
    family_name="api",
    min_non_missing=MIN_NON_MISSING,
    use_abs=USE_ABS_CORR
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

print("\nImpact column summary:")
print(df[impact_col].describe())

print("\nTop rain correlations:")
print(rain_summary.head(10))

print("\nTop API correlations:")
print(api_summary.head(10))
