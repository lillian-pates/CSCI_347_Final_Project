# ============================================================
# FINAL EVENT-LEVEL CUSTOM FEATURE SELECTION SCRIPT
#
# Input:
#   combined_data_1975_2025_all.csv
#
# Output:
#   1) event_level_features_custom_selected.csv
#   2) custom_feature_selection_summary.csv
#   3) rain_intensity_check.csv
#
# What this does:
#   - reads the full daily dataset
#   - parses dates and identifies a flood label + flow column
#   - builds daily flow percentile features
#   - collapses nearby flood dates into event centers
#   - samples non-flood event centers away from flood dates
#   - builds event-level features using +/- 2 day windows
#   - keeps the event structure:
#       * event_id
#       * center_date
#       * is_flood_event
#       * window_n_days
#       * flood_label_in_window
#   - keeps the variables you requested:
#       * storm_min_distance_km
#       * storm_within_300km
#       * rain_3day
#       * api .9
#       * duration_above_90
#       * area_above_90
#       * humidity variables
#       * wind variables
#   - chooses ONE 6h/12h rain variable based on lowest absolute
#     correlation with rain_3day at the event level
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------------
# PATH SETUP
# -------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "combined_data_1975_2025_all.csv"

OUTPUT_REDUCED_FILE = DATA_DIR / "event_level_features_custom_selected.csv"
OUTPUT_SUMMARY_FILE = DATA_DIR / "custom_feature_selection_summary.csv"
OUTPUT_RAIN_CHECK_FILE = DATA_DIR / "rain_intensity_check.csv"

# -------------------------------
# USER SETTINGS
# -------------------------------
WINDOW_BEFORE = 2
WINDOW_AFTER = 2
MIN_EVENT_GAP = 5
NON_FLOOD_MULTIPLIER = 1.0
MIN_NON_MISSING = 10
SEED = 42


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


def pick_cols(df, candidates):
    return [c for c in candidates if c in df.columns]


def make_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def empirical_percentile(series):
    return series.rank(method="average", pct=True)


def collapse_event_dates(dates, min_gap_days=5):
    if len(dates) == 0:
        return []

    dates = pd.to_datetime(pd.Series(dates)).sort_values().tolist()
    kept = [dates[0]]

    for d in dates[1:]:
        if (d - kept[-1]).days >= min_gap_days:
            kept.append(d)

    return kept


def sample_non_flood_dates(df, date_col, flood_dates, n_needed, min_gap_days=5, seed=42):
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
    start = pd.Timestamp(center_date) - pd.Timedelta(days=before)
    end = pd.Timestamp(center_date) + pd.Timedelta(days=after)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


def safe_max(s):
    return np.nan if s.dropna().empty else s.max()


def safe_mean(s):
    return np.nan if s.dropna().empty else s.mean()


def safe_sum(s):
    return np.nan if s.dropna().empty else s.sum()


def safe_min(s):
    return np.nan if s.dropna().empty else s.min()


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
# LOAD DAILY DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE)

date_col = first_existing(df, ["date", "Date", "datetime", "day"], required=True)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

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

# likely numeric columns
numeric_candidates = [
    flood_col, flow_col,
    "rain_1day", "rain_3day", "rain_5day", "rain_7day", "rain_14day",
    "tp_6h_intensity", "tp_12h_intensity", "tp_6h_max", "tp_12h_max",
    "api_k_0_90", "api_k_0_95", "api", "api_k_0_98", "api_k_0_99",
    "storm_min_distance_km", "min_storm_distance", "storm_distance", "dist_to_storm",
    "storm_wind", "max_storm_wind", "wind", "WS2M", "RH2M"
]
df = make_numeric(df, numeric_candidates)

# binary flood label
df[flood_col] = pd.to_numeric(df[flood_col], errors="coerce").fillna(0)
df[flood_col] = (df[flood_col] > 0).astype(int)

# daily flow percentile features
df["flow_percentile"] = empirical_percentile(pd.to_numeric(df[flow_col], errors="coerce"))
df["above_90_daily"] = (df["flow_percentile"] > 0.90).astype(int)
df["flow_exceedance_90_daily"] = np.maximum(df["flow_percentile"] - 0.90, 0)

# normalize storm distance naming
storm_distance_source = first_existing(
    df,
    ["storm_min_distance_km", "storm_distance_min", "min_storm_distance", "storm_distance", "dist_to_storm"],
    required=False
)
if storm_distance_source is not None:
    df["storm_min_distance_km"] = pd.to_numeric(df[storm_distance_source], errors="coerce")

# normalize api .9 naming
api_09_col = first_existing(
    df,
    ["api_k_0_90", "api_0_90", "api_09", "api.9", "api9"],
    required=True
)

# -----------------------------
# IDENTIFY EVENT CENTERS
# -----------------------------
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
    seed=SEED
)

# -----------------------------
# IDENTIFY HUMIDITY / WIND / RAIN INTENSITY COLUMNS
# -----------------------------
humidity_daily_cols = find_cols_by_keywords(
    df.columns,
    keywords=["rh", "humidity"],
    exclude_keywords=["flood", "impact", "severity"]
)

wind_daily_cols = find_cols_by_keywords(
    df.columns,
    keywords=["wind", "ws2m"],
    exclude_keywords=["window", "flood", "impact", "severity"]
)

# compare exactly these two requested intensity variables
rain_intensity_candidates = pick_cols(
    df,
    ["tp_6h_intensity_mm_hr", "tp_12h_intensity_mm_hr"]
)

df = make_numeric(df, humidity_daily_cols + wind_daily_cols + rain_intensity_candidates)

# -----------------------------
# EVENT FEATURE ENGINEERING
# -----------------------------
def engineer_event_features(window_df, center_date, event_id, is_flood):
    row = {
        "event_id": event_id,
        "center_date": pd.Timestamp(center_date),
        "is_flood_event": int(is_flood),
        "window_n_days": len(window_df),
        "flood_label_in_window": int(window_df[flood_col].max())
    }

    # requested core features
    row["storm_min_distance_km"] = safe_min(window_df["storm_min_distance_km"]) if "storm_min_distance_km" in window_df.columns else np.nan
    row["storm_within_300km"] = int((window_df["storm_min_distance_km"] <= 300).fillna(False).any()) if "storm_min_distance_km" in window_df.columns else np.nan

    row["rain_3day"] = safe_max(window_df["rain_3day"]) if "rain_3day" in window_df.columns else np.nan
    row["api_k_0_90"] = safe_max(window_df[api_09_col]) if api_09_col in window_df.columns else np.nan

    row["duration_above_90"] = safe_sum(window_df["above_90_daily"]) if "above_90_daily" in window_df.columns else np.nan
    row["area_above_90"] = safe_sum(window_df["flow_exceedance_90_daily"]) if "flow_exceedance_90_daily" in window_df.columns else np.nan

    # humidity features
    for c in humidity_daily_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    # wind features
    for c in wind_daily_cols:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    # rain intensity features (6h / 12h candidates)
    for c in rain_intensity_candidates:
        row[f"{c}_max"] = safe_max(window_df[c])
        row[f"{c}_mean"] = safe_mean(window_df[c])

    return row


event_rows = []

# flood events
for i, d in enumerate(flood_dates, start=1):
    w = window_slice(df, date_col, d, before=WINDOW_BEFORE, after=WINDOW_AFTER)
    event_rows.append(engineer_event_features(w, d, f"F{i:03d}", is_flood=1))

# non-flood events
for i, d in enumerate(non_flood_dates, start=1):
    w = window_slice(df, date_col, d, before=WINDOW_BEFORE, after=WINDOW_AFTER)
    event_rows.append(engineer_event_features(w, d, f"N{i:03d}", is_flood=0))

event_df = pd.DataFrame(event_rows).sort_values("center_date").reset_index(drop=True)

# -----------------------------
# CHOOSE ONE OF THE TWO REQUESTED RAIN INTENSITY VARIABLES
# BASED ON LOWEST ABS CORR WITH rain_3day
# -----------------------------
event_rain_intensity_candidates = pick_cols(
    event_df,
    ["tp_6h_intensity_mm_hr_max", "tp_12h_intensity_mm_hr_max"]
)

intensity_rows = []
best_intensity = None
lowest_abs_corr = np.inf

for c in event_rain_intensity_candidates:
    pair = event_df[["rain_3day", c]].dropna()

    if len(pair) < MIN_NON_MISSING:
        intensity_rows.append({
            "candidate": c,
            "n_non_missing": len(pair),
            "corr_with_rain_3day": np.nan,
            "abs_corr_with_rain_3day": np.nan,
            "selected": False
        })
        continue

    corr = pair["rain_3day"].corr(pair[c])
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

intensity_df = pd.DataFrame(intensity_rows)

# -----------------------------
# FINAL KEEP LIST
# -----------------------------
id_vars = [
    "event_id",
    "center_date",
    "is_flood_event",
    "window_n_days",
    "flood_label_in_window"
]

requested_core = [
    "storm_min_distance_km",
    "storm_within_300km",
    "rain_3day",
    "api_k_0_90",
    "duration_above_90",
    "area_above_90"
]

if best_intensity is not None:
    requested_core.append(best_intensity)

humidity_event_cols = [c for c in event_df.columns if any(k in c.lower() for k in ["rh", "humidity"])]
wind_event_cols = [c for c in event_df.columns if ("wind" in c.lower() or "ws2m" in c.lower())]

final_keep = []
for c in id_vars + requested_core + humidity_event_cols + wind_event_cols:
    if c in event_df.columns and c not in final_keep:
        final_keep.append(c)

reduced_df = event_df[final_keep].copy()

# nice date format
reduced_df["center_date"] = pd.to_datetime(reduced_df["center_date"], errors="coerce").dt.strftime("%-m/%-d/%y")

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
reduced_df.to_csv(OUTPUT_REDUCED_FILE, index=False)
intensity_df.to_csv(OUTPUT_RAIN_CHECK_FILE, index=False)

summary_rows = [
    {"item": "input_file", "value": str(INPUT_FILE)},
    {"item": "date_col", "value": date_col},
    {"item": "flood_col", "value": flood_col},
    {"item": "flow_col", "value": flow_col},
    {"item": "api_09_col_used", "value": api_09_col},
    {"item": "n_flood_events", "value": len(flood_dates)},
    {"item": "n_nonflood_events", "value": len(non_flood_dates)},
    {"item": "selected_rain_intensity_from_requested_pair", "value": best_intensity},
    {"item": "selected_rain_intensity_abs_corr_with_rain_3day", "value": lowest_abs_corr},
    {"item": "humidity_event_columns", "value": ", ".join(humidity_event_cols)},
    {"item": "wind_event_columns", "value": ", ".join(wind_event_cols)},
    {"item": "final_output_columns", "value": ", ".join(final_keep)}
]

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("Saved reduced dataset to:", OUTPUT_REDUCED_FILE)
print("Saved summary to:", OUTPUT_SUMMARY_FILE)
print("Saved rain intensity check to:", OUTPUT_RAIN_CHECK_FILE)

print("\nFlood events used:", len(flood_dates))
print("Non-flood events used:", len(non_flood_dates))
print("Selected rain intensity variable:", best_intensity)
print("Absolute correlation with rain_3day:", lowest_abs_corr)

print("\nFinal columns:")
for c in final_keep:
    print(" -", c)
