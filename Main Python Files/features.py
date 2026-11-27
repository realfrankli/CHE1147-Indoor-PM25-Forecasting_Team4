"""Load bins and build forecasting examples."""

import numpy as np
import pandas as pd

from data_loader import load_household_files, resample_data

TARGET = "PM_MET_IN1"
BIN = "10min"
STEP = pd.Timedelta(BIN)
AGES = (0, 10, 20, 50, 110)
HISTORY = [f"indoor_pm_age{age}m" for age in AGES]
SENSORS = {
    "outdoor_pm": ["PM_MET_OUT"],
    "ventilation": ["ACH", "QCFM_Total", "QCFM_Exhaust", "QCFM_Supply", "QCFM_Infiltration"],
    "environment": ["CO2_ETC_IN1", "T_ETC_IN1", "RH_ETC_IN1", "T_ETC_OUT", "RH_ETC_OUT"],
    "activity": ["IBU_CLB", "IBU_CLF", "IBU_CRB", "IBU_CRF", "IBU_CCB", "IBU_OVN"],
    "calendar": ["target_hour_sin", "target_hour_cos", "target_is_weekend"],
}
KEYS = ["household_id", "issue_time"]
METADATA = KEYS + [
    "target_start", "target_end", "actual", "uncapped_actual",
    "previous_target_actual", "persistence",
]


def load_bins(root, min_valid_minutes=5):
    """Aggregate PM by distinct observed minutes and keep each home's regular grid."""
    raw = load_household_files(root)
    indoor = raw[["household_id", "timestamp", TARGET]].copy()
    indoor[TARGET] = pd.to_numeric(indoor[TARGET], errors="coerce")
    indoor = indoor.loc[np.isfinite(indoor[TARGET])].copy()
    indoor["minute"] = indoor["timestamp"].dt.floor("min")
    minute_pm = indoor.groupby(["household_id", "minute"], as_index=False)[TARGET].mean()
    coverage = (
        minute_pm.groupby(["household_id", pd.Grouper(key="minute", freq=BIN)])[TARGET]
        .agg(pm_mean="mean", valid_pm_minutes="count")
        .reset_index()
        .rename(columns={"minute": "timestamp"})
    )
    bins = resample_data(raw, freq=BIN).drop(columns=TARGET, errors="ignore")
    bins = bins.merge(coverage, on=["household_id", "timestamp"], how="left", validate="one_to_one")
    bins["valid_pm_minutes"] = bins["valid_pm_minutes"].fillna(0).astype(int)
    eligible = bins["valid_pm_minutes"].ge(min_valid_minutes) & bins["pm_mean"].ge(0)
    bins[TARGET] = bins["pm_mean"].where(eligible)
    return bins.sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def make_examples(bins, lead_minutes=10, cap=100.0):
    """Measure lead to the end of a fixed 10-minute target interval."""
    if lead_minutes <= 0 or lead_minutes % 10:
        raise ValueError("lead_minutes must be a positive multiple of 10")
    steps = lead_minutes // 10
    frame = bins.sort_values(["household_id", "timestamp"]).copy()
    gaps = frame.groupby("household_id", sort=False)["timestamp"].diff().dropna()
    if not gaps.eq(STEP).all():
        raise ValueError("Pass the full regular grid, including bins without measurements.")
    frame["pm_uncapped"] = frame[TARGET]
    if cap is not None:
        frame[TARGET] = frame[TARGET].clip(upper=cap)
    homes = frame.groupby("household_id", sort=False)
    frame["issue_time"] = frame["timestamp"] + STEP
    frame["target_start"] = homes["timestamp"].shift(-steps)
    frame["target_end"] = frame["target_start"] + STEP
    frame["actual"] = homes[TARGET].shift(-steps)
    frame["uncapped_actual"] = homes["pm_uncapped"].shift(-steps)
    frame["previous_target_actual"] = homes["pm_uncapped"].shift(1 - steps)
    frame["persistence"] = frame[TARGET]
    for age, name in zip(AGES, HISTORY):
        frame[name] = homes[TARGET].shift(age // 10)

    hour = frame["target_start"].dt.hour + frame["target_start"].dt.minute / 60
    frame["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["target_is_weekend"] = frame["target_start"].dt.dayofweek.ge(5).astype(int)

    correct_lead = (frame["target_end"] - frame["issue_time"]).eq(
        pd.Timedelta(minutes=lead_minutes)
    )
    eligible = correct_lead & frame["actual"].notna() & frame["persistence"].notna()
    return frame.loc[eligible].reset_index(drop=True)


def feature_columns(examples, sensor_groups=()):
    selected = HISTORY + [column for group in sensor_groups for column in SENSORS[group]]
    return [
        column for column in dict.fromkeys(selected)
        if column in examples and pd.api.types.is_numeric_dtype(examples[column])
    ]