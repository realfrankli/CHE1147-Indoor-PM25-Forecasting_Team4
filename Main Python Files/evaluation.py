"""Score predictions and compute household / onset diagnostics."""

import numpy as np
import pandas as pd


def score_predictions(predictions, model_names, actual_column="actual"):
    """Return MAE, RMSE, and MASE for each model.

    MASE = MAE(model) / MAE(persistence).
    Values below 1 mean the model beats the persistence baseline.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    actual = predictions[actual_column]
    baseline_mae = mean_absolute_error(actual, predictions["persistence"])
    rows = []
    for name in model_names:
        pred = predictions[name]
        mae = mean_absolute_error(actual, pred)
        rows.append({
            "model": "Persistence" if name == "persistence" else name,
            "mae": mae,
            "rmse": np.sqrt(mean_squared_error(actual, pred)),
            "mase": mae / baseline_mae if baseline_mae > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mae")


def paired_household_errors(scored, model_names):
    baseline = (scored["actual"] - scored["persistence"]).abs()
    differences = {
        name: (scored["actual"] - scored[name]).abs() - baseline
        for name in model_names
    }
    return pd.DataFrame(differences).groupby(scored["household_id"]).mean()


def onset_scores(scored, model, threshold=30.0):
    eligible = scored.loc[scored["previous_target_actual"].notna()]
    onset = (
        eligible["previous_target_actual"].lt(threshold)
        & eligible["uncapped_actual"].ge(threshold)
    )
    alarm = eligible["persistence"].lt(threshold) & eligible[model].ge(threshold)
    hits = onset & alarm
    starts = eligible.loc[hits, "target_start"] - eligible.loc[hits, "issue_time"]
    return {
        "observed_onsets": int(onset.sum()),
        "alarms": int(alarm.sum()),
        "recall": hits.sum() / onset.sum() if onset.any() else np.nan,
        "precision": hits.sum() / alarm.sum() if alarm.any() else np.nan,
        "false_alarms_per_100_intervals": (
            100 * (alarm & ~onset).sum() / len(eligible) if len(eligible) else np.nan
        ),
        "median_minutes_to_onset_bin_start": (starts.dt.total_seconds() / 60).median(),
    }
