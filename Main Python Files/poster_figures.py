"""
poster_figure.py

Poster-ready figures for the indoor PM2.5 forecasting project.
Call these functions from the main notebook.
"""

from pathlib import Path
import ast
import json
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Shared settings
# =============================================================================

TARGET = "PM_MET_IN1"
PM_CAP = 100
MIN_VALID_MINUTES = 5
BIN_MINUTES = 10

BIN_DURATION = pd.Timedelta(minutes=BIN_MINUTES)
BIN_FREQ = f"{BIN_MINUTES}min"

FEATURE_LABELS = {
    "indoor_pm_age0m": "Indoor PM2.5 (latest)",
    "indoor_pm_age10m": "Indoor PM2.5 (−10 min)",
    "indoor_pm_age20m": "Indoor PM2.5 (−20 min)",
    "indoor_pm_age50m": "Indoor PM2.5 (−50 min)",
    "indoor_pm_age110m": "Indoor PM2.5 (−110 min)",
    "target_hour_sin": "Hour (sin)",
    "target_hour_cos": "Hour (cos)",
    "target_is_weekend": "Weekend",
    "PM_MET_OUT": "Outdoor PM2.5",
    "ACH": "ACH",
    "QCFM_Total": "Total airflow",
    "QCFM_Exhaust": "Exhaust airflow",
    "QCFM_Supply": "Supply airflow",
    "QCFM_Infiltration": "Infiltration airflow",
    "CO2_ETC_IN1": "Indoor CO2",
    "T_ETC_IN1": "Indoor temperature",
    "RH_ETC_IN1": "Indoor RH",
    "T_ETC_OUT": "Outdoor temperature",
    "RH_ETC_OUT": "Outdoor RH",
    "actual": "Indoor PM2.5",
}


# =============================================================================
# 1. Preprocessing examples
# =============================================================================

def _load_pm_data(file, target=TARGET, pm_cap=PM_CAP, min_valid=MIN_VALID_MINUTES):
    df = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(
                file,
                usecols=lambda col: col.strip() in ["Time", target],
                low_memory=False,
                encoding=encoding,
            )
            break
        except Exception:
            continue
    if df is None:
        return None, None

    df.columns = df.columns.str.strip()
    if "Time" not in df.columns or target not in df.columns:
        return None, None

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=["Time"]).set_index("Time").sort_index()

    raw = df[target].replace([np.inf, -np.inf], np.nan).groupby(level=0).mean()
    if raw.dropna().empty:
        return None, None

    minute_data = raw.resample("1min").mean()
    resampled = minute_data.resample(BIN_FREQ, closed="left", label="right")

    bins = pd.DataFrame({
        "uncapped_mean": resampled.mean(),
        "valid_minutes": resampled.count(),
    })
    bins["valid"] = bins["valid_minutes"] >= min_valid
    bins["processed_mean"] = bins["uncapped_mean"].clip(upper=pm_cap).where(bins["valid"])
    return raw, bins


def plot_preprocessing_examples(
    data_dir,
    output_file="pm25_preprocessing_examples.png",
    pm_cap=PM_CAP,
    window_size=6,
):
    """Side-by-side poster figure: concentration capping and minimum data coverage."""
    files = sorted(Path(data_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    cap_example = None
    missing_example = None

    for file in files:
        raw, bins = _load_pm_data(file, pm_cap=pm_cap)
        if raw is None or bins is None or len(bins) < window_size:
            continue

        # Find a capping example
        if cap_example is None:
            hits = np.flatnonzero(
                (bins["valid"] & (bins["uncapped_mean"] > pm_cap)).to_numpy()
            )
            if len(hits):
                pos = hits[0]
                start = max(0, min(pos - 2, len(bins) - window_size))
                cap_example = {
                    "raw": raw,
                    "bins": bins.iloc[start : start + window_size].copy(),
                    "selected_end": bins.index[pos],
                }

        # Find a coverage example
        if missing_example is None:
            valid = bins["valid"].astype(int)
            partial = (
                (bins["valid_minutes"] > 0)
                & (bins["valid_minutes"] < MIN_VALID_MINUTES)
            ).astype(int)
            empty = (bins["valid_minutes"] == 0).astype(int)
            roll_valid = valid.rolling(window_size).sum()
            roll_invalid = (partial + empty).rolling(window_size).sum()
            eligible = (roll_valid >= 1) & (roll_invalid >= 1)
            if eligible.any():
                end_pos = np.flatnonzero(eligible.to_numpy())[0]
                start_pos = end_pos - window_size + 1
                missing_example = {
                    "raw": raw,
                    "bins": bins.iloc[start_pos : end_pos + 1].copy(),
                }

        if cap_example is not None and missing_example is not None:
            break

    if cap_example is None or missing_example is None:
        raise RuntimeError("Could not find suitable preprocessing examples.")

    with plt.rc_context({"font.family": "Arial"}):
        fig, axes = plt.subplots(1, 2, figsize=(19, 8))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.90, wspace=0.22)

        # ----- Left: Concentration capping -----
        ax = axes[0]
        bin_w = cap_example["bins"]
        start = bin_w.index[0] - BIN_DURATION
        end = bin_w.index[-1]
        raw_w = cap_example["raw"][
            (cap_example["raw"].index >= start) & (cap_example["raw"].index < end)
        ]
        minute = raw_w.resample("1min").mean()

        ax.plot(minute.index, minute, color="#98c1d9", lw=1.5, alpha=0.85, zorder=2)
        ax.scatter(
            raw_w.index, raw_w, color="#98c1d9", s=40, alpha=0.85,
            label="Raw data", zorder=3,
        )
        ax.plot(
            bin_w.index,
            bin_w["uncapped_mean"].where(bin_w["valid"]),
            color="#3d5a80", ls="--", marker="o", mfc="white",
            label="Uncapped mean", zorder=4,
        )
        ax.plot(
            bin_w.index, bin_w["processed_mean"],
            color="#293241", marker="s", label="Capped mean", zorder=5,
        )
        ax.axhline(pm_cap, color="#ee6c4d", ls="--", label=f"{pm_cap} µg/m³ cutoff")

        selected = cap_example["selected_end"]
        if selected in bin_w.index:
            val = bin_w.loc[selected, "uncapped_mean"]
            ax.annotate(
                f"{val:.1f} → {pm_cap}",
                xy=(selected, val), xytext=(0, 12),
                textcoords="offset points", ha="center",
                fontsize=18, fontweight="bold", color="#ee6c4d",
            )

        ax.set_title("Concentration capping", fontweight="bold", fontsize=28)
        ax.set_xlabel("Time (HH:MM)", fontweight="bold", fontsize=24)
        ax.set_ylabel(r"Indoor PM$_{\mathbf{2.5}}$ (µg/m³)", fontweight="bold", fontsize=24)
        ax.legend(frameon=False, prop={"size": 16, "weight": "bold"})
        ax.tick_params(labelsize=16)

        # ----- Right: Minimum data coverage -----
        ax = axes[1]
        bin_w = missing_example["bins"]
        start = bin_w.index[0] - BIN_DURATION
        end = bin_w.index[-1]
        raw_w = missing_example["raw"][
            (missing_example["raw"].index >= start) & (missing_example["raw"].index < end)
        ]

        ax.scatter(
            raw_w.index, raw_w, color="#98c1d9", s=40, alpha=0.85,
            label="Raw data", zorder=3,
        )
        ax.plot(
            bin_w.index, bin_w["processed_mean"],
            color="#293241", marker="s", label="Accepted bin", zorder=5,
        )

        labeled_rejected = False
        for bin_end, row in bin_w.iterrows():
            if not row["valid"]:
                ax.axvspan(
                    bin_end - BIN_DURATION, bin_end,
                    color="#ee6c4d", alpha=0.12, zorder=0,
                )
                if row["valid_minutes"] > 0 and not labeled_rejected:
                    ax.scatter(
                        bin_end, row["uncapped_mean"],
                        marker="x", s=160, color="#ee6c4d",
                        label="Rejected mean", zorder=6,
                    )
                    labeled_rejected = True

        ax.set_title("Minimum data coverage", fontweight="bold", fontsize=28)
        ax.set_xlabel("Time (HH:MM)", fontweight="bold", fontsize=24)
        ax.set_ylabel(r"Indoor PM$_{\mathbf{2.5}}$ (µg/m³)", fontweight="bold", fontsize=24)
        ax.legend(frameon=False, prop={"size": 16, "weight": "bold"})
        ax.tick_params(labelsize=16)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()

    print(f"Saved: {output_file}")


# =============================================================================
# 2. Correlation matrix (from a notebook snapshot)
# =============================================================================
def plot_pearson_correlation(
    correlation: pd.DataFrame | None = None,
    output_file: str | Path = "pearson_correlation.png",
    show_values: bool = False,
    title: str = "Feature and target correlations in development homes",
) -> None:
    """
    Pearson correlation heatmap using the verified snapshot or a live DataFrame.
    """
    LABELS = {
        "indoor_pm_age0m": "Indoor PM2.5 (latest)",
        "indoor_pm_age10m": "Indoor PM2.5 (-10 min)",
        "indoor_pm_age20m": "Indoor PM2.5 (-20 min)",
        "indoor_pm_age50m": "Indoor PM2.5 (-50 min)",
        "indoor_pm_age110m": "Indoor PM2.5 (-110 min)",
        "target_hour_sin": "Target hour (sin)",
        "target_hour_cos": "Target hour (cos)",
        "target_is_weekend": "Weekend",
        "PM_MET_OUT": "Outdoor PM2.5",
        "ACH": "Air changes per hour",
        "QCFM_Total": "Total airflow",
        "QCFM_Exhaust": "Exhaust airflow",
        "QCFM_Supply": "Supply airflow",
        "QCFM_Infiltration": "Infiltration airflow",
        "CO2_ETC_IN1": "Indoor CO2",
        "T_ETC_IN1": "Indoor temperature",
        "RH_ETC_IN1": "Indoor relative humidity",
        "T_ETC_OUT": "Outdoor temperature",
        "RH_ETC_OUT": "Outdoor relative humidity",
        "actual": "Indoor PM2.5 (target)",
    }

    if correlation is None:
        # Fallback to the embedded snapshot
        from io import StringIO
        correlation = pd.read_csv(StringIO(
            # paste the full CSV string from the notebook here if you want a pure fallback
        ), index_col=0)

    order = list(correlation.columns)
    labels = [LABELS.get(key, key) for key in order]
    values = correlation.loc[order, order].to_numpy(dtype=float)
    positions = np.arange(len(order))

    with plt.rc_context({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelcolor": "#293241",
        "xtick.color": "#293241",
        "ytick.color": "#293241",
    }):
        fig, ax = plt.subplots(figsize=(12, 10.6), layout="constrained")
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#e6e6e6")
        image = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1,
                          interpolation="nearest", aspect="equal")
        ax.set_xticks(positions, labels, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_yticks(positions, labels)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.set_title(title, color="#293241", fontsize=17, fontweight="bold", pad=13)

        if show_values:
            for row, column in np.ndindex(values.shape):
                r = values[row, column]
                if np.isfinite(r):
                    ax.text(column, row, f"{r:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(r) > 0.65 else "#293241")

        bar = fig.colorbar(image, ax=ax, ticks=np.linspace(-1, 1, 5), shrink=0.89, pad=0.027)
        bar.set_label("Pearson correlation (r)", fontsize=13)
        fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
    print(f"Saved: {output_file}")





