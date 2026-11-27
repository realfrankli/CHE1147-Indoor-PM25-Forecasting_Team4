"""Plotting functions for the indoor PM2.5 forecasting report."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------

FIG_WIDTH = 14
TOP_3_HEIGHT = 6
ALL_FEATURES_HEIGHT = None
BORDER_WIDTH = 1.2
TITLE_PADDING = 15

ALL_TITLE = "Feature Ablation Across Models"
TOP_3_TITLE = "Top Features by Ablation"
ALL_OUTPUT_FILE = "feature_ablation_rmse_all_features.png"
TOP_3_OUTPUT_FILE = "feature_ablation_rmse_top_3.png"

MODEL_COLORS = {
    "Linear": "#3d5a80",
    "Ridge": "#98c1d9",
    "LightGBM": "#ee6c4d",
    "XGBoost": "#293241",
}

FEATURE_LABELS = {
    "indoor_pm_age0m": "Indoor PM2.5 (latest)",
    "indoor_pm_age10m": "Indoor PM2.5 (-10 min)",
    "indoor_pm_age20m": "Indoor PM2.5 (-20 min)",
    "indoor_pm_age50m": "Indoor PM2.5 (-50 min)",
    "indoor_pm_age110m": "Indoor PM2.5 (-110 min)",
    "calendar_hour": "Target hour (sin + cos)",
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
}

FONT_SIZES = {
    "feature_labels": 25,
    "x_tick_labels": 25,
    "x_axis_label": 30,
    "legend": 28,
    "title": 35,
}


def compute_ablation(development, holdout, fitted, columns):
    """Leave-one-feature-out ablation on the holdout set."""
    feature_names = list(columns)
    X_dev = development[feature_names]
    y_dev = development["actual"]
    X_holdout = holdout[feature_names]
    y_holdout = holdout["actual"]

    hour_pair = ("target_hour_sin", "target_hour_cos")
    feature_groups = {
        name: [name] for name in feature_names if name not in hour_pair
    }
    hour_columns = [name for name in hour_pair if name in feature_names]
    if hour_columns:
        feature_groups["calendar_hour"] = hour_columns

    records = []
    for model_name in MODEL_COLORS:
        template = fitted[model_name]
        full_model = clone(template).fit(X_dev, y_dev)
        full_rmse = root_mean_squared_error(y_holdout, full_model.predict(X_holdout))

        for feature, removed_columns in feature_groups.items():
            remaining = [n for n in feature_names if n not in removed_columns]
            reduced = clone(template).fit(X_dev[remaining], y_dev)
            reduced_rmse = root_mean_squared_error(
                y_holdout, reduced.predict(X_holdout[remaining])
            )
            records.append({
                "model": model_name,
                "feature": feature,
                "full_rmse": full_rmse,
                "reduced_rmse": reduced_rmse,
                "rmse_change": reduced_rmse - full_rmse,
            })

    ablation = pd.DataFrame(records)
    changes = ablation.pivot(
        index="feature", columns="model", values="rmse_change"
    ).reindex(columns=list(MODEL_COLORS))

    order = changes.sum(axis=1).sort_values(ascending=False, kind="stable").index
    return changes.loc[order]


def plot_ablation(changes, title, height, output_file):
    """Horizontal bar chart of RMSE change by feature and model."""
    labels = [FEATURE_LABELS.get(name, name) for name in changes.index]
    positions = np.arange(len(changes))
    bar_height = 0.19
    offsets = (np.arange(len(MODEL_COLORS)) - (len(MODEL_COLORS) - 1) / 2) * bar_height

    with plt.rc_context({"font.family": "Arial"}):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, height), layout="constrained")

        for offset, (model_name, color) in zip(offsets, MODEL_COLORS.items()):
            ax.barh(
                positions + offset,
                changes[model_name],
                height=bar_height * 0.9,
                color=color,
                label=model_name,
            )

        ax.set_yticks(positions, labels=labels)
        legend_space = max(1.25, 0.12 * len(changes))
        ax.set_ylim(len(changes) - 0.5 + legend_space, -0.5)
        ax.tick_params(axis="y", labelsize=FONT_SIZES["feature_labels"], length=0, pad=8)
        ax.tick_params(axis="x", labelsize=FONT_SIZES["x_tick_labels"])
        ax.axvline(0, color="#293241", linewidth=0.8)

        ax.set_xlabel(
            "Change in holdout RMSE (µg/m³)",
            fontsize=FONT_SIZES["x_axis_label"],
            fontweight="bold",
        )
        ax.set_title(
            title,
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            loc="center",
            pad=TITLE_PADDING,
        )

        ax.legend(
            ncol=2,
            loc="lower right",
            frameon=False,
            prop={"family": "Arial", "size": FONT_SIZES["legend"], "weight": "bold"},
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(BORDER_WIDTH)
            spine.set_color("black")

        ax.grid(False)
        ax.set_facecolor("white")

        fig.savefig(output_file, dpi=300, facecolor="none", edgecolor="none")
        plt.show()


def run_ablation_plots(development, holdout, fitted, columns):
    """Compute ablation and show both plots."""
    changes = compute_ablation(development, holdout, fitted, columns)

    all_height = (
        ALL_FEATURES_HEIGHT
        if ALL_FEATURES_HEIGHT is not None
        else max(TOP_3_HEIGHT, 2.5 + 0.65 * len(changes))
    )

    plot_ablation(changes, ALL_TITLE, all_height, ALL_OUTPUT_FILE)
    plot_ablation(changes.head(3), TOP_3_TITLE, TOP_3_HEIGHT, TOP_3_OUTPUT_FILE)

    return changes


# ---------------------------------------------------------------------------
# RMSE by forecast horizon
# ---------------------------------------------------------------------------

def plot_rmse_by_horizon(rmse_by_horizon, output_file="test_rmse_grouped_bar_chart.png"):
    """Grouped bar chart of test RMSE by model and forecast horizon."""
    FIG_WIDTH = 11
    FIG_HEIGHT = 8
    BAR_WIDTH = 0.23
    BAR_SPACING = 0.02
    BORDER_WIDTH = 1.2
    BAR_EDGE_WIDTH = 0.8
    TITLE_PADDING = 15
    Y_PADDING = 0.22
    Y_LIMITS = None

    FONT_SIZES = {
        "x_tick_labels": 20,
        "y_tick_labels": 20,
        "x_axis_label": 30,
        "y_axis_label": 30,
        "legend": 20,
        "title": 30,
    }

    TITLE = "Model Performance by Forecast Horizon"
    MODELS = {
        "Persistence": "Persistence",
        "Linear": "Linear",
        "Ridge": "Ridge",
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
    }
    HORIZON_STYLES = {
        10: {"label": "10 minutes", "color": "#98c1d9"},
        30: {"label": "30 minutes", "color": "#3d5a80"},
        60: {"label": "60 minutes", "color": "#293241"},
    }

    x = np.arange(len(MODELS))
    n_horizons = len(HORIZON_STYLES)
    offsets = (np.arange(n_horizons) - (n_horizons - 1) / 2) * (BAR_WIDTH + BAR_SPACING)

    y_max = rmse_by_horizon.loc[list(HORIZON_STYLES), list(MODELS)].max().max()

    with plt.rc_context({"font.family": "Arial"}):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained")

        for offset, (horizon, style) in zip(offsets, HORIZON_STYLES.items()):
            ax.bar(
                x + offset,
                rmse_by_horizon.loc[horizon, list(MODELS)],
                width=BAR_WIDTH,
                edgecolor="black",
                linewidth=BAR_EDGE_WIDTH,
                **style,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(list(MODELS.values()))

        if Y_LIMITS is None:
            ax.set_ylim(0, y_max * (1 + Y_PADDING))
        else:
            ax.set_ylim(*Y_LIMITS)

        ax.tick_params(axis="x", labelsize=FONT_SIZES["x_tick_labels"])
        ax.tick_params(axis="y", labelsize=FONT_SIZES["y_tick_labels"])

        ax.set_xlabel("Model", fontsize=FONT_SIZES["x_axis_label"], fontweight="bold")
        ax.set_ylabel("Test RMSE (µg/m³)", fontsize=FONT_SIZES["y_axis_label"], fontweight="bold")

        if TITLE:
            ax.set_title(
                TITLE,
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                loc="center",
                pad=TITLE_PADDING,
            )

        ax.legend(
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            prop={"family": "Arial", "size": FONT_SIZES["legend"], "weight": "bold"},
            frameon=False,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(BORDER_WIDTH)
            spine.set_color("black")

        ax.grid(False)
        ax.set_facecolor("white")
        ax.margins(x=0.03)

        fig.savefig(output_file, dpi=300, facecolor="none", edgecolor="none")
        plt.show()


# ---------------------------------------------------------------------------
# Observed vs predicted (XGBoost)
# ---------------------------------------------------------------------------

def plot_xgboost_observed_vs_predicted(
    by_horizon,
    output_file="xgboost_observed_vs_predicted.png",
    horizons=(10, 60),
):
    """Side-by-side observed vs predicted scatter plots for XGBoost."""
    from pathlib import Path
    from matplotlib.ticker import MultipleLocator

    PANEL_WIDTH = 8
    FIG_HEIGHT = 8
    TICK_INTERVAL = 20
    POINT_SIZE = 12
    POINT_ALPHA = 0.8
    BORDER_WIDTH = 1.2
    REFERENCE_LINE_WIDTH = 1.2
    TITLE_PADDING = 15
    AXIS_PADDING = 0.02

    FONT_SIZES = {
        "x_tick_labels": 30,
        "y_tick_labels": 30,
        "x_axis_label": 35,
        "y_axis_label": 35,
        "title": 35,
    }

    XGBOOST_COLOR = "#293241"
    REFERENCE_COLOR = "#3d5a80"

    plot_data = {
        minutes: by_horizon[minutes][["actual", "XGBoost"]].dropna()
        for minutes in horizons
    }

    lower = min(0, min(frame.min().min() for frame in plot_data.values()))
    upper = max(100, max(frame.max().max() for frame in plot_data.values()))
    padding = AXIS_PADDING * (upper - lower)
    limits = (lower - padding, upper + padding)

    with plt.rc_context({"font.family": "Arial"}):
        fig, axes = plt.subplots(
            1,
            len(horizons),
            figsize=(PANEL_WIDTH * len(horizons), FIG_HEIGHT),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        if len(horizons) == 1:
            axes = [axes]

        for ax, minutes in zip(axes, horizons):
            frame = plot_data[minutes]
            ax.scatter(
                frame["actual"],
                frame["XGBoost"],
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                color=XGBOOST_COLOR,
                linewidths=0,
            )
            ax.plot(
                limits,
                limits,
                linestyle="--",
                color=REFERENCE_COLOR,
                linewidth=REFERENCE_LINE_WIDTH,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(axis="x", labelsize=FONT_SIZES["x_tick_labels"])
            ax.tick_params(axis="y", labelsize=FONT_SIZES["y_tick_labels"])
            ax.set_xlabel(
                r"Observed indoor PM$_{\mathbf{2.5}}$ (µg/m³)",
                fontsize=FONT_SIZES["x_axis_label"],
                fontweight="bold",
            )
            ax.set_title(
                f"{minutes}-minute forecast",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                loc="center",
                pad=TITLE_PADDING,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(BORDER_WIDTH)
                spine.set_color("black")
            ax.grid(False)
            ax.set_facecolor("white")

        axes[0].set_xlim(limits)
        axes[0].set_ylim(limits)
        axes[0].xaxis.set_major_locator(MultipleLocator(TICK_INTERVAL))
        axes[0].yaxis.set_major_locator(MultipleLocator(TICK_INTERVAL))
        axes[0].set_ylabel(
            r"Predicted indoor PM$_{\mathbf{2.5}}$ (µg/m³)",
            fontsize=FONT_SIZES["y_axis_label"],
            fontweight="bold",
        )

        fig.savefig(output_file, dpi=300, facecolor="none", edgecolor="none")
        plt.show()

def plot_concentration_stratified_mae(
    predictions: pd.DataFrame,
    output_file: str | Path = "concentration_stratified_mae.png",
) -> pd.DataFrame:
    """
    Grouped bar chart of MAE by observed indoor PM2.5 concentration band.

    Parameters
    ----------
    predictions : DataFrame
        Must contain columns: actual, persistence, LightGBM, XGBoost
    output_file : str or Path
        Where to save the figure.

    Returns
    -------
    DataFrame with MAE by band and model.
    """
    required = ["actual", "persistence", "LightGBM", "XGBoost"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pred = predictions.copy()
    pred["Persistence"] = pred["persistence"]

    MODELS = ["Persistence", "LightGBM", "XGBoost"]
    BANDS = ["<5", "5–<10", "10–<30", "≥30"]
    COLORS = {
        "Persistence": "#777777",
        "LightGBM": "#EE6C4D",
        "XGBoost": "#293241",
    }
    NAVY = "#293241"

    pred["band"] = pd.cut(
        pred["actual"],
        [-np.inf, 5, 10, 30, np.inf],
        labels=BANDS,
        right=False,
    )

    rows = []
    for band in BANDS:
        group = pred.loc[pred["band"] == band]
        if group.empty:
            raise ValueError(f"No observations in band {band}")
        for model in MODELS:
            rows.append({
                "range": band,
                "model": model,
                "n": len(group),
                "mae": np.mean(np.abs(group["actual"] - group[model])),
            })
    results = pd.DataFrame(rows)

    # Plot
    with plt.rc_context({"font.family": "Arial"}):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13.6, 4.9),
            gridspec_kw={"width_ratios": [2.55, 1]},
        )
        bar_width = 0.25
        low_bands = BANDS[:3]
        counts = results.drop_duplicates("range").set_index("range")["n"]

        for i, model in enumerate(MODELS):
            model_data = results[results["model"] == model].set_index("range")

            # Panel (a)
            values = model_data.loc[low_bands, "mae"].to_numpy()
            bars = ax1.bar(
                np.arange(3) + (i - 1) * bar_width,
                values,
                bar_width,
                color=COLORS[model],
                label=model,
            )
            for bar, value in zip(bars, values):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Panel (b)
            value = model_data.loc["≥30", "mae"]
            x = (i - 1) * bar_width
            ax2.bar(x, value, bar_width, color=COLORS[model])
            ax2.text(
                x, value + 0.12, f"{value:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax1.set_xticks(
            np.arange(3),
            [f"{band}\n(n={int(counts[band]):,})" for band in low_bands],
        )
        ax2.set_xticks([0], [f"≥30\n(n={int(counts['≥30']):,})"])

        low_max = results.loc[results["range"] != "≥30", "mae"].max()
        high_max = results.loc[results["range"] == "≥30", "mae"].max()
        ax1.set_ylim(0, max(1.55, low_max * 1.19))
        ax2.set_ylim(0, high_max * 1.21)

        ax1.set(ylabel="MAE (µg/m³)", title="(a) Low and moderate concentrations")
        ax2.set(title="(b) Elevated concentrations")

        for ax in (ax1, ax2):
            ax.set_xlabel("Observed indoor PM₂.₅ (µg/m³)")
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax1.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
        fig.suptitle(
            "Prediction error by observed concentration",
            fontsize=14, y=1.04, color=NAVY,
        )
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()

    return results