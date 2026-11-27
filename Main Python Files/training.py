"""Model selection and prediction generation using Optuna for tree models."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_score

import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Use absolute import so the file works both as a script and as a module
from models import SEED, model_specs, _pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _group_cv_mae(estimator, X, y, folds):
    """Mean absolute error under the given GroupKFold splits."""
    scores = []
    for train_idx, valid_idx in folds:
        fit = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fit.predict(X.iloc[valid_idx])
        scores.append(mean_absolute_error(y.iloc[valid_idx], pred))
    return float(np.mean(scores))


def _tune_lightgbm(X, y, folds, n_trials=10):
    """Optuna search for LightGBM. Returns best params and CV MAE."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        }
        model = _pipeline(
            LGBMRegressor(**params, random_state=SEED, n_jobs=1, verbosity=-1)
        )
        return _group_cv_mae(model, X, y, folds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _tune_xgboost(X, y, folds, n_trials=10):
    """Optuna search for XGBoost. Returns best params and CV MAE."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        }
        model = _pipeline(
            XGBRegressor(
                **params,
                random_state=SEED,
                n_jobs=1,
                verbosity=0,
                objective="reg:squarederror",
            )
        )
        return _group_cv_mae(model, X, y, folds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def select_models(development, columns, n_splits=5, n_trials=10):
    """Tune models on development homes using GroupKFold and return fitted estimators.

    Linear and Ridge use the grids in model_specs().
    LightGBM and XGBoost are tuned with Optuna.

    Parameters
    ----------
    development : pd.DataFrame
        Training examples (development households only).
    columns : list of str
        Feature columns to use.
    n_splits : int
        Number of GroupKFold splits.
    n_trials : int
        Number of Optuna trials for tree models.

    Returns
    -------
    dict, pd.DataFrame
        Fitted models and a summary table of cross-validation MAE.
    """
    X = development[columns]
    y = development["actual"]
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, development["household_id"]))
    baseline = development["persistence"]
    baseline_mae = np.mean([
        mean_absolute_error(y.iloc[valid], baseline.iloc[valid]) for _, valid in folds
    ])
    rows = [{"model": "Persistence", "cv_mae": baseline_mae, "parameters": {}}]
    fitted = {}

    specs = model_specs()

    # Linear (no search)
    linear_est, _ = specs["Linear"]
    linear_mae = -cross_val_score(
        linear_est, X, y, scoring="neg_mean_absolute_error", cv=folds
    ).mean()
    fitted["Linear"] = clone(linear_est).fit(X, y)
    rows.append({"model": "Linear", "cv_mae": float(linear_mae), "parameters": {}})

    # Ridge (small Optuna search over alpha)
    def ridge_objective(trial):
        alpha = trial.suggest_categorical("model__alpha", [0.01, 0.1, 1, 10, 100])
        est = clone(specs["Ridge"][0]).set_params(model__alpha=alpha)
        return _group_cv_mae(est, X, y, folds)

    ridge_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    ridge_study.optimize(ridge_objective, n_trials=5, show_progress_bar=False)
    best_alpha = ridge_study.best_params["model__alpha"]
    ridge_est = clone(specs["Ridge"][0]).set_params(model__alpha=best_alpha)
    fitted["Ridge"] = ridge_est.fit(X, y)
    rows.append({
        "model": "Ridge",
        "cv_mae": float(ridge_study.best_value),
        "parameters": {"model__alpha": best_alpha},
    })

    # LightGBM with Optuna
    print(f"Tuning LightGBM with Optuna ({n_trials} trials)...")
    lgbm_params, lgbm_mae = _tune_lightgbm(X, y, folds, n_trials=n_trials)
    lgbm_model = _pipeline(
        LGBMRegressor(**lgbm_params, random_state=SEED, n_jobs=1, verbosity=-1)
    )
    fitted["LightGBM"] = lgbm_model.fit(X, y)
    rows.append({
        "model": "LightGBM",
        "cv_mae": float(lgbm_mae),
        "parameters": lgbm_params,
    })

    # XGBoost with Optuna
    print(f"Tuning XGBoost with Optuna ({n_trials} trials)...")
    xgb_params, xgb_mae = _tune_xgboost(X, y, folds, n_trials=n_trials)
    xgb_model = _pipeline(
        XGBRegressor(
            **xgb_params,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
            objective="reg:squarederror",
        )
    )
    fitted["XGBoost"] = xgb_model.fit(X, y)
    rows.append({
        "model": "XGBoost",
        "cv_mae": float(xgb_mae),
        "parameters": xgb_params,
    })

    return fitted, pd.DataFrame(rows).sort_values("cv_mae")


def development_predictions(examples, estimator, columns, n_splits=5):
    """Generate out-of-fold predictions on the development set."""
    X = examples[columns]
    y = examples["actual"]
    groups = examples["household_id"]
    output = pd.Series(index=X.index, dtype=float)
    for train, valid in GroupKFold(n_splits=n_splits).split(X, y, groups):
        fit = clone(estimator).fit(X.iloc[train], y.iloc[train])
        output.iloc[valid] = fit.predict(X.iloc[valid])
    return output


def final_predictions(examples, fitted, columns):
    """Apply fitted models to a holdout set and return a prediction table."""
    result = examples[[
        "household_id", "issue_time", "target_start", "target_end",
        "actual", "uncapped_actual", "previous_target_actual", "persistence"
    ]].copy()
    for name, estimator in fitted.items():
        result[name] = estimator.predict(examples[columns])
    return result