"""Model definitions and hyperparameter search spaces."""

from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEED = 42


def _pipeline(estimator, scale=False):
    """Build a scikit-learn pipeline with median imputation and optional scaling."""
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def model_specs():
    """Return model name -> (pipeline, hyperparameter grid).

    Tree-model grids are kept for reference; Optuna is used in select_models.
    """
    return {
        "Linear": (_pipeline(LinearRegression(), scale=True), {}),
        "Ridge": (
            _pipeline(Ridge(), scale=True),
            {"model__alpha": [0.01, 0.1, 1, 10, 100]},
        ),
        "LightGBM": (
            _pipeline(LGBMRegressor(random_state=SEED, n_jobs=1, verbosity=-1)),
            {},  # tuned with Optuna
        ),
        "XGBoost": (
            _pipeline(XGBRegressor(
                random_state=SEED, n_jobs=1, verbosity=0, objective="reg:squarederror"
            )),
            {},  # tuned with Optuna
        ),
    }