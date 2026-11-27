# Modeling Decisions

This note records the main choices made for the indoor PM₂.₅ forecasting pipeline and why they were made.

## 1. Problem and target

- **Task**: Predict indoor PM₂.₅ a short time ahead from household sensor streams.
- **Target**: Indoor PM₂.₅ at the end of a fixed 10-minute bin (`actual`).
- **Lead time**: Measured from the issue time to the end of the target bin (10, 30, or 60 minutes).
- **Why**: This matches how a real-time alert would work—when the forecast is issued, the target interval has not yet finished.

## 2. Data resolution

- Raw data are minute-level.
- We resample to a regular **10-minute grid**.
- A bin is kept only if it has at least 5 valid indoor-PM minutes.
- **Why**: 10-minute bins balance noise reduction with the ability to capture short spikes. The same grid is reused for longer lead times so results stay comparable.

## 3. Train / test split

- **Household holdout**: 13 homes are held out completely; the remaining 55 are development homes.
- Holdout IDs: 007, 008, 013, 017, 025, 037, 040, 059, 101, 102, 116, 122, 127.
- All feature selection, hyperparameter tuning, and cross-validation are done on development homes only.
- **Why**: A pure time-based split can still leak household-specific patterns. Holding out entire homes tests whether the model generalizes to unseen houses.

## 4. Features

Predictors used for the main results (19 features):

- Lagged indoor PM: ages 0, 10, 20, 50, 110 minutes
- Calendar: hour sine/cosine, weekend indicator
- Outdoor PM: `PM_MET_OUT`
- Ventilation: ACH, QCFM_Total, QCFM_Exhaust, QCFM_Supply, QCFM_Infiltration
- Environment: indoor/outdoor CO₂, temperature, relative humidity

**Why**:
- Recent indoor PM is the strongest signal (persistence baseline).
- Outdoor PM and ventilation capture infiltration.
- Temperature, humidity, and CO₂ reflect occupancy and activity.
- Calendar features capture daily and weekly patterns.

Measurements from the target interval are never used as predictors.

## 5. Models

| Model | Role |
|-------|------|
| Persistence | Baseline (predict the current value) |
| Linear regression | Simple linear baseline |
| Ridge regression | Linear model with L2 regularization |
| LightGBM | Gradient boosting (main tree model) |
| XGBoost | Second gradient-boosting model |

- Linear and Ridge use median imputation + standard scaling.
- Tree models use median imputation only.
- Hyperparameters are chosen by randomized search inside GroupKFold.
- **Why**: Persistence is the natural baseline for a highly autocorrelated series. Linear models provide an interpretable reference. Tree models can capture non-linear interactions without extensive manual feature engineering.

## 6. Cross-validation

- **GroupKFold by household** (5 folds).
- No household appears in both the training and validation fold of the same split.
- **Why**: Ordinary K-fold would allow the same home’s future data to influence its past, inflating performance. Grouping by household is the appropriate analogue of a time-series split when the unit of generalization is the home.

## 7. Evaluation metrics

- **MAE** (primary): average absolute error in µg/m³.
- **RMSE**: penalizes large errors more heavily.
- **MASE** (Mean Absolute Scaled Error): MAE of the model divided by the MAE of the persistence baseline on the same set.
- **Why these metrics**:
  - MAE is easy to interpret in the original units (µg/m³).
  - RMSE highlights models that make occasional large errors.
  - MASE is scale-free and directly answers “is the model better than persistence?” A value below 1 means the model beats the baseline; a value above 1 means it is worse.
- **Why not R²**: R² measures variance explained relative to the mean. On a strongly autocorrelated series the mean is a weak baseline, so R² can look high even when the model is barely better than predicting the last observed value. MASE avoids this problem by scaling against the appropriate naïve forecast.

## 8. Outlier handling

- Indoor PM is capped at **100 µg/m³** for training and for the main reported metrics.
- Sensitivity was checked at Cap 50, Cap 150, and no cap.
- An uncapped version of the target is kept for onset-detection analysis.
- **Why**: Extreme spikes (sometimes >1000 µg/m³) dominate squared-error metrics and can be sensor artifacts or rare events. Capping stabilizes training while still allowing a separate check on high-concentration onsets. Cap 100 was retained as the main setting because it stabilizes training without removing as much of the high-concentration range as Cap 50.

## 9. Code organization
features.py          # load_bins, make_examples, feature_columns

models.py            # model_specs, select_models, final_predictions

evaluation.py        # score_predictions, paired_household_errors, onset_scores

plots.py             # concentration-stratified MAE and other diagnostics

poster_figure.py     # preprocessing examples, correlation matrix, observed-vs-predicted

Main_File.ipynb      # main entry point


- **Why**: Separating data preparation, model fitting, scoring, and plotting makes the notebook shorter and lets the same functions be reused for different lead times.

## 10. Poster and diagnostic figures

The following helpers are used for the final report and poster:

- `plot_preprocessing_examples` — raw vs 10-minute binned vs capped series for three homes
- `plot_pearson_correlation` — correlation matrix of the 19 predictors plus target
- `plot_observed_vs_predicted` — scatter of predictions versus measurements
- `plot_concentration_stratified_mae` — MAE by concentration band for Persistence, LightGBM, and XGBoost

These are called from the main notebook after model evaluation.

## 11. What is deliberately not done

- No future information is used in features (all lags are strictly past).
- No household identity is used as a feature.
- Holdout homes are never used for tuning or feature selection.
- Spike-specific features that would require knowing the target are excluded.
- No operational alert system or health-outcome analysis is claimed.

These choices keep the evaluation honest and the results comparable across lead times.