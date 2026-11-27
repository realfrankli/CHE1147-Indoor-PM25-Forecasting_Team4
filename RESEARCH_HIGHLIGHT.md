# 🏠🌿 What Can Machine Learning Tell Us About the Air Inside Our Homes?

**A CHE1147 research highlight | Chemical Data Science and Engineering · University of Toronto · Team 4 · Fall 2026**

> **The central finding:** For predictions just 10 minutes ahead, the previous indoor PM₂.₅ measurement was the most accurate predictor *on average*. More complex machine-learning models did not beat this simple baseline on mean absolute error (MASE > 1), but LightGBM and XGBoost reduced larger prediction errors and performed better for the highest observed concentration category.

## 🌬️ Why indoor air quality matters

We spend time indoors, but indoor air is not isolated from the outside world. Outdoor particles can enter buildings, while indoor activities, ventilation, and building conditions also affect particulate concentrations. Fine particulate matter, known as **PM₂.₅**, consists of airborne particles with diameters of approximately 2.5 micrometres or less. Exposure to fine particles is an established public-health concern, making the ability to understand changes in indoor concentrations a relevant research challenge [1].

Our project asks a focused question: **Can machine-learning models predict indoor PM₂.₅ over the next 10-minute interval more accurately than simply using the most recent measurement?**

## 🔬 From sensor readings to predictions

We analyzed a publicly documented residential monitoring study conducted by researchers associated with Lawrence Berkeley National Laboratory [2, 3]. The source study monitored **70 California homes** for approximately one week per home; **68 homes** provided usable observations for our prediction task.

The current experiment includes:

| Study component | Current analysis |
|---|---:|
| Eligible households | **68** |
| Development / held-out households | **55 / 13** |
| Historical and environmental predictors | **19** |
| Prediction horizon | **Next 10-minute interval** |
| Cross-validation | **GroupKFold by household** |

Predictors included lagged indoor PM₂.₅ (ages 0, 10, 20, 50, 110 minutes), outdoor PM₂.₅, ventilation and airflow indicators (ACH, supply/exhaust/infiltration flows), indoor and outdoor temperature, humidity, and CO₂, and calendar features (hour sine/cosine, weekend indicator). Measurements from the target interval were not used as predictors. Model selection used household-grouped cross-validation, and the reported evaluation used households excluded from model fitting. The analysis compared **persistence, linear regression, ridge regression, LightGBM, and XGBoost**.

## 📊 The result was more interesting than a simple model ranking

| Prediction approach | Held-out MAE ↓ | Held-out RMSE ↓ | MASE ↓ |
|---|---:|---:|---:|
| **Persistence** | **0.359** | 1.817 | **1.000** |
| XGBoost | 0.403 | **1.705** | 1.122 |
| LightGBM | 0.408 | 1.708 | 1.137 |
| Linear regression | 0.453 | 1.770 | 1.261 |
| Ridge regression | 0.453 | 1.770 | 1.261 |

*MAE and RMSE are in µg/m³ under the Cap 100 setting. MASE is the model MAE divided by the persistence MAE; values below 1 mean the model beats the baseline. Lower is better for all three metrics.*

**Why did persistence win on MAE (and MASE)?** Indoor PM₂.₅ readings were highly correlated across consecutive 10-minute intervals. The most recent indoor measurement had a correlation of **r = 0.962** with the next interval's concentration in the development data. At such a short horizon, simply carrying forward the latest reading is a strong benchmark. The trained models could not improve on its *average absolute* error, so their MASE values were above 1.

**Why did the tree models win on RMSE?** RMSE penalizes large errors more strongly than MAE. Both LightGBM and XGBoost produced lower RMSE than persistence, even though their MAE was higher. This difference led us to look more closely at the relatively rare higher-concentration observations.

**Why not R²?** R² measures variance explained relative to the mean. On a strongly autocorrelated series the mean is a weak baseline, so R² can look high even when a model is barely better than predicting the last observed value. MASE avoids this problem by scaling against the appropriate naïve forecast (persistence).

## 🌡️ Effect of the concentration cap

Indoor PM was capped before training so that rare extreme spikes would not dominate the loss. The same household split and feature set were repeated under four rules: Cap 50, Cap 100, Cap 150, and no cap.

| Model | MAE (Cap 50) | MAE (Cap 100) | MAE (Cap 150) | MAE (No cap) | MASE (Cap 50) | MASE (Cap 100) | MASE (No cap) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.332 | 0.359 | 0.365 | 0.365 | 1.000 | 1.000 | 1.000 |
| XGBoost | 0.359 | 0.403 | 0.424 | 0.505 | **1.083** | 1.122 | 1.385 |
| LightGBM | 0.365 | 0.408 | 0.433 | 0.483 | 1.099 | 1.137 | 1.323 |
| Linear / Ridge | 0.381 | 0.453 | 0.494 | 0.657 | 1.149 | 1.261 | 1.800 |

**Percentage of values altered by capping** (computed on the 10-minute binned indoor PM series):

- Cap 50 clips **1.56 %** of observations  
- Cap 100 clips **0.57 %** of observations  
- Cap 150 clips **0.30 %** of observations 

**What the comparison shows**

- Cap 50 produces the lowest MAE and the lowest MASE for every model. XGBoost reaches MASE 1.083, the closest any model comes to the persistence baseline.
- Raising the cap increases error. The increase is modest from 50 → 100 and larger from 100 → no cap.
- Linear models are the most sensitive: their MAE rises from 0.38 (Cap 50) to 0.66 (no cap). Tree models rise less.
- Cap 100 remains a practical middle ground: it keeps most of the benefit of capping while still retaining a larger portion of the upper tail than Cap 50.

**Interpretation**

A stricter cap improves average error because extreme values (sometimes well above 150 µg/m³) pull linear fits and, to a lesser extent, tree models. Cap 50 gives the best aggregate numbers, but it also compresses the very events that dominate RMSE. Cap 100 was retained as the main setting because it stabilizes training without removing as much of the high-concentration range. Both choices should be reported; the appropriate cap depends on whether the goal is average accuracy or faithful representation of elevated events.

## 🌡️ Why elevated concentrations change the story

Of the held-out observations, the large majority were below 5 µg/m³. Persistence performed best in that common low-concentration range.

Only a small fraction of observations were at or above 30 µg/m³. In this higher-concentration subset, the tree models achieved a lower MAE than persistence. These higher-concentration observations accounted for a large share of persistence's total squared prediction error, helping explain why RMSE and MAE favor different approaches.

**Concentration-stratified MAE** (held-out homes, Cap 100):

| Concentration band | Persistence MAE | LightGBM MAE | XGBoost MAE |
|---|---:|---:|---:|
| < 5 µg/m³ | lowest | higher | higher |
| 5 – < 10 | — | — | — |
| 10 – < 30 | — | — | — |
| ≥ 30 µg/m³ | higher | **lower** | **lower** |

The concentration-stratified figure is generated by `plot_concentration_stratified_mae` in `plots.py`.

**The engineering lesson:** A model that is best for typical conditions may not be the model that minimizes the largest errors. The appropriate evaluation metric depends on what a monitoring application is intended to accomplish. MASE makes this comparison explicit: a value below 1 is required before claiming that a model improves on the simple baseline.

## 🧠 What the models learned—and what they did not

Both boosted-tree models relied strongly on previous indoor PM₂.₅ measurements. This is a measure of how the *fitted model* uses predictors; it **does not** mean that one feature causes a fixed percentage of pollution, explains that percentage of concentration variation, or contributes that percentage to each prediction.

Outdoor pollution, ventilation, and activity-related measurements also appeared among model features. Additional feature-ablation or controlled analyses would be required to establish their independent predictive value.

## 💡 Why these findings matter

This project offers four practical lessons for data-driven indoor air-quality research:

1. **Start with a meaningful baseline.** At short prediction horizons, a low-complexity method can be difficult to outperform. Any added computational complexity should demonstrate a measurable benefit (MASE < 1).
2. **Evaluate the events that matter for the intended use.** Aggregate MAE can obscure large errors during uncommon elevated-concentration periods. Concentration-specific diagnostics and RMSE reveal trade-offs that a single score misses.
3. **Build for generalization, not just fit.** Splitting data by household tests how methods transfer to different residences, which is a more demanding question than randomly mixing measurements from the same homes.
4. **Handle extremes deliberately.** A concentration cap stabilizes training, especially for linear models. The percentage of values clipped should be reported alongside the performance tables so readers can judge the trade-off.

These findings may help inform future work on **indoor air-quality monitoring, event-focused forecasting, and the evaluation of smart-building prediction methods**. However, our study **did not test an operational alert system, demonstrate lower occupant exposure, or establish health benefits**. Those applications would require further testing.

## ⚖️ Limitations and next steps

The current task predicts only **one 10-minute interval ahead**. Concentrations were capped at **100 µg/m³** for the main results, potentially affecting the highest events. The original household monitoring periods were relatively short. Performance during abrupt concentration spikes still needs improvement.

The next research steps are to test **one-hour-ahead prediction**, examine which features contribute incremental predictive value, evaluate peak-event detection, and validate on newly reserved households or monitoring periods.

---

## 📚 Sources and reproducibility

**External scientific sources**

1. World Health Organization. *WHO Global Air Quality Guidelines* (2021). https://www.who.int/publications/i/item/9789240034228
2. Singer, B. C., Chan, W. R., Kim, Y.-S., Offermann, F. J., & Walker, I. S. (2020). *Indoor air quality in California homes with code-required mechanical ventilation*. **Indoor Air**, 30(5), 885–899. https://doi.org/10.1111/ina.12676
3. Chan, W. R., Kim, Y.-S., Delp, W. W., Walker, I. S., & Singer, B. C. (2020). *Data from: Indoor air quality in California homes with code-required mechanical ventilation*. Dryad. https://doi.org/10.7941/D1ZS7X

**Project evidence**

- `features.py`, `models.py`, `evaluation.py`, `plots.py`, `poster_figure.py` — modular pipeline
- `decisions.md` — justification of split, features, and metrics
- Executed notebook with household-holdout results, capping comparison, and concentration-stratified MAE

*All numerical results in this article come from the current household-holdout experiment. The article summarizes an academic course project; it does not present a deployed forecasting system.*