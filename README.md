<div align="center">

# 🏠🌿 Indoor PM₂.₅ Forecasting with Machine Learning
### CHE1147 · Chemical Data Science and Engineering · University of Toronto
**Team 4 · Fall 2026**

### 📰 Research Highlight

Discover the key findings and potential impact of our CHE1147 project.

**[Read Our Research Article: What Can Machine Learning Tell Us About the Air Inside Our Homes?](RESEARCH_HIGHLIGHT.md)**

![Python](https://img.shields.io/badge/Python-3.11-3572A5)
![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-F37626)
![Project](https://img.shields.io/badge/Study-Residential%20Air%20Quality-0E7490)
![Validation](https://img.shields.io/badge/Evaluation-Grouped%20by%20Home-7041A6)

**[🔎 Overview](#-project-overview)** · **[🗂 Repository map](#-repository-map)** · **[🧪 Modeling workflow](#-modeling-workflow)** · **[📊 Results](#-results)** · **[⚙️ Setup](#%EF%B8%8F-setup--run)**

</div>

---

## 🔎 Project overview

We explore whether supervised regression and gradient-boosted trees can predict **next-interval indoor PM₂.₅ concentrations** in residential homes, compared with a simple persistence baseline. The project uses measurements from the LBNL California residential indoor-air-quality dataset and explores changes over **10-minute intervals**.

> **Research question:** Can machine-learning models improve next-interval indoor PM₂.₅ prediction when evaluated in residential buildings excluded from training?

The repository contains the original team materials plus a cleaned modular pipeline, research highlight, and diagnostic plots. All original raw files are preserved unchanged.

## 🗂 Repository map

| Location | What is inside? |
|---|---|
| 📓 **[`Main_File.ipynb`](Main_File.ipynb)** | **Main entry-point notebook** (open this to run the analysis) |
| 📰 **[`RESEARCH_HIGHLIGHT.md`](RESEARCH_HIGHLIGHT.md)** | Concise findings summary with tables and interpretation |
| 📝 **[`decisions.md`](decisions.md)** | Justification of modeling choices (split, features, capping, metrics) |
| 🐍 **[`Main Python Files/`](Main%20Python%20Files/)** | Modular pipeline scripts (`config.py`, `pipeline.py`, `data_loader.py`, `feature_engineering.py`, tuning scripts, etc.) |
| 🗃️ **[`DATA/`](DATA/)** | Original raw monitoring files (preserved exactly) |
| 🖼️ **[`plots/`](plots/)** or `assets/figures/` | Generated diagnostic figures |
| 🚀 **[`START_HERE.md`](START_HERE.md)** | Clone / upload instructions for teammates |

### 🧭 Quick links

| I want to… | Open |
|---|---|
| **Read the findings** (no code needed) | [`RESEARCH_HIGHLIGHT.md`](RESEARCH_HIGHLIGHT.md) |
| Understand the modeling choices | [`decisions.md`](decisions.md) |
| **Reproduce or modify the analysis** | [`Main_File.ipynb`](Main_File.ipynb) |
| Review raw household monitoring data | [`DATA/`](DATA/) |
| Get clone / setup instructions | [`START_HERE.md`](START_HERE.md) |

---

## 🧪 Modeling workflow

The analysis is organized as a modular pipeline called from `Main_File.ipynb`:

- **Split:** Household hold-out (homes excluded from training) to test generalization to unseen residences.
- **Resolution:** 10-minute bins.
- **Features:** Lagged indoor PM, outdoor PM, ventilation/ACH, temperature/humidity, time features, and household-level statistics.
- **Models:** Persistence baseline, Linear Regression, Ridge, LightGBM, and XGBoost.
- **Capping:** Indoor PM values above a chosen threshold (50 / 100 / 150 µg/m³) are clipped before modeling; the percentage of observations affected is reported in the research highlight.
- **Evaluation:** MAE, RMSE, and R² on held-out households.

You do **not** need to run any code simply to read the results. Open the research highlight and any committed figures.

---

## 📌 Study size (approximate)

| Study attribute | Value |
|---|---:|
| Source households | ~70 |
| Eligible households after filtering | ~64–68 |
| 10-minute observations | ~60k–70k |
| Development / held-out split | Household-level hold-out |
| Predictors | Reduced set (see `decisions.md`) |
| Evaluation | Household hold-out + optional grouped CV |

Exact counts appear in the executed notebook and research highlight.

## 📊 Results

Detailed metrics, capping comparisons, and interpretation are in **[`RESEARCH_HIGHLIGHT.md`](RESEARCH_HIGHLIGHT.md)**.

Typical pattern observed:
- Persistence is a strong baseline at 10-minute resolution because of high temporal autocorrelation.
- Gradient-boosted models (especially LightGBM) often improve RMSE and R² relative to linear models.
- Capping extreme spikes reduces the influence of rare high-PM events; the fraction of points altered is small (see highlight for exact percentages).

### Example figures (if present in `assets/figures/` or `plots/`)

| Figure | Description |
|---|---|
| Model comparison | Held-out MAE / RMSE across models |
| Errors by concentration | Performance stratified by observed PM level |
| Temporal autocorrelation | Strength of dependence at different lags |
| Feature importance | LightGBM / XGBoost top predictors |
| Correlation heatmap | Relationships among selected features |

---

## ⚙️ Setup & run

### 1. Obtain the repository

Use GitHub Desktop to clone, or:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CHE1147-Indoor-PM25-Forecasting.git
cd CHE1147-Indoor-PM25-Forecasting
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-paper.in
jupyter lab
```

### 3. Run the analysis

Open and run **[`Main_File.ipynb`](Main_File.ipynb)**.  
It is the single entry point; it loads data from `DATA/`, applies feature engineering, trains the models, and generates plots.

To read results only, skip this step and open `RESEARCH_HIGHLIGHT.md`.

---

## 📖 Notes on interpretation

- The models predict the **next 10-minute interval**.
- Household hold-out is used so performance reflects generalization to unseen homes.
- Extreme indoor PM values can be capped (thresholds and percentages of clipped observations are documented in the research highlight).
- Persistence is included as a simple, strong baseline.

More details: [`RESEARCH_HIGHLIGHT.md`](RESEARCH_HIGHLIGHT.md) · [`decisions.md`](decisions.md).

---

## 👥 Team

**Dante Flores · Frank Li · Yazan Zamel**

CHE1147 — Chemical Data Science and Engineering, University of Toronto  
Course instructor: Seyed Mohamad Moosavi

## 📚 References

- [Singer et al. (2020), *Indoor air quality in California homes with code-required mechanical ventilation*](https://doi.org/10.1111/ina.12676)
- [Original research dataset at Dryad](https://doi.org/10.7941/D1ZS7X)
- [Ke et al. (2017), LightGBM](https://papers.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [Chen & Guestrin (2016), XGBoost](https://doi.org/10.1145/2939672.2939785)
- [CHE1147 course repository](https://github.com/AI4ChemS/CHE1147)

## 💬 Thank you!

---

<div align="center"><sub>CHE1147 · Team 4 · Fall 2026 · Research/teaching project, not an operational air-quality warning service.</sub></div>
