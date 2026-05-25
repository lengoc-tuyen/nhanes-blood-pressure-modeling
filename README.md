# NHANES Blood Pressure Modeling

**Course:** MTH00051 — Applied Mathematics and Statistics  
**Dataset:** NHANES 2021–2023 (National Health and Nutrition Examination Survey, Cycle L)  
**Target variable:** Systolic blood pressure (mmHg)

---

## Overview

This project develops and evaluates linear regression models for systolic blood pressure prediction from demographic, anthropometric, and biochemical features collected in the NHANES survey. All statistical methods — including Ordinary Least Squares, Ridge Regression, and Kernel Ridge Regression — are implemented from first principles without reliance on scikit-learn or statsmodels.

---

## Repository Structure

```
nhanes-blood-pressure-modeling/
├── part1/                          # Core statistical algorithms
│   ├── ols_implementation.py       # OLS: β̂ = (XᵀX)⁻¹Xᵀy, hat matrix, Gauss-Markov demo
│   ├── ridge_lasso.py              # Ridge/Lasso regression, VIF computation
│   ├── residual_analysis.py        # Model metrics (R², MAE, RMSE), coefficient inference, diagnostic plots
│   ├── cross_validation.py         # k-fold cross-validation
│   └── part1_notebook.ipynb        # Demonstrations and theory
│
├── part2/                          # Data pipeline and model comparison
│   ├── data/
│   │   ├── dataset/                # Raw NHANES .xpt files (DEMO_L, BPXO_L, BMX_L, ...)
│   │   ├── merge_nhanes.py         # Merge raw .xpt files → nhanes_stroke_analysis.csv
│   │   ├── prep_checkpoint.py      # Outlier removal + BP averaging → nhanes_pre_pipeline.csv
│   │   └── nhanes_pre_pipeline.csv # Pre-imputation checkpoint (pipeline input)
│   │
│   ├── data_pipeline.py            # DataPipeline class (imputation, encoding, scaling)
│   ├── model_comparison.py         # Model training, evaluation, and visualization
│   ├── advanced_methods.py         # Kernel Ridge Regression (bonus)
│   └── part2_notebook.ipynb        # Analysis notebook
│
├── requirement.txt
└── README.md
```

---

## Data Pipeline

### Step 1 — Data Acquisition and Merging (`merge_nhanes.py`)

Raw NHANES data files (SAS XPORT format) are merged on the participant identifier `SEQN` via left joins across 10 survey components: demographics (DEMO), blood pressure (BPXO), body measures (BMX), smoking (SMQ), diabetes (DIQ), cardiac history (MCQ), physical activity (PAQ), sleep (SLQ), total cholesterol (TCHOL), and biochemistry (BIOPRO). The merged dataset contains 21 selected features.

### Step 2 — Outlier Removal and Feature Construction (`prep_checkpoint.py`)

Observations violating physiological plausibility constraints are removed (e.g., systolic BP outside [40, 260] mmHg, BMI outside [10, 90] kg/m²). Three repeated systolic measurements (BPXOSY1–3) are averaged into a single target variable `SYSTOLIC_TARGET`; pulse readings (BPXOPLS1–3) are similarly averaged into `BPXOPLS`. This produces `nhanes_pre_pipeline.csv` — the pre-imputation checkpoint used as input to the pipeline.

**Output:** 7,515 observations × 10 features; missing values preserved for downstream imputation.

### Step 3 — DataPipeline (`data_pipeline.py`)

The `DataPipeline` class implements a sklearn-style `fit` / `transform` interface with strict separation between training and test sets to prevent data leakage.

| Operation | Continuous features | Categorical features |
|---|---|---|
| Imputation | Median (fit on train) | Mode (fit on train) |
| Encoding | Z-score standardisation | One-hot encoding (categories from train) |

- `fit(X_train)` — learns median, mode, mean, standard deviation, and category maps exclusively from the training partition.
- `transform(X)` — applies learned parameters to any partition; unseen categories are encoded as zero vectors.
- `load_and_split(csv_path, test_size=0.2, seed=42)` — stratified random split into train (6,012) and test (1,503) observations.

**Final feature set (10 dimensions):** `bias`, `RIDAGEYR`, `BMXBMI`, `BPXOPLS`, `LBXTC`, `LBXSCR`, `RIAGENDR_2.0`, `SMQ020_2.0`, `DIQ010_2.0`, `DIQ010_3.0`

---

## Model Comparison (`model_comparison.py`)

Three models are trained on the standardised feature matrix and evaluated on the held-out test set.

### Model 1 — OLS (Full)

Closed-form solution:

$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$

All 9 features retained.

### Model 2 — OLS (Selective)

Two-stage feature selection applied before estimation:

1. **p-value filter** — features with p > 0.05 under the t-test from `coef_inference` are removed.
2. **VIF filter** — features with Variance Inflation Factor > 10 are removed iteratively (highest VIF first) to address multicollinearity.

Retained features: age (`RIDAGEYR`), BMI (`BMXBMI`), total cholesterol (`LBXTC`), sex (`RIAGENDR_2.0`).

### Model 3 — Ridge Regression

$$\hat{\beta}_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$$

Optimal regularisation parameter $\lambda^*$ selected by 5-fold cross-validation over $\lambda \in [10^{-3}, 10^4]$ (log-uniform grid, 50 points). Best $\lambda^* = 37.28$.

### Test Set Results

| Model | MAE | RMSE | R² | R²_adj | Features |
|---|---|---|---|---|---|
| OLS (Full) | 11.1602 | 14.7232 | 0.2988 | 0.2946 | 9 |
| **OLS (Selective)** | **11.1586** | **14.7127** | **0.2998** | **0.2980** | **4** |
| Ridge | 11.1621 | 14.7196 | 0.2992 | 0.2950 | 9 |

OLS with feature selection achieves the best generalisation performance while using only four predictors, confirming that multicollinearity reduction and parsimony improve out-of-sample fit in this dataset.

**Generated figures:** `fig_feature_importance.png`, `fig_model_comparison.png`, `fig_residual_analysis.png`, `fig_cv_lambda.png`

---

## Kernel Ridge Regression (`advanced_methods.py`)

Kernel Ridge Regression extends ridge regression to a reproducing kernel Hilbert space via the kernel trick:

$$\hat{y}(x^*) = k(x^*)^\top (K + \lambda I)^{-1} y$$

where $K_{ij} = k(x_i, x_j)$ is the Gram matrix under the RBF (Gaussian) kernel:

$$k(x, x') = \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

Hyperparameters $(\lambda, \ell)$ are selected by 5-fold cross-validation on a subsampled training set ($n = 800$) over a $8 \times 5$ grid ($\lambda \in [10^{-1}, 10^3]$, $\ell \in [10^{-1}, 10^1]$). The linear kernel limit ($\ell \to \infty$) is analogous to standard ridge regression.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| OLS | 11.1602 | 14.7232 | 0.2988 |
| Kernel Ridge ($\lambda^*=0.1$, $\ell^*=10$) | 11.0830 | 14.8792 | 0.2839 |

**Generated figures:** `fig_kernel_cv_heatmap.png`, `fig_kernel_predicted_vs_actual.png`

---

## Reproducibility

### Requirements

```
Python >= 3.10
numpy, pandas, matplotlib, seaborn
```

Install dependencies:

```bash
pip install -r requirement.txt
```

### Running the Pipeline

```bash
# 1. Merge raw NHANES files (requires .xpt files in part2/data/dataset/)
python part2/data/merge_nhanes.py

# 2. Create pre-imputation checkpoint
python part2/data/prep_checkpoint.py

# 3. Train models and evaluate
python part2/model_comparison.py

# 4. Kernel Ridge Regression (bonus)
python part2/advanced_methods.py
```

All scripts are self-contained and write output figures to their respective directories.

---

## Dataset

NHANES 2021–2023 (Cycle L) public-use data files are available from the U.S. Centers for Disease Control and Prevention:  
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

Place the downloaded `.xpt` files in `part2/data/dataset/` before running `merge_nhanes.py`.
