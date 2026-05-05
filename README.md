# Supervised Learning for Loan Default Prediction
### Machine Learning Assignment — University of Europe for Applied Sciences

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Overview

This project applies supervised machine learning techniques to predict whether a borrower will repay their loan, using the **Loan Prediction Dataset 2025**. The target variable `loan_paid_back` is binary (0 = default, 1 = repaid), making this a **binary classification** task.

The project investigates 7 research questions covering baseline performance, model comparison, preprocessing effects, feature importance, metric sensitivity, robustness, and final model recommendation.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | Loan Prediction Dataset 2025 |
| **Source** | [https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025) |
| **Rows** | 20,000 |
| **Columns** | 22 (21 input features + 1 target) |
| **Target** | `loan_paid_back` (0 = default, 1 = repaid) |
| **Class balance** | 80% repaid / 20% default |
| **Missing values** | None |
| **Kaggle usability** | 10.00 / 10.00 |

### Input Features
`age`, `gender`, `marital_status`, `education_level`, `annual_income`, `monthly_income`, `employment_status`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `loan_purpose`, `interest_rate`, `loan_term`, `installment`, `grade_subgrade`, `num_of_open_accounts`, `total_credit_limit`, `current_balance`, `delinquency_history`, `public_records`, `num_of_delinquencies`

---

## 🔬 Research Questions

| # | Research Question |
|---|---|
| RQ1 | How effectively can baseline models predict loan repayment? |
| RQ2 | Which model achieves the best predictive performance? |
| RQ3 | How do preprocessing strategies (encoding, scaling, SMOTE) affect performance? |
| RQ4 | Which features contribute most to predicting loan repayment? |
| RQ5 | How does model ranking change across different evaluation metrics? |
| RQ6 | How robust is the best model under CV and data quality degradation? |
| RQ7 | Which model offers the best trade-off for real-world deployment? |

---

## 📁 Repository Structure

```
loan-prediction-supervised-learning/
│
├── RQ1_Baseline_Performance.ipynb        # Logistic Regression, Decision Tree, k-NN
├── RQ2_Model_Comparison.ipynb            # 6-model comparison including XGBoost, RF, SVM
├── RQ3_Preprocessing_Effects.ipynb       # Encoding + Scaling + SMOTE ablation study
├── RQ4_Feature_Importance.ipynb          # XGBoost gain + SHAP analysis
├── RQ5_Metric_Sensitivity.ipynb          # Rank matrix + Spearman correlations
├── RQ6_Robustness.ipynb                  # Cross-validation + noise + missingness
├── RQ7_Final_Recommendation.ipynb        # Decision matrix + radar chart + fairness check
│
├── figures/
│   ├── fig1_baseline.png
│   ├── fig2_model_comparison.png
│   ├── fig3_preprocessing.png
│   ├── fig4_feature_importance.png
│   ├── fig4b_shap.png
│   ├── fig5_metric_sensitivity.png
│   ├── fig6_robustness.png
│   └── fig7_radar.png
│
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| k-Nearest Neighbours (k=5) | Instance-based |
| Support Vector Machine (RBF) | Kernel-based |
| Random Forest (n=200) | Ensemble — Bagging |
| XGBoost (n=200, lr=0.1) | Ensemble — Boosting |

---

## 📏 Evaluation Metrics

- **Recall** — primary metric (minimising missed defaults)
- **AUC-ROC** — secondary metric (overall discrimination)
- Accuracy, Precision, F1-Score — reported for completeness
- Spearman rank correlation — metric sensitivity analysis (RQ5)
- Standard deviation across CV folds — robustness analysis (RQ6)

---

## ▶️ How to Run

### Option A — Run on Kaggle (recommended)

1. Go to [https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025)
2. Click **"New Notebook"**
3. Upload one of the `.ipynb` files from this repository (**File → Import Notebook**)
4. In the right panel under **Datasets**, add `loan-prediction-dataset-2025`
5. The dataset path is already set to:
   ```python
   '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'
   ```
6. Click **"Run All"** → then **"Save Version → Save & Run All (Commit)"**
7. Run notebooks in order: RQ1 → RQ2 → RQ3 → RQ4 → RQ5 → RQ6 → RQ7

### Option B — Run locally

1. Clone the repository:
   ```bash
   git clone https://github.com/daleogont/loan-prediction-supervised-learning.git
   cd loan-prediction-supervised-learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   [https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025)

4. Place the CSV file at:
   ```
   loan_dataset_20000.csv
   ```

5. Update the dataset path in each notebook from:
   ```python
   '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'
   ```
   to:
   ```python
   'loan_dataset_20000.csv'
   ```

6. Launch Jupyter and run notebooks in order RQ1 → RQ7:
   ```bash
   jupyter notebook
   ```

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
xgboost>=1.7.0
imbalanced-learn>=0.11.0
scipy>=1.11.0
shap>=0.42.0
```

Install with:
```bash
pip install -r requirements.txt
```

> **Note:** All libraries are pre-installed in Kaggle notebooks. No manual installation needed when running on Kaggle.

---

## 📈 Key Results (Summary)

| Model | Accuracy | F1-Score | AUC |
|---|---|---|---|
| Logistic Regression | 0.802 | 0.68 | 0.84 |
| Decision Tree | 0.836 | 0.75 | 0.83 |
| k-NN (k=5) | 0.817 | 0.71 | 0.85 |
| SVM (RBF) | 0.864 | 0.80 | 0.90 |
| Random Forest | 0.883 | 0.83 | 0.92 |
| **XGBoost** | **0.901** | **0.86** | **0.94** |

> Results above are from executed notebooks. XGBoost achieves the best performance across all metrics.

---

## 🔗 Links

- 📊 **Dataset:** [Loan Prediction Dataset 2025 on Kaggle](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025)
- 💻 **Kaggle Profile:** [https://www.kaggle.com/daleogont](https://www.kaggle.com/daleogont)
- 🐙 **GitHub:** [https://github.com/daleogont/loan-prediction-supervised-learning](https://github.com/daleogont/loan-prediction-supervised-learning)

---

## 👤 Author

**Daniil** — MSc Data Science, University of Europe for Applied Sciences
