# ML SessionS #3 — Customer Churn

## About

Binary classification problem predicting which bank customers are likely to churn (leave the bank), so retention efforts can be targeted at the customers most at risk.

## Dataset

- **Source:** [OpenML — id 46362](https://www.openml.org/search?type=data&status=active&id=46362)
- 25 features, no missing values. Already includes engineered features (TF-IDF on surname, interaction terms, one-hot encoded geography and gender). Target: `exited` (1 = churned, 0 = stayed).

## Contents

```
n_003/
├── data/
│   └── customer_churn.arff   # raw dataset
├── hist/                     # earlier training experiments (baseline → tuned XGBoost, SMOTE)
├── eda.ipynb                 # exploratory data analysis
├── prj.ipynb                 # project notebook
├── train.py                  # final training script (best model)
└── report.md                 # full write-up: EDA findings, model experiments, conclusions
```
