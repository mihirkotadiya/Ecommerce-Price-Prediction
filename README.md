# Ecommerce Price Prediction

This repository contains an end-to-end machine learning pipeline for predicting ecommerce product prices using the Olist dataset.

## Project structure

- `main.py`: orchestrates the pipeline from data loading to model training, evaluation, visualization, and model saving.
- `src/data_loader.py`: loads and merges dataset CSVs, fills missing values, and returns a combined dataframe.
- `src/eda.py`: performs exploratory data analysis and saves plots to `images/`.
- `src/feature_engineering.py`: creates time-based and business-informed features, encodes categorical data, and splits the dataset.
- `src/model.py`: trains Linear Regression, Random Forest, and XGBoost regression models.
- `src/evaluation.py`: calculates model metrics and prints evaluation results.
- `src/visualization.py`: generates and saves diagnostic plots.
- `requirements.txt`: project dependencies.

## Setup

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Run the project:
```powershell
python main.py
```

The script will generate image artifacts in `images/` and save the trained XGBoost model to `models/xgb_best_model.joblib`.

## Notes

- Keep raw dataset files in `data/` and do not commit large or proprietary files to Git if not necessary.
- Add more robust validation, testing, and deployable packaging for production readiness.
