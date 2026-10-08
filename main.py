from pathlib import Path
import joblib

from src.data_loader import load_data
from src.eda import perform_eda
from src.feature_engineering import feature_engineering
from src.model import train_models
from src.evaluation import evaluate_models
from src.visualization import plot_feature_importance, plot_actual_vs_predicted, plot_residual_distribution

# Load Dataset
df = load_data()

# EDA
perform_eda(df)

# Feature Engineering
X_train, X_test, y_train, y_test, X_train_s, X_test_s = feature_engineering(df)

# Model Training
lr, rf, xgb, y_test, y_preds_lr, y_preds_rf, y_preds_xgb = train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_s,
    X_test_s
)

# Evaluation
evaluate_models(
    y_test,
    y_preds_lr,
    y_preds_rf,
    y_preds_xgb
)

# Visualization
plot_feature_importance(
    xgb,
    X_train.columns
)
plot_actual_vs_predicted(y_test, y_preds_xgb)
plot_residual_distribution(y_test, y_preds_xgb)

# Save best model artifact for later reuse
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(xgb, MODEL_DIR / "xgb_best_model.joblib")
print(f"Saved trained XGBoost model to: {MODEL_DIR / 'xgb_best_model.joblib'}")
