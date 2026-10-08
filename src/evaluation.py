from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_models(y_test, y_preds_lr, y_preds_rf, y_preds_xgb):
    print(f"\n{'Model Evaluation':^70}")
    print(f"""{'Model':<20}
        {'MAE':>12}
        {'RMSE':>12}
        {'R2 Score':>12}
    """)
    print("-"*70)

    print(f"""{'Linear Regression':<20}
            {mean_absolute_error(y_test, y_preds_lr):>12.2f}
            {np.sqrt(mean_squared_error(y_test, y_preds_lr)):>12.2f}
            {r2_score(y_test, y_preds_lr):>12.2f}
        """)
    print(f"""{'Random Forest':<20}
            {mean_absolute_error(y_test, y_preds_rf):>12.2f}
            {np.sqrt(mean_squared_error(y_test, y_preds_rf)):>12.2f}
            {r2_score(y_test, y_preds_rf):>12.2f}
        """)
    xgb_r2 = r2_score(y_test, y_preds_xgb)
    rf_r2 = r2_score(y_test, y_preds_rf)
    lr_r2 = r2_score(y_test, y_preds_lr)

    print(f"""{'XGBoost':<20}
            {mean_absolute_error(y_test, y_preds_xgb):>12.2f}
            {np.sqrt(mean_squared_error(y_test, y_preds_xgb)):>12.2f}
            {xgb_r2:>12.2f}
        """)

    print("-"*70)

    best_score = max(lr_r2, rf_r2, xgb_r2)
    best_model = (
        'Linear Regression' if best_score == lr_r2 else
        'Random Forest' if best_score == rf_r2 else
        'XGBoost Regressor'
    )

    print(f"\nBest model : {best_model}")
    print(f"Final R2 Score : {best_score:4f} ({best_score*100:2f}%)")

    return best_score
        
        