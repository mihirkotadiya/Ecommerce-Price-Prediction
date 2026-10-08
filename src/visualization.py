import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok= True)



def plot_feature_importance(model, feature_names):
    #Importance
    importance = pd.Series(
        model.feature_importances_,
        index = feature_names
    )

    top15 = importance.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 7))

    sns.barplot(
        x = top15.values,
        y = top15.index
    )

    plt.title("Top 15 Important Fetures")
    plt.savefig(os.path.join(IMAGE_DIR, "top_important_features.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

def plot_actual_vs_predicted(y_test, y_preds_xgb):
    plt.figure(figsize = (8, 6))

    plt.scatter(
        y_test,
        y_preds_xgb,
        alpha = 0.4
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.savefig(os.path.join(IMAGE_DIR, "actual_vs_predicted.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

def plot_residual_distribution(y_test, y_preds_xgb):
    residual = y_test - y_preds_xgb
    plt.figure(figsize=(8,5))
    sns.histplot(residual, kde = True)
    plt.title("Residual Distribution")
    plt.savefig(os.path.join(IMAGE_DIR, "residual_distribution.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    return residual


def plot_residual_scatter(y_test, y_preds_xgb, residual):
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, residual, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Price")
    plt.ylabel("Residual")
    plt.title("Residuals vs Actual")
    plt.savefig(os.path.join(IMAGE_DIR, "residual_scatter.png"), dpi = 300, bbox_inches = "tight")
    plt.show()
