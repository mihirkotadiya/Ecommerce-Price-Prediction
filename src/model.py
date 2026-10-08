from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def train_models(X_train, X_test, y_train, y_test, X_train_s, X_test_s):
        #Linear Regression
    print(f"\nLinear Regression")
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)

    y_preds_lr = lr.predict(X_test_s)

        #Random Forest
    print(f"\nRandom Forest")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf = 2,
        random_state=42,
        n_jobs = -1
    )
    rf.fit(X_train_s,y_train)

    y_preds_rf = rf.predict(X_test_s)

    
         #XGBoost
    print(f"\nXGBoost result!")
    XGBoost = XGBRegressor(
        n_estimators = 200,
        max_depth = 15,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        n_jobs = -1,
        random_state = 42
    )
    XGBoost.fit(X_train, y_train)
    y_preds_xgb = XGBoost.predict(X_test)

    return (
    lr,
    rf,
    XGBoost,
    y_test,
    y_preds_lr,
    y_preds_rf,
    y_preds_xgb
)