from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df.dropna()[['floorAreaSqM']], test_size=0.2, random_state=42)
y_train, y_test = train_test_split(df.dropna()[['bedrooms']], test_size=0.2, random_state=42)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def ridge_model_summary(X_train, X_test, y_train, y_test, alpha=1.0):
    # 1. Fit Ridge Model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 2. Performance Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.2f}")
    print(f"  RMSE     : {rmse:.2f}")

    # 3. Residuals
    residuals = y_test - y_pred

    # 4. Assumption Checks

    # Linearity: Plot predicted vs actual
    plt.figure(figsize=(5,4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Linearity Check)")
    plt.grid(True)
    plt.show()

    # Residual Normality
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution (Normality)")
    plt.xlabel("Residuals")
    plt.grid(True)
    plt.show()

    # Homoscedasticity: Residuals vs Predictions
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted (Homoscedasticity)")
    plt.grid(True)
    plt.show()

    # Multicollinearity (VIF) – on training data
    print("\n📈 VIF for Multicollinearity (X_train only):")
    X_vif = pd.DataFrame(X_train, columns=X_train.columns if hasattr(X_train, 'columns') else [f"X{i}" for i in range(X_train.shape[1])])
    X_vif = sm.add_constant(X_vif)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print(vif_data.round(2))

    print("\n✅ Assumptions Checked:")
    print("  ✔️ Linearity – Scatter plot of actual vs predicted.")
    print("  ✔️ Normality – Histogram of residuals.")
    print("  ✔️ Homoscedasticity – Residuals vs predicted plot.")
    print("  ✔️ Multicollinearity – VIF (VIF > 5 or 10 indicates problem).")

    return model

ridge_model_summary(X_train, X_test, y_train, y_test, alpha=1.0)
