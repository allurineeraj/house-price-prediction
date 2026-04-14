import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

print("Shape:", df.shape)
print(df.head())
print(df.describe())
# Check missing values
print(df.isnull().sum())

# Price distribution
plt.figure(figsize=(8, 4))
plt.hist(df['MedHouseVal'], bins=50,
         color='steelblue', edgecolor='white')
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value ($100k)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True,
            fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Model 1: Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train_sc, y_train)

# Model 2: Random Forest (improved)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)  # RF doesn't need scaling

print("Both models trained!")
from sklearn.metrics import (mean_squared_error,
    mean_absolute_error, r2_score)

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    print(f"\n{name}")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    return y_pred

lr_pred = evaluate("Linear Regression",
                   lr, X_test_sc, y_test)
rf_pred = evaluate("Random Forest",
                   rf, X_test, y_test)
# Actual vs Predicted plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, title in zip(
    axes, [lr_pred, rf_pred],
    ["Linear Regression", "Random Forest"]):
    ax.scatter(y_test, pred,
               alpha=0.3, s=10, color='steelblue')
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', lw=1.5)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Feature importance
feat_imp = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values()
feat_imp.plot(kind='barh',
              figsize=(7,5), color='teal')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()