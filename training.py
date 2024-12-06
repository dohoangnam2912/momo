import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample Bitcoin price data (replace with actual data)
data = pd.Series(np.random.randn(1000) * 0.5 + 50, name="Bitcoin_Price")  # Random data for illustration

# 1. ACF Features for Original Data
def calculate_acf_features(series, lags=10, seasonal_lag=None):
    acf_vals = acf(series, nlags=lags)
    acf_features = {
        'acf_1': acf_vals[1],  # First autocorrelation coefficient
        'acf_sum_squares_10': np.sum(acf_vals[1:lags+1]**2)  # Sum of squares of first 10 ACF
    }
    
    if seasonal_lag:
        acf_seasonal = acf(series, nlags=seasonal_lag)
        acf_features['acf_seasonal'] = acf_seasonal[seasonal_lag]  # Seasonal lag autocorrelation
    
    return acf_features

acf_original = calculate_acf_features(data, lags=10, seasonal_lag=7)  # Assuming seasonality is 7 days (example)

# 2. ACF Features for Differenced Data
data_diff1 = data.diff().dropna()  # First differencing
data_diff2 = data.diff().diff().dropna()  # Twice differencing

acf_diff1 = calculate_acf_features(data_diff1, lags=10, seasonal_lag=7)
acf_diff2 = calculate_acf_features(data_diff2, lags=10, seasonal_lag=7)

# Combine all features
features = {**acf_original, **acf_diff1, **acf_diff2}

# Display the features
print(features)

# Example: Assume we have the features in a pandas DataFrame and target (next day's Bitcoin price)
features_df = pd.DataFrame([features])
target = data.shift(-1).dropna()  # Next day's price

# Align features and target
features_df = features_df.iloc[:-1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2, shuffle=False)

# Train LightGBM model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, importance_type='split', max_num_features=10, figsize=(10, 6))
plt.title('Feature Importance')
plt.show()
