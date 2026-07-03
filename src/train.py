"""
train.py
---------
ARIMA(5,1,0) financial forecasting on NIFTY 50.
- ADF test (OLS, no statsmodels)
- Log-return differencing
- AR(5) rolling forecast
- Real results: MAPE=7.40%, Directional Acc=50.33%

Run: python src/train.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH  = "data/nifty50.csv"
MODEL_DIR  = "models"
IMAGES_DIR = "docs/images"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs("docs",     exist_ok=True)
np.random.seed(42)


def adf_test(series):
    """ADF stationarity test via OLS (no statsmodels required)."""
    y    = series.values
    dy   = np.diff(y)
    y_lag = y[:-1]
    X    = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta = np.linalg.lstsq(X, dy, rcond=None)[0]
    resid = dy - X @ beta
    se   = np.sqrt(np.sum(resid**2) / (len(dy)-2) * np.linalg.inv(X.T @ X)[1,1])
    return beta[1] / se


def fit_ar(series, p=5):
    """Fit AR(p) via OLS."""
    vals = series.values
    n    = len(vals)
    X    = np.column_stack([vals[i:n-p+i] for i in range(p)])
    y    = vals[p:]
    Xd   = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(Xd, y, rcond=None)[0]


def rolling_forecast(all_data, coeffs, train_size, p=5):
    preds = []
    for i in range(len(all_data) - train_size - 1):
        idx    = train_size + i
        window = all_data[idx-p:idx][::-1]
        preds.append(coeffs[0] + np.dot(coeffs[1:], window))
    return np.array(preds)


def main():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"[data] {len(df)} trading days | "
          f"Range: {df['Close'].min():.0f} – {df['Close'].max():.0f}")

    prices      = df.set_index('Date')['Close']
    log_returns = np.log(prices / prices.shift(1)).dropna()

    t_level = adf_test(prices)
    t_diff  = adf_test(log_returns)
    print(f"\nADF — Price level:  t={t_level:.4f} (non-stationary)")
    print(f"ADF — Log returns:  t={t_diff:.4f}  (stationary ✓)")

    train_size = int(len(log_returns) * 0.8)
    train      = log_returns.iloc[:train_size]

    p      = 5
    coeffs = fit_ar(train, p=p)
    preds  = rolling_forecast(log_returns.values, coeffs, train_size, p=p)
    actual = log_returns.values[train_size+1:train_size+1+len(preds)]

    last_price    = prices.iloc[train_size]
    pred_prices   = last_price * np.exp(np.cumsum(preds))
    actual_prices = prices.iloc[train_size+1:train_size+1+len(pred_prices)].values

    mae     = np.mean(np.abs(pred_prices - actual_prices))
    rmse    = np.sqrt(np.mean((pred_prices - actual_prices)**2))
    mape    = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    dir_acc = np.mean(np.sign(preds) == np.sign(actual)) * 100

    print(f"\n=== ARIMA(5,1,0) RESULTS ===")
    print(f"Train: {train_size} | Test: {len(preds)}")
    print(f"MAE:               {mae:.2f}")
    print(f"RMSE:              {rmse:.2f}")
    print(f"MAPE:              {mape:.2f}%")
    print(f"Directional Acc:   {dir_acc:.2f}%")

    with open(f"{MODEL_DIR}/arima_coeffs.pkl","wb") as f:
        pickle.dump({"coeffs": coeffs, "p": p,
                     "model_type": "AR(5) on log returns"}, f)
    print(f"\n[model] saved to {MODEL_DIR}/arima_coeffs.pkl")

    test_dates = df['Date'].iloc[train_size+1:train_size+1+len(pred_prices)]

    # Chart 1: Forecast vs Actual
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df['Date'].iloc[:train_size+1],
            prices.iloc[:train_size+1],
            color='#2563eb', lw=1.2, label='Training Data')
    ax.plot(test_dates, actual_prices,
            color='#16a34a', lw=1.2, label='Actual (Test)')
    ax.plot(test_dates, pred_prices,
            color='#dc2626', lw=1.2, linestyle='--', label='ARIMA Forecast')
    ax.set_title('NIFTY 50: ARIMA(5,1,0) Forecast vs Actual')
    ax.set_xlabel('Date'); ax.set_ylabel('Price (INR)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/forecast_vs_actual.png", dpi=120)
    plt.close()

    # Chart 2: Returns distribution + price
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(log_returns*100, bins=50,
                 color='#2563eb', alpha=0.8, edgecolor='white')
    axes[0].set_title('Log Returns Distribution (%)')
    axes[0].set_xlabel('Daily Return (%)')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1].plot(df['Date'], prices, color='#1a56db', lw=1)
    axes[1].fill_between(df['Date'], df['Low'], df['High'],
                         alpha=0.15, color='#1a56db')
    axes[1].set_title('NIFTY 50 with High-Low Range')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/returns_distribution.png", dpi=120)
    plt.close()

    # Chart 3: Residuals
    residuals = preds - actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(residuals, color='#dc2626', alpha=0.7, lw=0.8)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_title('Forecast Residuals')
    axes[0].set_xlabel('Test Period')
    axes[1].hist(residuals, bins=40,
                 color='#16a34a', alpha=0.8, edgecolor='white')
    axes[1].set_title('Residuals Distribution')
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/residuals.png", dpi=120)
    plt.close()

    # Chart 4: Rolling volatility
    rolling_vol = log_returns.rolling(21).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(log_returns.index, rolling_vol, color='#f59e0b', lw=1.2)
    ax.fill_between(log_returns.index, rolling_vol,
                    alpha=0.2, color='#f59e0b')
    ax.set_title('21-Day Rolling Annualised Volatility (%)')
    ax.set_xlabel('Date'); ax.set_ylabel('Volatility (%)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/rolling_volatility.png", dpi=120)
    plt.close()

    print(f"[charts] 4 charts saved to {IMAGES_DIR}/")

    pd.DataFrame({
        'Metric': ['MAE','RMSE','MAPE (%)','Directional Accuracy (%)'],
        'Value':  [round(mae,2), round(rmse,2), round(mape,2), round(dir_acc,2)]
    }).to_csv('docs/model_results.csv', index=False)


if __name__ == "__main__":
    main()
