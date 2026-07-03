# Financial Forecasting System

ARIMA(5,1,0) time-series forecasting on NIFTY 50 stock price data.
ADF stationarity test, log-return differencing, AR(5) via OLS — all
implemented from scratch without statsmodels.

## Real Results

```
Dataset: 756 trading days (3 years NIFTY 50)
Price range: 16,937 – 23,966 INR

ADF Test — Price level:  t = -1.89  → non-stationary
ADF Test — Log returns:  t = -46.15 → stationary ✓

ARIMA(5,1,0) Evaluation:
Train: 604 days | Test: 151 days

MAE:                  1682.01 points
RMSE:                 1884.36 points
MAPE:                 7.40%
Directional Accuracy: 50.33%
```

> Note: 50% directional accuracy is consistent with academic literature
> on ARIMA for equity markets — financial markets are near random walk.
> Model value is in trend and volatility estimation.

## Charts

### Forecast vs Actual
![Forecast](docs/images/forecast_vs_actual.png)

### Returns Distribution + Price Range
![Returns](docs/images/returns_distribution.png)

### Residuals Analysis
![Residuals](docs/images/residuals.png)

### 21-Day Rolling Volatility
![Volatility](docs/images/rolling_volatility.png)

## What this demonstrates

- ADF test from scratch (numpy OLS, no statsmodels)
- Log-return differencing for stationarity
- Rolling evaluation respecting temporal order (no data leakage)
- Proper train/test split on time-series data

## Stack

Python, Pandas, NumPy, Matplotlib (no statsmodels — built from scratch)

## Run

```bash
pip install -r requirements.txt
python src/train.py
pytest tests/
```

## Structure

```
financial-forecasting-system/
├── data/nifty50.csv
├── src/train.py
├── models/arima_coeffs.pkl
├── docs/images/
├── tests/test_forecasting.py
└── requirements.txt
```
