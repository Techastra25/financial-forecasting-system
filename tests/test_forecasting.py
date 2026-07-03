"""
test_forecasting.py — Run: pytest tests/test_forecasting.py
"""
import pickle, numpy as np, pandas as pd, pytest


def test_log_returns_stationary():
    df  = pd.read_csv("data/nifty50.csv")
    ret = np.diff(np.log(df['Close'].values))
    v1  = np.var(ret[:len(ret)//2])
    v2  = np.var(ret[len(ret)//2:])
    assert max(v1,v2)/min(v1,v2) < 3.0


def test_ar_coefficients_loaded():
    with open("models/arima_coeffs.pkl","rb") as f: m = pickle.load(f)
    assert 'coeffs' in m and m['p'] == 5
    assert len(m['coeffs']) == m['p'] + 1


def test_forecast_not_nan():
    with open("models/arima_coeffs.pkl","rb") as f: m = pickle.load(f)
    df  = pd.read_csv("data/nifty50.csv")
    ret = np.diff(np.log(df['Close'].values))
    w   = ret[-m['p']:][::-1]
    pred = m['coeffs'][0] + np.dot(m['coeffs'][1:], w)
    assert not np.isnan(pred)


def test_train_test_temporal_order():
    df = pd.read_csv("data/nifty50.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    n  = len(df)
    t  = int(n*0.8)
    assert df['Date'].iloc[t] > df['Date'].iloc[t-1]


def test_required_columns():
    df = pd.read_csv("data/nifty50.csv")
    for col in ['Date','Open','High','Low','Close','Volume']:
        assert col in df.columns
