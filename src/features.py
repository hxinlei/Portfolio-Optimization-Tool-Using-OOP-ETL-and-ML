from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change().dropna(how="all")

def engineer_features(tickers: list[str], close_prices: pd.DataFrame, volumes: pd.DataFrame):
    # match your original logic
    returns = close_prices.pct_change().shift(-1)
    avg_return = returns.mean(axis=1)

    df = pd.DataFrame(index=close_prices.index)
    df["Return"] = avg_return

    rep = tickers[0]
    df["Close"] = close_prices[rep]
    df["Volume"] = volumes[rep]

    ohlc = yf.download(rep, start="1990-01-01", auto_adjust=True, progress=False)
    for col in ["Open", "High", "Low"]:
        df[col] = ohlc[col] if col in ohlc.columns else np.nan

    horizons = [2, 5, 60, 250, 1000]
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    for h in horizons:
        df[f"Close_Ratio_{h}"] = df["Close"] / df["Close"].rolling(h).mean()
        df[f"Trend_{h}"] = df["Return"].shift(1).rolling(h).sum()
        predictors += [f"Close_Ratio_{h}", f"Trend_{h}"]

    df = df.dropna()
    return df, predictors
