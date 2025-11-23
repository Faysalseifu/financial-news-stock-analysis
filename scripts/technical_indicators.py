import yfinance as yf
import pandas as pd
import talib


def load_stock(symbol="AAPL", start="2018-01-01", end="2023-01-01"):
    df = yf.download(symbol, start=start, end=end)
    df = df.dropna()
    return df


def add_indicators(df, symbol=None):
    """Add common technical indicators to `df`.

    Ensures a 1-D numpy array is passed to TA-Lib by extracting a Series
    from `df['Close']` when yfinance returns a DataFrame/MultiIndex.
    """
    close = df["Close"]

    # If yfinance returned a DataFrame (multi-ticker or MultiIndex cols),
    # pick the correct Series.
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close_series = close.iloc[:, 0]
        else:
            # Prefer `symbol` column if provided, otherwise take first column
            if symbol is not None and symbol in close.columns:
                close_series = close[symbol]
            else:
                close_series = close.iloc[:, 0]
    else:
        close_series = close

    # Ensure numeric 1-D numpy array for TA-Lib
    close_vals = close_series.astype(float).to_numpy()

    sma20 = talib.SMA(close_vals, timeperiod=20)
    sma50 = talib.SMA(close_vals, timeperiod=50)
    rsi = talib.RSI(close_vals, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close_vals, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # Assign results back to a copy of the original DataFrame (aligned by index)
    out = df.copy()
    out["SMA_20"] = sma20
    out["SMA_50"] = sma50
    out["RSI"] = rsi
    out["MACD"] = macd
    out["MACD_signal"] = macd_signal
    out["MACD_hist"] = macd_hist

    return out


if __name__ == "__main__":
    symbol = "AAPL"
    df = load_stock(symbol=symbol)
    print(df.head())
    df = add_indicators(df, symbol=symbol)
    df.to_csv("outputs/aapl_indicators.csv")
    print("Saved indicators to outputs/")
