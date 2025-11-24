import pandas as pd

# Optional dependency: yfinance. Keep import guarded so module can be imported
# even if yfinance is not installed (useful for notebooks that may not have
# all optional packages installed at import time).
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    yf = None
    YFINANCE_AVAILABLE = False

# Prefer TA-Lib when available for performance and compatibility with existing code.
# If TA-Lib (C binary) is not installed, fall back to `pandas_ta` if available and
# otherwise raise a clear error at runtime.
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except Exception:
    talib = None  # type: ignore
    TALIB_AVAILABLE = False

try:
    import pandas_ta as pandas_ta  # optional fallback
    PANDAS_TA_AVAILABLE = True
except Exception:
    pandas_ta = None
    PANDAS_TA_AVAILABLE = False


def load_stock(symbol="AAPL", start="2018-01-01", end="2023-01-01"):
    if not YFINANCE_AVAILABLE or yf is None:
        raise ImportError(
            "yfinance is required to download stock data. Install with `pip install yfinance` or provide cached CSVs in `outputs/`."
        )

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

    # Compute indicators using TA-Lib if available, otherwise try pandas_ta.
    # Ensure numeric 1-D numpy array for TA-Lib usage
    close_vals = close_series.astype(float).to_numpy()

    if TALIB_AVAILABLE and talib is not None:
        sma20 = talib.SMA(close_vals, timeperiod=20)
        sma50 = talib.SMA(close_vals, timeperiod=50)
        rsi = talib.RSI(close_vals, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(
            close_vals, fastperiod=12, slowperiod=26, signalperiod=9
        )
    elif PANDAS_TA_AVAILABLE and pandas_ta is not None:
        # Use pandas_ta which operates on pandas Series and returns Series
        cs = close_series.astype(float)
        sma20 = pandas_ta.sma(cs, length=20).to_numpy()
        sma50 = pandas_ta.sma(cs, length=50).to_numpy()
        rsi = pandas_ta.rsi(cs, length=14).to_numpy()
        macd_df = pandas_ta.macd(cs, fast=12, slow=26, signal=9)
        # pandas_ta.macd returns a DataFrame-like object with columns ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
        macd = macd_df.iloc[:, 0].to_numpy()
        macd_signal = macd_df.iloc[:, 1].to_numpy()
        macd_hist = macd_df.iloc[:, 2].to_numpy()
    else:
        raise ImportError(
            "TA-Lib not available and pandas_ta not installed. Install one: `pip install TA-Lib` (requires C library) or `pip install pandas_ta`."
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
