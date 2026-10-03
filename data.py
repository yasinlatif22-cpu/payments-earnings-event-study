"""Download and cache market data (Yahoo Finance via yfinance).

Yahoo data is convenient but not institutional quality: prices can be revised,
consensus EPS is a snapshot, and earnings timestamps are approximate. Every
download is cached under data/raw/ so a run can be repeated on identical data.
Delete data/raw/ to download fresh data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path("data/raw")


def _cache_path(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Dividend- and split-adjusted closes, one column per ticker. `end` is exclusive."""
    path = _cache_path(f"prices_{'-'.join(sorted(tickers))}_{start}_{end}.csv")
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"].dropna()
    if px.empty:
        raise RuntimeError(f"No price data returned for {tickers}")
    px.to_csv(path)
    return px


def load_risk_free_rate(start: str, end: str) -> tuple[float, str]:
    """Average 3-month T-bill yield over the window (^IRX is quoted in percent)."""
    path = _cache_path(f"irx_{start}_{end}.csv")
    try:
        if path.exists():
            irx = pd.read_csv(path, index_col=0)["irx"]
        else:
            irx = yf.download("^IRX", start=start, end=end,
                              progress=False)["Close"].squeeze()
            irx.rename("irx").to_csv(path)
        rate = float(np.nanmean(irx.values)) / 100
        if not np.isnan(rate):
            return rate, "average ^IRX over the window"
    except Exception:
        pass
    return 0.0, "assumed 0 (could not download ^IRX)"


def surprise_pct(reported, estimate):
    """EPS surprise in percent of the absolute consensus estimate."""
    return (reported - estimate) / abs(estimate) * 100


def load_earnings(ticker: str) -> pd.DataFrame:
    """Past earnings releases: announce_date, hour (New York time), EPS surprise in %."""
    path = _cache_path(f"earnings_{ticker}.csv")
    if path.exists():
        return pd.read_csv(path, parse_dates=["announce_date"])
    df = yf.Ticker(ticker).get_earnings_dates(limit=40)
    df = df.dropna(subset=["Reported EPS", "EPS Estimate"]).copy()
    local = df.index.tz_localize(None)       # keep New York wall-clock time
    out = pd.DataFrame({
        "announce_date": local.normalize(),
        "hour": local.hour,
        "surprise_pct": surprise_pct(df["Reported EPS"].to_numpy(),
                                     df["EPS Estimate"].to_numpy()),
    })
    out.to_csv(path, index=False)
    return out
