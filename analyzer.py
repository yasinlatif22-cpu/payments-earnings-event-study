"""
Risk profile of Visa (V) and Mastercard (MA) against the S&P 500 (SPY).

A fixed date range makes the results reproducible. Returns are simple daily
returns computed from dividend-adjusted closing prices.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKERS = ["V", "MA"]
BENCH = "SPY"
START, END = "2025-10-02", "2026-10-03"   # END is exclusive
TRADING_DAYS = 252
ROLL = 60                                  # rolling window in trading days


def load_prices():
    px = yf.download(TICKERS + [BENCH], start=START, end=END,
                     auto_adjust=True, progress=False)["Close"]
    return px.dropna()


def risk_free_rate():
    """Average 3-month T-bill yield (^IRX is quoted in percent) over the sample."""
    try:
        irx = yf.download("^IRX", start=START, end=END, progress=False)["Close"]
        rate = float(np.nanmean(irx.values)) / 100
        if not np.isnan(rate):
            return rate, "average ^IRX over the sample"
    except Exception:
        pass
    return 0.0, "assumed 0 (could not download ^IRX)"


def drawdown(ret):
    """Drawdown from the running peak; the starting $1 counts as the first peak."""
    wealth = (1 + ret).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


def risk_table(ret, rf):
    """One row of risk/return statistics per column of `ret`."""
    mkt = ret[BENCH]
    mkt_total = (1 + mkt).prod() - 1
    rows = {}
    for t in ret.columns:
        r = ret[t]
        total = (1 + r).prod() - 1
        ann_ret = r.mean() * TRADING_DAYS                       # arithmetic, for Sharpe
        vol = r.std() * np.sqrt(TRADING_DAYS)
        # downside deviation: only returns below 0 count, annualised
        downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
        rows[t] = {
            "Total return": total,
            "Return vs SPY": total - mkt_total,
            "Ann. vol": vol,
            "Downside dev": downside,
            "Max drawdown": drawdown(r).min(),
            "Beta vs SPY": np.cov(r, mkt)[0, 1] / mkt.var(),
            "Sharpe": (ann_ret - rf) / vol,
            "Sortino": (ann_ret - rf) / downside,
        }
    return pd.DataFrame(rows).T


def main():
    ret = load_prices().pct_change().dropna()
    rf, rf_note = risk_free_rate()

    table = risk_table(ret, rf)
    pretty = table.copy()
    for col in pretty.columns:
        fmt = "{:.2f}" if col in ("Beta vs SPY", "Sharpe", "Sortino") else "{:.2%}"
        pretty[col] = pretty[col].map(fmt.format)

    print(f"\nPeriod: {ret.index[0]:%Y-%m-%d} to {ret.index[-1]:%Y-%m-%d} "
          f"({len(ret)} trading days)")
    print(f"Risk-free rate: {rf:.2%} ({rf_note})")
    print("\nRisk summary")
    print(pretty.T.to_string())

    print("\nCorrelation of daily returns")
    print(ret.corr().round(2).to_string())

    gap = ret["V"] - ret["MA"]
    print(f"\nV minus MA: summed daily gap {gap.sum():.2%}, "
          f"annualised tracking error {gap.std() * np.sqrt(TRADING_DAYS):.2%}")

    table.round(4).to_csv("risk_summary.csv")

    # Charts
    cum = (1 + ret).cumprod() - 1
    dd = drawdown(ret)
    roll_vol = ret[TICKERS].rolling(ROLL).std() * np.sqrt(TRADING_DAYS)
    roll_corr = ret["V"].rolling(ROLL).corr(ret["MA"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    (cum * 100).plot(ax=ax[0, 0], title="Cumulative return (%)")
    (dd * 100).plot(ax=ax[0, 1], title="Drawdown from peak (%)")
    (roll_vol * 100).plot(ax=ax[1, 0],
                          title=f"Rolling {ROLL}-day annualised volatility (%)")
    roll_corr.plot(ax=ax[1, 1], title=f"Rolling {ROLL}-day correlation, V vs MA")
    plt.tight_layout()
    plt.savefig("risk_chart.png", dpi=150)
    print("\nSaved risk_summary.csv and risk_chart.png")


if __name__ == "__main__":
    main()
