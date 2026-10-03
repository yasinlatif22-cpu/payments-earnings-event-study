"""Risk and performance statistics. Pure functions: no downloads, easy to test."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: P_t / P_(t-1) - 1."""
    return prices.pct_change().dropna()


def total_return(ret):
    """Compounded return over the whole sample."""
    return (1 + ret).prod() - 1


def drawdown(ret):
    """Drawdown from the running peak of wealth; the starting $1 counts as the first peak."""
    wealth = (1 + ret).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


def annualised_vol(ret, periods: int = TRADING_DAYS):
    """Sample std of returns times sqrt(periods); assumes returns are uncorrelated over time."""
    return ret.std() * np.sqrt(periods)


def downside_deviation(ret, periods: int = TRADING_DAYS, target: float = 0.0):
    """Root-mean-square of returns below `target` (other days count as 0), annualised."""
    shortfall = np.minimum(ret - target, 0.0)
    return np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods)


def beta(y, x):
    """OLS slope of y on x: cov(y, x) / var(x)."""
    return np.cov(y, x)[0, 1] / np.var(x, ddof=1)


def sharpe(ret, rf: float = 0.0, periods: int = TRADING_DAYS):
    """(annualised arithmetic mean return - risk-free rate) / annualised volatility."""
    return (ret.mean() * periods - rf) / annualised_vol(ret, periods)


def sortino(ret, rf: float = 0.0, periods: int = TRADING_DAYS):
    """Like Sharpe, but divides by downside deviation instead of total volatility."""
    return (ret.mean() * periods - rf) / downside_deviation(ret, periods)


def risk_table(ret: pd.DataFrame, benchmark: str, rf: float = 0.0) -> pd.DataFrame:
    """One row of statistics per column of `ret`; `benchmark` names the market column."""
    mkt = ret[benchmark]
    mkt_total = total_return(mkt)
    rows = {}
    for t in ret.columns:
        r = ret[t]
        rows[t] = {
            "Total return": total_return(r),
            f"Return vs {benchmark}": total_return(r) - mkt_total,
            "Ann. vol": annualised_vol(r),
            "Downside dev": downside_deviation(r),
            "Max drawdown": drawdown(r).min(),
            f"Beta vs {benchmark}": beta(r, mkt),
            "Sharpe": sharpe(r, rf),
            "Sortino": sortino(r, rf),
        }
    return pd.DataFrame(rows).T


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Beta of y on x over a rolling window of observations."""
    return y.rolling(window).cov(x) / x.rolling(window).var()
