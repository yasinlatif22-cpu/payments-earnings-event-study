"""Earnings event study: market-model abnormal returns around earnings releases.

Pure functions: no downloads. The design choices (windows, estimation period,
winsorising) are fixed in config.py before looking at results.
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as cfg


def fit_market_model(stock, mkt, i_event, est_len=cfg.EST_LEN, est_gap=cfg.EST_GAP):
    """alpha, beta and residual std from `est_len` days that end `est_gap` days before the event."""
    end = i_event - est_gap
    start = end - est_len
    if start < 0:
        return None
    x, y = mkt.iloc[start:end].values, stock.iloc[start:end].values
    beta, alpha = np.polyfit(x, y, 1)
    sigma = (y - alpha - beta * x).std(ddof=2)
    return alpha, beta, sigma


def event_positions(index, announce_date, hour, timing="yahoo"):
    """(i0, ir): position of the announcement date and of the reaction day.

    After-close releases react on the next trading day. `timing` overrides Yahoo's
    timestamp: "after_close", "before_open", or "yahoo" (after close if hour >= 16).
    """
    i0 = index.searchsorted(announce_date)
    if timing == "after_close":
        after = True
    elif timing == "before_open":
        after = False
    else:
        after = hour >= 16
    return i0, i0 + int(after)


def event_row(ticker, ev, ret, benchmark):
    """Returns and abnormal returns around one earnings release, or None if out of range."""
    stock, mkt = ret[ticker], ret[benchmark]
    i0, ir = event_positions(ret.index, ev.announce_date, ev.hour,
                             cfg.RELEASE_TIMING.get(ticker, "yahoo"))
    if ir + max(b for _, b in cfg.WINDOWS.values()) >= len(ret):
        return None
    model = fit_market_model(stock, mkt, ir)
    if model is None:
        return None
    alpha, beta, sigma = model
    row = {"ticker": ticker, "announce_date": ev.announce_date,
           "reaction_date": ret.index[ir], "surprise_pct": ev.surprise_pct,
           "surprise_w": float(np.clip(ev.surprise_pct, -cfg.SURPRISE_CAP, cfg.SURPRISE_CAP)),
           "beta": beta}
    for name, (a, b) in cfg.WINDOWS.items():
        s, m = stock.iloc[ir + a: ir + b + 1], mkt.iloc[ir + a: ir + b + 1]
        ar = s - (alpha + beta * m)                   # abnormal return per day
        row[f"raw_{name}"] = (1 + s).prod() - 1
        row[f"mktadj_{name}"] = (1 + s).prod() - (1 + m).prod()
        row[f"car_{name}"] = ar.sum()                 # cumulative abnormal return
        row[f"t_{name}"] = ar.sum() / (sigma * np.sqrt(len(ar)))
    s, m = stock.iloc[i0: i0 + 2], mkt.iloc[i0: i0 + 2]
    row["car_0_1"] = (s - (alpha + beta * m)).sum()   # window that does not depend on timing
    s, m = stock.iloc[i0 - cfg.RUNUP_DAYS: i0], mkt.iloc[i0 - cfg.RUNUP_DAYS: i0]
    row["pre60"] = (1 + s).prod() - (1 + m).prod()    # market-adjusted run-up before the release
    return row


def build_events(ret, earnings: dict, benchmark: str) -> pd.DataFrame:
    """Event table for several stocks. `earnings` maps ticker -> DataFrame from data.load_earnings."""
    rows = []
    for ticker, df in earnings.items():
        for ev in df.itertuples():
            row = event_row(ticker, ev, ret, benchmark)
            if row:
                rows.append(row)
    return (pd.DataFrame(rows).sort_values(["ticker", "reaction_date"])
            .reset_index(drop=True))


def reaction_summary(events, tickers):
    """Beat rate, surprise and average reaction size per stock."""
    g = events.groupby("ticker")
    return pd.DataFrame({
        "events": g.size(),
        "beat rate": g["surprise_pct"].apply(lambda s: (s > 0).mean()),
        "mean surprise %": g["surprise_pct"].mean(),
        "mean CAR 1d": g["car_1d"].mean(),
        "mean |CAR| 1d": g["car_1d"].apply(lambda s: s.abs().mean()),
        "mean |CAR| 0-1": g["car_0_1"].apply(lambda s: s.abs().mean()),
    }).reindex(tickers)


def earnings_day_test(ret, events, tickers, benchmark):
    """Are absolute abnormal returns larger on reaction days than on other days? (Welch t-test)"""
    rows = []
    for t in tickers:
        beta, alpha = np.polyfit(ret[benchmark], ret[t], 1)
        ar = ret[t] - alpha - beta * ret[benchmark]
        is_ev = ar.index.isin(events.loc[events.ticker == t, "reaction_date"])
        on, off = ar[is_ev].abs(), ar[~is_ev].abs()
        tstat, p = stats.ttest_ind(on, off, equal_var=False)
        rows.append({"ticker": t, "earnings day": on.mean(), "other days": off.mean(),
                     "n": len(on), "t": tstat, "p": p})
    return pd.DataFrame(rows).set_index("ticker")


def regress_car(events, xcol, tickers, core):
    """CAR[0,+1] on `xcol` for all events, peers only, core only, and each stock."""
    groups = [("all", events),
              ("peers only", events[~events.ticker.isin(core)]),
              ("core only", events[events.ticker.isin(core)])]
    groups += [(t, events[events.ticker == t]) for t in tickers]
    rows = []
    for label, d in groups:
        if len(d) < 5:
            continue
        res = stats.linregress(d[xcol], d["car_0_1"])
        rho, p_rho = stats.spearmanr(d[xcol], d["car_0_1"])
        rows.append({"group": label, "n": len(d), "slope": res.slope,
                     "R2": res.rvalue ** 2, "p": res.pvalue,
                     "spearman rho": rho, "rho p": p_rho})
    return pd.DataFrame(rows).set_index("group")


def gap_decomposition(ret, events, a, b, window=2, last=cfg.TRADING_DAYS):
    """How much of the a-minus-b return gap accrues in the windows around a's and b's earnings."""
    gap = ret[a] - ret[b]
    mask = pd.Series(False, index=ret.index)
    for r in events[events.ticker.isin([a, b])].itertuples():
        i = ret.index.searchsorted(r.announce_date)
        mask.iloc[i: i + window] = True
    recent = ret.index.isin(ret.index[-last:])
    return {
        "event days, share of all days": mask.mean(),
        "event days, share of squared gap": (gap[mask] ** 2).sum() / (gap ** 2).sum(),
        "gap summed in event windows": gap[mask].sum(),
        "gap summed on other days": gap[~mask].sum(),
        "last year: gap total": gap.iloc[-last:].sum(),
        "last year: tracking error": gap.iloc[-last:].std() * np.sqrt(cfg.TRADING_DAYS),
        "last year: gap in event windows": gap[mask & recent].sum(),
        "last year: gap on other days": gap[~mask & recent].sum(),
    }


def latest_vs_history(events, tickers):
    """Latest reaction of each stock against its own history of |CAR[0,+1]|."""
    rows = []
    for t in tickers:
        d = events[events.ticker == t]
        latest = d.iloc[-1]
        rows.append({"ticker": t, "reaction date": latest.reaction_date,
                     "CAR[0,+1]": latest.car_0_1, "surprise %": latest.surprise_pct,
                     "bigger than share of past": (d["car_0_1"].abs() < abs(latest.car_0_1)).mean()})
    return pd.DataFrame(rows).set_index("ticker")


def largest_gap_days(ret, a, b, n=5):
    """The n days with the largest absolute return gap between a and b.

    Descriptive only: picking extreme days after the fact is not a test.
    """
    gap = (ret[a] - ret[b]).rename("gap")
    top = gap.abs().sort_values(ascending=False).head(n).index
    return ret.loc[top, [a, b]].join(gap)


def event_time_profile(ret, earnings, tickers, benchmark, days=range(-3, 6)):
    """Mean |abnormal return| for each trading day relative to the announcement date.

    Day 0 is the announcement date itself, not the assumed reaction day, so the
    profile does not depend on the before-open / after-close assumption: stocks that
    report after the close should peak on day +1, those that report before the open on day 0.
    """
    out = {}
    for t in tickers:
        stock, mkt = ret[t], ret[benchmark]
        rows = []
        for ev in earnings[t].itertuples():
            i0 = ret.index.searchsorted(ev.announce_date)
            model = fit_market_model(stock, mkt, i0)
            if model is None or i0 + max(days) >= len(ret):
                continue
            alpha, beta, _ = model
            rows.append({k: abs(stock.iloc[i0 + k] - (alpha + beta * mkt.iloc[i0 + k]))
                         for k in days})
        out[t] = pd.DataFrame(rows).mean()
    return pd.DataFrame(out)
