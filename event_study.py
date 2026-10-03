"""
Earnings event study: payments stocks (V and MA, plus four peers).

Question: how much of a stock's move around earnings is explained by the EPS
surprise or by its run-up into the print, and how much of the V-vs-MA
performance gap comes from earnings windows?

Choices fixed BEFORE looking at results (to avoid p-hacking):
  - Market model: stock return = alpha + beta * SPY return, estimated on the
    250 trading days that end 30 trading days before each reaction day.
  - Primary window is [0, +1] from the announcement date, which is robust to
    before-open vs after-close timing. 1/3/5-day windows are secondary.
  - Surprise = (reported EPS - consensus EPS) / |consensus EPS|, winsorised at
    +/-50 percentage points. Spearman rank correlation is the robustness check.
  - H2: reaction rises with the surprise (slope > 0).
  - H3: reaction falls with the market-adjusted run-up over the prior 60
    trading days (slope < 0).
  - V and MA were analysed first; the peers are a replication on new data, so
    "peers only" is the cleanest test.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORE = ["V", "MA"]
PEERS = ["AXP", "PYPL", "FIS", "GPN"]
TICKERS = CORE + PEERS
MARKET = "SPY"
START, END = "2019-06-01", "2026-10-03"
EST_LEN, EST_GAP = 250, 30
SURPRISE_CAP = 50
# (first day, last day) in trading days, relative to the reaction day
WINDOWS = {"1d": (0, 0), "3d": (0, 2), "5d": (0, 4)}


def load_prices():
    px = yf.download(TICKERS + [MARKET], start=START, end=END,
                     auto_adjust=True, progress=False)["Close"]
    return px.dropna()


def load_events(ticker):
    """Past earnings releases with consensus EPS, reported EPS and timestamp."""
    df = yf.Ticker(ticker).get_earnings_dates(limit=40)
    df = df.dropna(subset=["Reported EPS", "EPS Estimate"]).copy()
    df["surprise_pct"] = ((df["Reported EPS"] - df["EPS Estimate"])
                          / df["EPS Estimate"].abs() * 100)
    local = df.index.tz_localize(None)       # keep New York wall-clock time
    df["announce_date"] = local.normalize()
    df["hour"] = local.hour
    return df[["announce_date", "hour", "surprise_pct"]].reset_index(drop=True)


def fit_market_model(stock, mkt, i_event):
    """alpha, beta and residual std from the window before the event."""
    end = i_event - EST_GAP
    start = end - EST_LEN
    if start < 0:
        return None
    x, y = mkt.iloc[start:end].values, stock.iloc[start:end].values
    beta, alpha = np.polyfit(x, y, 1)
    sigma = (y - alpha - beta * x).std(ddof=2)
    return alpha, beta, sigma


def event_row(ticker, ev, ret):
    stock, mkt = ret[ticker], ret[MARKET]
    i0 = ret.index.searchsorted(ev.announce_date)   # first trading day >= announcement
    ir = i0 + (1 if ev.hour >= 16 else 0)           # after-close reports react next day
    if ir + max(b for _, b in WINDOWS.values()) >= len(ret):
        return None
    model = fit_market_model(stock, mkt, ir)
    if model is None:
        return None
    alpha, beta, sigma = model
    row = {"ticker": ticker, "announce_date": ev.announce_date,
           "reaction_date": ret.index[ir], "surprise_pct": ev.surprise_pct,
           "surprise_w": float(np.clip(ev.surprise_pct, -SURPRISE_CAP, SURPRISE_CAP)),
           "beta": beta}
    for name, (a, b) in WINDOWS.items():
        s, m = stock.iloc[ir + a: ir + b + 1], mkt.iloc[ir + a: ir + b + 1]
        ar = s - (alpha + beta * m)                   # abnormal return per day
        row[f"raw_{name}"] = (1 + s).prod() - 1
        row[f"mktadj_{name}"] = (1 + s).prod() - (1 + m).prod()
        row[f"car_{name}"] = ar.sum()                 # cumulative abnormal return
        row[f"t_{name}"] = ar.sum() / (sigma * np.sqrt(len(ar)))
    s, m = stock.iloc[i0: i0 + 2], mkt.iloc[i0: i0 + 2]
    row["car_0_1"] = (s - (alpha + beta * m)).sum()   # timing-robust primary window
    s, m = stock.iloc[i0 - 60: i0], mkt.iloc[i0 - 60: i0]
    row["pre60"] = (1 + s).prod() - (1 + m).prod()    # market-adjusted run-up, 60 days before
    return row


def report_regression(title, events, tickers, xcol):
    """CAR[0,+1] on xcol for several groups, with a rank-correlation check."""
    print(f"\n{title}")
    groups = [("all", events),
              ("peers only", events[~events.ticker.isin(CORE)]),
              ("V+MA only", events[events.ticker.isin(CORE)])]
    groups += [(t, events[events.ticker == t]) for t in tickers]
    for label, d in groups:
        if len(d) < 5:
            continue
        res = stats.linregress(d[xcol], d["car_0_1"])
        rho, p_rho = stats.spearmanr(d[xcol], d["car_0_1"])
        print(f"{label:>10}: n={len(d):>3}  slope={res.slope:8.4f}  "
              f"R2={res.rvalue ** 2:.3f}  p={res.pvalue:.3f}  |  "
              f"spearman rho={rho:6.3f}  p={p_rho:.3f}")


def main():
    ret = load_prices().pct_change().dropna()

    rows = []
    for t in TICKERS:
        if t not in ret.columns:
            print(f"Skipping {t}: no price data")
            continue
        try:
            evs = load_events(t)
        except Exception as e:
            print(f"Skipping {t}: earnings dates unavailable ({e})")
            continue
        for ev in evs.itertuples():
            r = event_row(t, ev, ret)
            if r:
                rows.append(r)
    events = (pd.DataFrame(rows).sort_values(["ticker", "reaction_date"])
              .reset_index(drop=True))
    events.round(4).to_csv("events.csv", index=False)
    tickers = [t for t in TICKERS if t in set(events["ticker"])]

    # 0. Is the one-year V-vs-MA gap actually surprising?
    gap = ret["V"] - ret["MA"]
    last = gap.iloc[-252:]
    te = last.std() * np.sqrt(252)
    print(f"\n1-year gap (V minus MA, summed daily): {last.sum():.2%}")
    print(f"Annualised tracking error of the gap:  {te:.2%}")
    print(f"Gap in tracking-error units:           {last.sum() / te:.2f}")

    # 1. Reaction summary
    g = events.groupby("ticker")
    summary = pd.DataFrame({
        "events": g.size(),
        "beat rate": g["surprise_pct"].apply(lambda s: (s > 0).mean()),
        "mean surprise %": g["surprise_pct"].mean(),
        "mean CAR 1d": g["car_1d"].mean(),
        "mean |CAR| 1d": g["car_1d"].apply(lambda s: s.abs().mean()),
        "mean |CAR| 0-1": g["car_0_1"].apply(lambda s: s.abs().mean()),
    }).reindex(tickers)
    print("\nEarnings reactions (market-model abnormal returns)")
    print(summary.round(3).to_string())

    # H1. Are earnings-day moves bigger than normal days?
    print("\nH1: earnings-day abnormal moves vs normal days")
    for t in tickers:
        beta, alpha = np.polyfit(ret[MARKET], ret[t], 1)
        ar = ret[t] - alpha - beta * ret[MARKET]
        is_ev = ar.index.isin(events.loc[events.ticker == t, "reaction_date"])
        on, off = ar[is_ev].abs(), ar[~is_ev].abs()
        tstat, p = stats.ttest_ind(on, off, equal_var=False)
        print(f"{t:>5}: |AR| on earnings day {on.mean():.2%} vs other days "
              f"{off.mean():.2%}  (n={len(on)}, t={tstat:.2f}, p={p:.3f})")

    report_regression("H2: CAR[0,+1] on EPS surprise (predicted slope > 0)",
                      events, tickers, "surprise_w")
    report_regression("H3: CAR[0,+1] on 60-day market-adjusted run-up (predicted slope < 0)",
                      events, tickers, "pre60")

    # 4. Where does the V-vs-MA gap come from? (V and MA releases only)
    mask = pd.Series(False, index=ret.index)
    for r in events[events.ticker.isin(CORE)].itertuples():
        i = ret.index.searchsorted(r.announce_date)
        mask.iloc[i: i + 2] = True
    print(f"\nV/MA earnings-window days are {mask.mean():.1%} of all days but "
          f"{(gap[mask] ** 2).sum() / (gap ** 2).sum():.1%} of the gap's squared moves")
    print(f"Gap summed over earnings windows: {gap[mask].sum():.2%}; "
          f"over all other days: {gap[~mask].sum():.2%}")
    in_last_year = ret.index.isin(ret.index[-252:])
    print(f"Last year only: gap in earnings windows {gap[mask & in_last_year].sum():.2%}; "
          f"other days {gap[~mask & in_last_year].sum():.2%}")

    # 5. Latest event vs own history
    print("\nLatest reaction vs each stock's own history of |CAR[0,+1]|")
    for t in tickers:
        d = events[events.ticker == t]
        latest = d.iloc[-1]
        pct = (d["car_0_1"].abs() < abs(latest.car_0_1)).mean()
        print(f"{t:>5} {latest.reaction_date:%Y-%m-%d}: CAR {latest.car_0_1:.2%}, "
              f"surprise {latest.surprise_pct:.1f}%, bigger than {pct:.0%} of its reactions")

    # Chart
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, t in enumerate(tickers):
        d = events[events.ticker == t]
        ax[0].scatter(d.surprise_w, d.car_0_1 * 100, label=t,
                      color=colors[i], alpha=0.8)
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set(title="Earnings reaction vs EPS surprise",
              xlabel="EPS surprise (%, winsorised)", ylabel="CAR 0 to +1 (%)")
    ax[0].legend()
    mean_abs = (events.groupby("ticker")["car_0_1"]
                .apply(lambda s: s.abs().mean() * 100).reindex(tickers))
    ax[1].bar(mean_abs.index, mean_abs.values, color=colors[:len(tickers)])
    ax[1].set(title="Mean absolute reaction by stock", ylabel="mean |CAR 0 to +1| (%)")
    plt.tight_layout()
    plt.savefig("event_study.png", dpi=150)
    print("\nSaved events.csv and event_study.png")


if __name__ == "__main__":
    main()
