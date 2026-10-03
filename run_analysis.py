"""Run the comparison set in config.py: risk profile, then earnings event study.

    python run_analysis.py

Reads settings from config.py, caches downloads in data/raw/, and writes tables
and charts to outputs/.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
import data
import event_study as es
import risk

OUT = Path("outputs")


def run_risk():
    a, b, bench = cfg.TICKER_A, cfg.TICKER_B, cfg.BENCHMARK
    ret = risk.simple_returns(data.load_prices([a, b, bench], cfg.RISK_START, cfg.RISK_END))
    rf, rf_note = data.load_risk_free_rate(cfg.RISK_START, cfg.RISK_END)

    table = risk.risk_table(ret, bench, rf)
    pretty = table.copy()
    for col in pretty.columns:
        fmt = "{:.2f}" if col.startswith(("Beta", "Sharpe", "Sortino")) else "{:.2%}"
        pretty[col] = pretty[col].map(fmt.format)

    print(f"\nPeriod: {ret.index[0]:%Y-%m-%d} to {ret.index[-1]:%Y-%m-%d} ({len(ret)} trading days)")
    print(f"Risk-free rate: {rf:.2%} ({rf_note})")
    print("\nRisk summary")
    print(pretty.T.to_string())
    print("\nCorrelation of daily returns")
    print(ret.corr().round(2).to_string())

    gap = ret[a] - ret[b]
    print(f"\n{a} minus {b}: summed daily gap {gap.sum():.2%}, "
          f"annualised tracking error {gap.std() * np.sqrt(cfg.TRADING_DAYS):.2%}")
    table.round(4).to_csv(OUT / "risk_summary.csv")

    cum = (1 + ret).cumprod() - 1
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    (cum * 100).plot(ax=ax[0, 0], title="Cumulative return (%)")
    (risk.drawdown(ret) * 100).plot(ax=ax[0, 1], title="Drawdown from peak (%)")
    ((1 + ret[a]).cumprod() / (1 + ret[b]).cumprod()).plot(
        ax=ax[0, 2], title=f"Relative performance, {a} / {b} (start = 1)")
    (ret[[a, b]].rolling(cfg.ROLL_WINDOW).std() * np.sqrt(cfg.TRADING_DAYS) * 100).plot(
        ax=ax[1, 0], title=f"Rolling {cfg.ROLL_WINDOW}-day annualised volatility (%)")
    for t in (a, b):
        risk.rolling_beta(ret[t], ret[bench], cfg.ROLL_WINDOW).plot(
            ax=ax[1, 1], label=t, title=f"Rolling {cfg.ROLL_WINDOW}-day beta vs {bench}")
    ax[1, 1].legend()
    ret[a].rolling(cfg.ROLL_WINDOW).corr(ret[b]).plot(
        ax=ax[1, 2], title=f"Rolling {cfg.ROLL_WINDOW}-day correlation, {a} vs {b}")
    plt.tight_layout()
    plt.savefig(OUT / "risk_chart.png", dpi=150)


def run_events():
    a, b, bench = cfg.TICKER_A, cfg.TICKER_B, cfg.BENCHMARK
    core = [a, b]
    wanted = core + [p for p in cfg.PEERS if p not in core]
    ret = risk.simple_returns(data.load_prices(wanted + [bench], cfg.EVENT_START, cfg.EVENT_END))

    earnings = {}
    for t in wanted:
        try:
            earnings[t] = data.load_earnings(t)
        except Exception as e:
            print(f"Skipping {t}: earnings dates unavailable ({e})")
    events = es.build_events(ret, earnings, bench)
    events.round({c: 4 for c in events.select_dtypes("number")}).to_csv(
        OUT / "events.csv", index=False)
    tickers = [t for t in wanted if t in set(events["ticker"])]

    print("\nEarnings reactions (market-model abnormal returns)")
    print(es.reaction_summary(events, tickers).round(3).to_string())

    print("\nH1: absolute abnormal return, earnings day vs other days")
    h1 = es.earnings_day_test(ret, events, tickers, bench)
    print(h1.round({"earnings day": 4, "other days": 4, "t": 2, "p": 3}).to_string())

    for title, xcol in [("H2: CAR[0,+1] on EPS surprise (predicted slope > 0)", "surprise_w"),
                        ("H3: CAR[0,+1] on 60-day market-adjusted run-up (predicted slope < 0)", "pre60")]:
        print(f"\n{title}")
        print(es.regress_car(events, xcol, tickers, core).round(3).to_string())

    gd = es.gap_decomposition(ret, events, a, b)
    print(f"\nWhere does the {a}-vs-{b} gap come from?")
    for k, v in gd.items():
        print(f"   {k}: {v:.2%}")
    te = gd["last year: tracking error"]
    print(f"   last year: gap in tracking-error units: {gd['last year: gap total'] / te:.2f}")

    print(f"\nLargest one-day {a}-vs-{b} gaps over the last year, in % "
          "(descriptive, picked after the fact; not a test)")
    print((es.largest_gap_days(ret.iloc[-cfg.TRADING_DAYS:], a, b) * 100).round(2).to_string())

    print("\nLatest reaction vs each stock's own history of |CAR[0,+1]|")
    lv = es.latest_vs_history(events, tickers)
    print(lv.round({c: 4 for c in lv.select_dtypes("number")}).to_string())

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, t in enumerate(tickers):
        d = events[events.ticker == t]
        ax[0].scatter(d.surprise_w, d.car_0_1 * 100, label=t, color=colors[i], alpha=0.8)
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set(title="Earnings reaction vs EPS surprise",
              xlabel="EPS surprise (%, winsorised)", ylabel="CAR 0 to +1 (%)")
    ax[0].legend()
    mean_abs = (events.groupby("ticker")["car_0_1"]
                .apply(lambda s: s.abs().mean() * 100).reindex(tickers))
    ax[1].bar(mean_abs.index, mean_abs.values, color=colors[:len(tickers)])
    ax[1].set(title="Mean absolute reaction by stock", ylabel="mean |CAR 0 to +1| (%)")
    plt.tight_layout()
    plt.savefig(OUT / "event_study.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for t in core:
        risk.rolling_beta(ret[t], ret[bench], cfg.BETA_WINDOW).plot(ax=ax, label=t)
    ax.axhline(1, color="gray", lw=0.5)
    ax.set(title=f"Rolling {cfg.BETA_WINDOW}-day beta vs {bench}", ylabel="beta")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "rolling_beta.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    run_risk()
    run_events()
    print("\nSaved tables and charts to outputs/")
