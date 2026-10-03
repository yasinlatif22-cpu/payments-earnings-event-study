import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

prices = yf.download(["V", "MA"], period="1y", progress=False)["Close"]
returns = prices.pct_change().dropna()
returns["Gap"] = returns["V"] - returns["MA"]

# 1. Biggest single-day gaps between V and MA
print("\nBiggest one-day gaps (daily returns, %)")
top_days = returns["Gap"].abs().sort_values(ascending=False).head(5).index
print(returns.loc[top_days].mul(100).round(2))

# 2. Cumulative relative performance of V vs MA, month-end snapshots
cum = (1 + returns[["V", "MA"]]).cumprod()
relative = (cum["V"] / cum["MA"] - 1) * 100
print("\nV outperformance vs MA, cumulative (%), month-end")
print(relative.resample("M").last().round(2))

# 3. Earnings dates
for ticker in ["V", "MA"]:
    print(f"\n{ticker} earnings dates")
    try:
        dates = yf.Ticker(ticker).get_earnings_dates(limit=8)
        print([d.strftime("%Y-%m-%d") for d in dates.index])
    except Exception as e:
        print("Could not fetch:", e)
