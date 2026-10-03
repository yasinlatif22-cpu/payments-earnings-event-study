import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

tickers = ["V", "MA"]
prices = yf.download(tickers, period="1y", progress=False)["Close"]
returns = prices.pct_change().dropna()

cumulative = (1 + returns).cumprod()
drawdown = cumulative / cumulative.cummax() - 1

summary = pd.DataFrame({
    "Avg daily return": returns.mean(),
    "Daily vol": returns.std(),
    "Annual vol": returns.std() * (252 ** 0.5),
    "Max drawdown": drawdown.min(),
})

formatted = summary.copy()
for col in formatted.columns:
    formatted[col] = formatted[col].map("{:.2%}".format)

print("\nLatest prices")
print(prices.tail().round(2))

print("\nRisk summary")
print(formatted)

print("\nCorrelation")
print(returns.corr().round(2))

summary.to_csv("summary.csv")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

(cumulative - 1).mul(100).plot(ax=axes[0])
axes[0].set_title("Cumulative return (%)")
axes[0].set_ylabel("%")

drawdown.mul(100).plot(ax=axes[1])
axes[1].set_title("Drawdown from peak (%)")
axes[1].set_ylabel("%")

plt.tight_layout()
plt.savefig("risk_chart.png", dpi=150)
print("\nSaved risk_chart.png")
