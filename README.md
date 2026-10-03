# Price & Risk Analyzer

Compares the risk profile of Visa (V) and Mastercard (MA) over the trailing year using Python.

## Question
How do two near-identical businesses differ in volatility, drawdown, and co-movement?

## Method
- Daily closing prices from Yahoo Finance via `yfinance`
- Daily returns and annualized volatility (daily std x sqrt(252))
- Max drawdown from compounded returns
- Return correlation matrix

## Results (trailing 1 year as of 2026-10-02)
| | Annual vol | Max drawdown |
|---|---|---|
| MA | 22.45% | -18.48% |
| V | 22.05% | -17.18% |

Return correlation: 0.86.

![Risk chart](risk_chart.png)

## Observations
Almost all of the roughly 8-point gap between V and MA came from two days. Visa reported fiscal Q2 after the close on 2026-04-28 and rose 8.26% the next day (MA +3.47%). On 2026-04-30, MA's own earnings day, MA fell 4.25% and V fell 1.50%. V's cumulative outperformance was under 3 points through March, reached 8.76 by the end of April, and stayed between about 6 and 11 afterward. Visa beat consensus adjusted EPS ($3.31 vs. $3.10), grew revenue 17%, raised full-year guidance, and announced a $20B buyback. I have not yet determined why MA fell on its own earnings day.

## Limitations
- One year of data is a small sample.
- Yahoo Finance data is not institutional quality.
- Volatility assumes returns are roughly normal. Real returns have fatter tails.
- Historical risk does not predict future risk.

## Run it
pip install yfinance pandas matplotlib
python analyzer.py

