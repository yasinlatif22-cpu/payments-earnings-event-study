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
Despite a 0.86 return correlation, V and MA finished the year about 8 percentage points apart (V +3.92%, MA -4.26%), and their worst drawdowns came at different times. V rallied sharply around May 2026 while MA kept sliding for another month. A next step is to check what drove V's move (earnings, guidance, or news) and why MA didn't follow.

## Limitations
- One year of data is a small sample.
- Yahoo Finance data is not institutional quality.
- Volatility assumes returns are roughly normal. Real returns have fatter tails.
- Historical risk does not predict future risk.

## Run it
pip install yfinance pandas matplotlib
python analyzer.py

