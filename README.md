## Pre-registered hypothesis (written before running the test)
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
The one-year V-minus-MA gap (about 8 points, summed daily returns) is ordinary for this pair: it is about 0.6 standard deviations of their annualised tracking error (11.7%). The largest single-day gap was Visa's reaction on 2026-04-29 (+8.26% vs. MA +3.47%) after it reported fiscal Q2 on 2026-04-28 after the close. Visa beat consensus adjusted EPS ($3.31 vs. $3.10), grew revenue 17%, raised full-year guidance, and announced a $20B buyback. But earnings windows did not drive the full-year gap: V and MA earnings windows contributed +1.9 points of the +7.5, and other days +5.6, because V's April gain was partly offset by MA outperforming V by about 2.8 and 3.2 points on its own earnings days in January and July. MA beat Q1 consensus (adjusted EPS $4.60 vs. $4.41) but fell on 2026-04-30, apparently because April-to-date data showed slowing cross-border growth. The explanations for the two April moves come from news coverage and have not been checked against primary filings.

## Limitations
- One year of data is a small sample.
- Yahoo Finance data is not institutional quality.
- Volatility assumes returns are roughly normal. Real returns have fatter tails.
- Historical risk does not predict future risk.

## Run it
pip install yfinance pandas matplotlib
python analyzer.py

## Event study: what moves V and MA around earnings?
Market-model abnormal returns (beta vs. SPY estimated on the 250 trading days ending 30 days before each release). Primary window is [0, +1] from the announcement date. Windows and tests were fixed before running. Sample: 25 releases per stock.

- The one-year V-minus-MA gap (7.5%, summed daily) is about 0.6 standard deviations of the pair's annualised tracking error (11.7%), so a gap that size is ordinary for two stocks with 0.86 correlation.
- Earnings reactions are large relative to normal days: mean absolute abnormal return on the reaction day was 2.56% (V) and 2.00% (MA) vs 0.78% and 0.87% on other days (Welch t-test on absolute values, p = 0.001 and p < 0.001).
- The EPS surprise explains little of the reaction: R2 = 0.03 (pooled, n = 50, p = 0.23). Both stocks beat consensus in 24 of 25 quarters, so the surprise has almost no variation, and the sample is too small to rule out a modest effect.
- Earnings windows are 4.8% of trading days but 19% of the gap's squared daily moves.

## Pre-registered hypothesis (written before running the test)
H3: the earnings reaction (CAR[0,+1]) is negatively related to the stock's market-adjusted return over the 60 trading days before the announcement. Primary test: pooled regression across both stocks, predicted slope < 0. The per-stock regressions are secondary.

**H3 result (run after the hypothesis above was committed): not supported.** Pooled slope = +0.067 (R2 = 0.02, p = 0.35, n = 50), the opposite sign from the prediction and statistically indistinguishable from zero. Per-stock slopes: V +0.17 (p = 0.13), MA -0.01 (p = 0.90). With 50 events this test cannot rule out a modest effect, but it gives no evidence that a weak run-up leads to a bigger positive reaction.

Taken together: earnings days move these stocks far more than normal days (H1), but neither the EPS surprise (H2) nor the prior run-up (H3) predicts the direction of the move in this sample. Guidance and current-quarter trends may matter more, but this data does not measure them and I have not tested that.

## Pre-registered replication on peers (written before running it)
V and MA were analysed first. I am now adding four peers (AXP, PYPL, FIS, GPN) as a replication on new data. Primary tests: H2 (reaction rises with EPS surprise, slope > 0) and H3 (reaction falls with the prior 60-day market-adjusted run-up, slope < 0), on the peers-only sample. EPS surprise is winsorised at +/-50 percentage points and Spearman rank correlation is the robustness check. I will not change windows or the model after seeing results. Events in the same quarter share market conditions, so p-values on pooled samples are optimistic.

## Peer replication results
Six stocks (V, MA, AXP, PYPL, FIS, GPN), 25 earnings releases each, 150 events.

- H1 replicates in every stock: mean absolute abnormal return on the reaction day is 2.0-7.9% vs. 0.8-1.4% on other days (p <= 0.002 in each). The mean absolute reaction over [0, +1] is largest for PYPL (8.3%), FIS (6.0%) and GPN (5.6%) and smallest for V (2.6%) and MA (2.4%).
- H2 not supported: peers-only slope = +0.0007 per percentage point of EPS surprise (R2 = 0.01, p = 0.33; Spearman rho = 0.05, p = 0.65).
- H3 not supported: peers-only slope = +0.023 (R2 = 0.001, p = 0.71), the opposite sign from the prediction.
- About 36 p-values are reported across groups and tests. One falls below 0.05 (V, Spearman for H3, p = 0.03, wrong sign), roughly what chance alone would produce, so I do not treat it as a finding.
- In this sample the typical size of an earnings reaction differs a lot by stock, but neither the EPS surprise nor the prior run-up predicts its direction.

Caveats: Yahoo consensus estimates are a rough proxy (AXP's mean surprise of 36% reflects outliers, which is why surprise is winsorised at +/-50 pp); events in the same quarter share market conditions, so pooled p-values are optimistic; 25 events per stock is small.
