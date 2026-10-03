"""Settings for one analysis run. Edit the tickers, then run: python run_analysis.py"""

TICKER_A = "V"
TICKER_B = "MA"
PEERS = ["AXP", "PYPL", "FIS", "GPN"]   # extra stocks for the event-study replication
BENCHMARK = "SPY"

# Risk analysis window (END is exclusive)
RISK_START, RISK_END = "2025-10-02", "2026-10-03"
# Event-study sample: long enough for ~25 earnings releases per stock
EVENT_START, EVENT_END = "2019-06-01", "2026-10-03"

TRADING_DAYS = 252
ROLL_WINDOW = 60          # trading days, for rolling volatility and correlation
BETA_WINDOW = 126         # trading days, for the long rolling-beta chart

# Event-study design, fixed before looking at results (see README)
EST_LEN, EST_GAP = 250, 30   # market model: 250 days ending 30 days before the reaction day
RUNUP_DAYS = 60              # prior run-up measured over 60 trading days
SURPRISE_CAP = 50            # winsorise EPS surprise at +/-50 percentage points
# (first day, last day) in trading days, relative to the reaction day
WINDOWS = {"1d": (0, 0), "3d": (0, 2), "5d": (0, 4)}

# When each company releases earnings: "after_close", "before_open", or "yahoo"
# (trust Yahoo's timestamp). V and MA were checked against their press releases.
RELEASE_TIMING = {"V": "after_close", "MA": "before_open"}
