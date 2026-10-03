"""Tests for the core calculations. Run with: pytest -q

Each expected value can be checked by hand.
"""
import numpy as np
import pandas as pd
import pytest

import config as cfg
import data
import event_study as es
import risk


def prices_to_returns(values):
    return risk.simple_returns(pd.DataFrame({"X": values})).X


def test_simple_returns():
    r = prices_to_returns([100, 110, 99])
    assert r.tolist() == pytest.approx([0.10, -0.10])


def test_max_drawdown_known_path():
    # 100 -> 120 -> 90 -> 110: peak 120, trough 90, drawdown = 90/120 - 1 = -25%
    r = prices_to_returns([100, 120, 90, 110])
    assert risk.drawdown(r).min() == pytest.approx(-0.25)


def test_drawdown_counts_the_starting_dollar():
    # 100 -> 80 -> 90: the first peak is the starting $1, so the drawdown is -20%
    r = prices_to_returns([100, 80, 90])
    assert risk.drawdown(r).min() == pytest.approx(-0.20)


def test_total_return():
    r = pd.Series([0.10, -0.10])
    assert risk.total_return(r) == pytest.approx(1.10 * 0.90 - 1)


def test_volatility_zero_for_constant_returns():
    assert risk.annualised_vol(pd.Series([0.01] * 20)) == pytest.approx(0.0, abs=1e-12)


def test_volatility_scaling():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert risk.annualised_vol(r) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))


def test_downside_deviation_by_hand():
    # Only -0.01 and -0.03 count: mean of squares = (0.0001 + 0.0009) / 4 = 0.00025
    r = pd.Series([0.02, -0.01, -0.03, 0.04])
    assert risk.downside_deviation(r) == pytest.approx(np.sqrt(0.00025) * np.sqrt(252))


def test_beta_recovers_known_slope():
    x = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    assert risk.beta(2 * x, x) == pytest.approx(2.0)


def test_sharpe_by_hand():
    r = pd.Series([0.01, 0.03, -0.01, 0.02])
    expected = (r.mean() * 252 - 0.04) / (r.std() * np.sqrt(252))
    assert risk.sharpe(r, rf=0.04) == pytest.approx(expected)


def test_rolling_beta_matches_plain_beta_on_full_window():
    x = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
    y = 1.5 * x + 0.001
    assert risk.rolling_beta(y, x, window=len(x)).iloc[-1] == pytest.approx(1.5)


def test_surprise_pct():
    assert data.surprise_pct(1.10, 1.00) == pytest.approx(10.0)
    assert data.surprise_pct(-0.5, -1.0) == pytest.approx(50.0)   # a smaller loss is a beat


def test_event_positions():
    idx = pd.bdate_range("2024-01-01", periods=10)
    day = idx[4]
    assert es.event_positions(idx, day, 16, "yahoo") == (4, 5)         # Yahoo says after close
    assert es.event_positions(idx, day, 8, "yahoo") == (4, 4)          # Yahoo says before open
    assert es.event_positions(idx, day, 6, "after_close") == (4, 5)    # override beats Yahoo
    assert es.event_positions(idx, day, 16, "before_open") == (4, 4)


def test_market_model_recovers_alpha_and_beta():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0, 0.01, 400))
    stock = 0.0002 + 1.5 * mkt
    alpha, beta, sigma = es.fit_market_model(stock, mkt, i_event=350)
    assert alpha == pytest.approx(0.0002)
    assert beta == pytest.approx(1.5)
    assert sigma == pytest.approx(0.0, abs=1e-12)


def make_returns(jump_day=None, jump=0.0, n=420):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2022-01-03", periods=n)
    mkt = rng.normal(0, 0.01, n)
    stock = 1.2 * mkt
    if jump_day is not None:
        stock[jump_day] += jump
    return pd.DataFrame({"AAA": stock, "SPY": mkt}, index=idx)


def test_abnormal_return_picks_up_an_injected_jump():
    ret = make_returns(jump_day=351, jump=0.05)          # +5% on the reaction day
    ev = pd.DataFrame({"announce_date": [ret.index[350]], "hour": [16],
                       "surprise_pct": [5.0]}).itertuples().__next__()
    row = es.event_row("AAA", ev, ret, "SPY")             # unknown ticker -> trust Yahoo's hour (after close)
    assert row["reaction_date"] == ret.index[351]
    assert row["car_1d"] == pytest.approx(0.05, abs=1e-9)
    assert row["car_0_1"] == pytest.approx(0.05, abs=1e-9)   # the two-day window contains the jump
    assert row["raw_1d"] == pytest.approx(1.2 * ret["SPY"].iloc[351] + 0.05)


def test_no_jump_means_no_abnormal_return():
    ret = make_returns()
    ev = pd.DataFrame({"announce_date": [ret.index[350]], "hour": [8],
                       "surprise_pct": [0.0]}).itertuples().__next__()
    row = es.event_row("AAA", ev, ret, "SPY")
    assert row["car_1d"] == pytest.approx(0.0, abs=1e-9)
    assert row["car_3d"] == pytest.approx(0.0, abs=1e-9)


def test_surprise_is_winsorised():
    ret = make_returns()
    ev = pd.DataFrame({"announce_date": [ret.index[350]], "hour": [8],
                       "surprise_pct": [400.0]}).itertuples().__next__()
    assert es.event_row("AAA", ev, ret, "SPY")["surprise_w"] == cfg.SURPRISE_CAP


def test_largest_gap_days():
    idx = pd.bdate_range("2024-01-01", periods=4)
    ret = pd.DataFrame({"A": [0.01, 0.05, 0.00, -0.01], "B": [0.01, 0.00, 0.00, 0.02]}, index=idx)
    top = es.largest_gap_days(ret, "A", "B", n=2)
    assert list(top.index) == [idx[1], idx[3]]           # gaps of 0.05 and -0.03
    assert top["gap"].tolist() == pytest.approx([0.05, -0.03])
