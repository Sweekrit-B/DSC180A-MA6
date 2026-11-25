"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import pandas as pd
import pytest

from src.config import FORBIDDEN, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_temp_delta_3_uses_prior_calendar_days(feats):
    """temp_delta_3 on date d must equal temp[d] - mean(temp[d-3..d-1]),
    with the window counted in calendar days, never including d itself.

    Checked against the raw data, not the feature table, so a window that
    silently spans the missing 2011-03-10 cannot agree with itself."""
    # Named date, by hand from data/raw/bike_daily.csv. temp on 2012-06-20 is
    # 28.7775; on 06-17, 06-18, 06-19 it is 19.8475, 18.711651, 24.351651.
    # 28.7775 - (19.8475 + 18.711651 + 24.351651) / 3 = 7.807233
    named = feats.set_index("dteday").temp_delta_3[pd.Timestamp("2012-06-20")]
    assert named == pytest.approx(7.807233, abs=1e-6)

    raw_temp = load_raw().set_index("dteday").temp
    checked = 0
    for date, delta in feats.set_index("dteday").temp_delta_3.items():
        window = [date - pd.Timedelta(days=k) for k in range(1, 4)]
        assert all(d in raw_temp.index for d in window), (
            f"temp_delta_3 at {date.date()} has a window crossing a missing day"
        )
        expected = raw_temp[date] - sum(raw_temp[d] for d in window) / 3
        assert delta == pytest.approx(expected), f"bad temp_delta_3 at {date.date()}"
        checked += 1
    assert checked > 600, "too few rows actually verified"


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique
