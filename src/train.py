"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train
"""
import json

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    # Last week's average, unchanged. The anchor of the diff_roll7 target, so
    # any model trained on that target must beat this, not just persistence.
    "baseline_roll7": PersistenceBaseline("cnt_roll_7"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


# Target transforms (Exercise 4). Each is (forward: frame -> z, inverse: z,
# frame -> cnt). Models fit z and are scored on cnt. Baselines never use one.
# diff_roll7 predicts the departure from last week's average: a level-free
# target, so trees trained on 2011 are not capped at 2011's maximum. The
# anchor cnt_roll_7 is built with shift(1), so it is known at prediction time.
TARGET_TRANSFORMS = {
    "none": (lambda d: d[TARGET], lambda z, d: z),
    "diff_roll7": (lambda d: d[TARGET] - d["cnt_roll_7"], lambda z, d: z + d["cnt_roll_7"].to_numpy()),
}


def fit_and_score(target: str = "none", include_test: bool = True) -> dict:
    """include_test=False scores validation only, for model selection."""
    forward, inverse = TARGET_TRANSFORMS[target]
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    splits = [("valid", valid)] + ([("test", test)] if include_test else [])

    results = {
        "target": target,
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    for name, model in MODELS.items():
        fwd, inv = TARGET_TRANSFORMS["none"] if name.startswith("baseline_") else (forward, inverse)
        model.fit(train[cols], fwd(train))
        results[name] = {}
        for split, part in splits:
            pred = inv(model.predict(part[cols]), part)
            results[name][split] = {
                "r2": round(float(r2_score(part[TARGET], pred)), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], pred)), 1),
            }
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGET_TRANSFORMS), default="none",
                        help="target transform; only 'none' writes reports/metrics.json")
    args = parser.parse_args()
    res = fit_and_score(args.target)
    # The untransformed pipeline owns METRICS; a transformed run never overwrites it.
    out = METRICS if args.target == "none" else METRICS.with_name(f"metrics_{args.target}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"OK train: target={args.target} -> {out.name}; valid r2 "
          + ", ".join(f"{k} {v['valid']['r2']}" for k, v in res.items() if isinstance(v, dict)))
