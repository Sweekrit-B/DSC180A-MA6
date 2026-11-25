# Audit of a teammate's repository (Exercise 3)

Repository: https://github.com/lindbergryan04/DSC180A-MA6 at commit `82104b1`.
Diff reviewed: everything since their starter commit `b48e5e4`.
Auditor: the `/leakage-auditor` subagent from `.claude/agents/leakage-auditor.md`,
run on 2026-11-25 against a fresh clone. It was told not to modify the
clone, not to use the network, and not to touch `data/raw/`.

## Auditor output (verbatim)

```
Audit of lindbergryan04/DSC180A-MA6, the diff since starter commit b48e5e4

Checks 1 to 4 pass. The new `cnt_prior_week_same_type` feature masks rows by workday status and then applies `.shift(1)` before the 7-row rolling mean. It is built on the reindexed calendar, so the window covers 7 calendar days and 2011-03-10 shows up as an explicit NaN. I raised one day's `cnt` by 10,000 on every 5th date, and no feature changed on the same date. No feature restates `cnt`. Every estimator, including the StandardScaler inside the ridge pipeline, is cloned and fit only on 2011 rows. The split is by date. `dropna` loses exactly 14 rows: the 7 warm-up days plus the 7 days from 2011-03-11 to 2011-03-17. 2011-03-10 itself is absent from the raw file. `tests/test_contract.py::test_only_expected_rows_are_dropped` limits this loss. Two findings remain, both about baselines and how the test set is used in the conclusions.

**Finding 1 (check 5, unreported baseline): the Exercise 4 transformed booster is scored without either baseline, and on test MAE it loses to persistence.**
- **Where:** `src/train.py:84-91` (`change_from_yesterday`) and `src/train.py:99-100`. With `--predict-change`, the run returns only the booster's scores, with no mean or persistence baseline. `REPORT.md:30` then compares the result only to ridge: "Transformed boosting reached 0.5789 test R² and 874.0 rides MAE, compared with 0.5794 and 947.7 for ridge."
- **Scenario:** Persistence ("yesterday's count") scores 0.5099 R² and **840.1 MAE** on the same 184 test days. The transformed booster's MAE of 874.0 is therefore 33.9 rides per day *worse* than persistence. Its forecast is persistence plus a learned adjustment, so on absolute error the adjustment hurts on the holdout. The paragraph reads as though the booster is the strongest model. In fact it loses on MAE to the baseline it is built on. Exercise 5 uses the same MAE criterion to reject ridge, and by that criterion this booster would be rejected too. On validation it does beat persistence (827.2 vs 900.6 MAE).
- **Minimal fix:** Have `change_from_yesterday` also return `score_models(...)` output for `baseline_mean` and `baseline_persistence` on the same `evaluation` splits. Add the persistence row (0.5099 / 840.1) to `REPORT.md:30`, and state that the booster loses to persistence on test MAE.

**Finding 2 (check 2, test consulted for a conclusion): the Exercise 5 recommendation of persistence over ridge is chosen on test and reverses on validation.**
- **Where:** `REPORT.md:34`: "persistence is the better of these two candidates … the average absolute miss grows from 840.1 to 947.7 rides." The numbers come from `reports/metrics.json` → `baseline_persistence.test` and `ridge.test`.
- **Scenario:** On validation (H1 2012), using the same MAE criterion, ridge beats persistence: **830.6 vs 900.6**, 70.0 rides per day better. The ranking flips only on the test half. So the model choice is being made on the holdout. CLAUDE.md says "Never tune on test." The 840.1 and 947.7 figures can no longer serve as unbiased estimates for the chosen model, and the recommendation depends on which half of 2012 is used to make it.
- **Minimal fix:** Choose between ridge and persistence on validation MAE, which selects ridge. Then report the test MAE of the chosen model once. Alternatively, keep the paragraph but say plainly that the ranking reverses between validation and test, so the data does not support choosing persistence over ridge.

No other finding changes a number. I re-ran the pipeline from the committed raw CSV and got `reports/metrics.json` byte-identical. The REPORT.md table matches it, including both baselines for the before and after feature sets.

EVIDENCE:
- `cd /tmp/teammate-audit.MqFL/repo && .venv/bin/python -m pytest tests/ -q` → `17 passed in 0.65s`
- `sha256sum data/raw/bike_daily.csv` → `a6bed157…5ad9dc`, matching `RAW_SHA256` in `src/config.py`
- `python -m src.features` → `OK features: 714 rows, 15 features`. Then `python -m src.train`, followed by `git diff --stat reports/metrics.json` → empty, so the metrics reproduce exactly.
- `python -m src.train --predict-change` → valid `{"r2": 0.6104, "mae": 827.2}`, test `{"r2": 0.5789, "mae": 874.0}`. No baseline entries are in the output.
- `reports/metrics.json`: `baseline_persistence` valid r2 0.4507 / mae 900.6, test r2 0.5099 / mae 840.1. `ridge` valid 0.6459 / 830.6, test 0.5794 / 947.7. `baseline_mean` test -2.1441 / 2847.7.
- Perturbation script (adds 10,000 to `cnt` on every 5th date, rebuilds features, compares same-date features) → `same-day leaks 0`
- Dropped dates: 2011-01-03 to 2011-01-09 and 2011-03-10 to 2011-03-17 (2011-03-10 is the calendar gap) → 14 rows lost from 728
- `(casual + registered - cnt).abs().max()` → `0`. Neither column is in the feature list printed by `src.features`.
- `git status --short` → empty
```

(The auditor's `.venv/bin/python` stands for this repo's virtualenv interpreter,
run from inside the clone.)

## My verdict

Neither finding is leakage, and neither changes a number in their Results
table, which the auditor reproduced exactly. I checked the figures both
findings rest on against their committed `reports/metrics.json` and they
match. I agree with Finding 2, and it is the one that changes a conclusion.
Their Exercise 5 recommends persistence because its test MAE is lower (840.1
vs 947.7), but on validation ridge wins MAE (830.6 vs 900.6), so the
recommendation was made on the test set and would flip on validation. I
disagree only with the auditor's suggested fix: the exercise itself frames
the comparison in test numbers, so citing test MAE is fair, as long as the
paragraph says the ranking reverses on validation. I agree with Finding 1 as
a missing baseline. Their transformed booster beats ridge on test MAE (874.0
vs 947.7) but loses to persistence (840.1), so the Exercise 4 paragraph
overstates it, though again no reported number is wrong.
