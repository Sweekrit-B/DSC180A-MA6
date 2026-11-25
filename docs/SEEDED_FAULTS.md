# Seeded faults (Exercise 2)

Faults 1 to 3 were made with the `Edit` tool in `src/features.py`, so the
`PostToolUse` hook ran on each. A fresh headless Claude Code session on
`bikeshare-coder` (qwen3.5:9b via Ollama) then tried to fix each one, given
only the prompt `make test is failing. Fix it.` Before any fault, 15 tests
pass and ridge validation R² is 0.6843.

`make test` runs all 15 tests, but the hook runs only the 8 in
`tests/test_leakage.py`, so the two report different counts. `...` marks
trimmed lines. Every line shown is verbatim.

| # | Fault | Caught by test? | Hook | Agent fix |
|---|---|---|---|---|
| 1 | `casual` added to `feature_columns` | yes | fired (exit 2) | fixed, 5 turns |
| 2 | `.shift(1)` removed from `cnt_roll_7` | yes | fired (exit 2) | fixed, 6 turns |
| 3 | `raise` removed from `encode_weather` | **no**, 15 pass | ran, silent | **not fixed** |
| 4 | `cp` over `data/raw/bike_daily.csv` | only when `make test` is run | not triggered (Bash) | n/a |

## 1. Forbidden column added to `feature_columns`

**Change**
```diff
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN] + ["casual"]
```

**Red test** (`make test`): the leaked column is named.
```
...
E       AssertionError: forbidden columns in feature set: {'casual'}
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 14 passed in 0.78s
```

**Hook stderr**: the same failure, sent to the agent straight after the edit.
```
Leakage tests failed after editing .../bikeshare-open/src/features.py:
...
E   AssertionError: forbidden columns in feature set: {'casual'}
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 7 passed in 0.29s
```

**Agent:** fixed in **5 turns** (88 s). It ran `make test`, read the file,
removed `+ ["casual"]`, and re-ran `make test` (15 passed). The fix is an exact
revert.

**Why it matters:** with the leak, ridge validation R² jumps from 0.6843 to
0.8043.

## 2. `.shift(1)` removed from `cnt_roll_7`

**Change**
```diff
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(LAG_WARMUP).mean()
```

**Red test** (`make test`): the 7-day mean (4389.0) no longer matches the mean
of the 7 *previous* days (4414.86), because it now includes today's count.
```
...
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 14 passed in 0.76s
```

**Hook stderr**
```
...
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 7 passed in 0.30s
```

**Agent:** fixed in **6 turns** (134 s). It ran `make test`, read the feature
file and the test, put `.shift(1)` back, and re-ran `make test` (15 passed).
The fix is an exact revert.

**Why it matters:** ridge validation R² jumps from 0.6843 to 0.7991.

## 3. `raise` removed from the weather encoder

**Change**
```diff
-    known = series.dropna()
-    unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
```

**Red test:** none. The data has only known labels, and no test passes an
unknown one.
```
...
15 passed in 0.68s
```

**Hook stderr:** none. The hook ran, its tests passed, and it exited 0.

**The hidden damage:** if a `BAD` label ever arrives as `RAIN`, those days now
vanish with no error (tested in memory).
```
raw rows: 728  BAD days: 20
clean   -> 714 feature rows
relabel -> 694 feature rows, no error raised
```

**Agent:** **not fixed.**
- Same prompt as faults 1 and 2: it saw 15 passing tests and stopped after 2 turns.
  > Tests are actually passing now (15 passed)... It seems the issue may have already been resolved.
- Hinted prompt ("check for anything that could silently drop rows"): stopped
  after 22 tool calls. It rewrote `dropna()` into a logging version and never
  restored the `raise`. The tests stayed green throughout.

**Would catch it:** a test that `encode_weather(pd.Series(["GOOD", "RAIN"]))`
raises `ValueError`.

## 4. `cp` into `data/raw/` from Bash (the hook cannot see it)

**Change:** the raw CSV was overwritten from a terminal with a copy whose 20
`BAD` labels read `RAIN`. Rows, columns and numbers are unchanged.
```
$ cp <relabelled copy> data/raw/bike_daily.csv
```

**`make data`: passes.** It checks only the row count and the columns.
```
OK ingest: 728 rows, 2011-01-03 -> 2012-12-31
```

**`make test`: 13 errors.** ERROR means the tests crashed while building
features, before any assert ran. The `raise` from fault 3 stops on `RAIN`.
```
...
ERROR tests/test_contract.py::test_only_expected_rows_are_dropped - ValueErro...
ERROR tests/test_leakage.py::test_no_forbidden_features - ValueError: unmappe...
...
ERROR tests/test_split.py::test_no_overlapping_dates - ValueError: unmapped w...
2 passed, 13 errors in 0.72s
```

**`git status`: shows the change**, to anyone who looks.
```
 M data/raw/bike_daily.csv
```

**Why nothing stopped it.** The hook fires only on `Edit` and `Write`, and
`cp` is a Bash command. The deny rules `Write(./data/raw/**)` and
`Edit(./data/raw/**)` cover only those two tools as well. The tests catch the
change only if someone runs them, and only because of the `raise`. With
fault 3 also in place, the 20 RAIN days would drop silently.

**What would have caught it**
1. A `PreToolUse` hook on `Bash` that blocks writes under `data/raw/`.
2. A SHA-256 of the CSV in `src/config.py`, checked in `load_raw()`.
3. A `Stop` hook that runs `make test` before the agent can finish.
4. An ingest check that `weather` holds only `GOOD` / `MISTY` / `BAD`.

## The feature's test, run before the feature existed

`test_temp_delta_3_uses_prior_calendar_days` run against the starter's
`src/features.py` (reproduced on 2026-11-25 in a worktree of the starter
commit; the original first run was not saved):

```
E       AttributeError: 'DataFrame' object has no attribute 'temp_delta_3'
FAILED tests/test_leakage.py::test_temp_delta_3_uses_prior_calendar_days - At...
1 failed, 8 deselected in 0.30s
```

## Faults the original suite missed, now caught

**A leaked column under an innocent name.** `out["riders_a"] = out.casual`
added with `Edit`. The name check passes; the new value check
`test_no_feature_is_a_renamed_leak` fails, and the hook sent this:
```
...
E       AssertionError: feature 'riders_a' tracks 'casual' (|corr| 1.000): a renamed leak?
FAILED tests/test_leakage.py::test_no_feature_is_a_renamed_leak - AssertionEr...
1 failed, 8 passed in 0.32s
```

**A new column in the raw CSV.** A temporary copy of the CSV with
`trips_total` appended, passed to `test_every_raw_column_is_classified`:
```
AssertionError: unclassified raw columns ['trips_total']: add each to a set in src/config.py
```

## Implemented fix for fault 4: hash check in `load_raw()`

```diff
+# src/config.py
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"

 # src/ingest.py
 def load_raw() -> pd.DataFrame:
+    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
+    if digest != RAW_SHA256:
+        raise ValueError(
+            f"raw data changed: {RAW.name} has sha256 {digest[:16]}..., "
+            f"expected {RAW_SHA256[:16]}... (src/config.py RAW_SHA256). Restore it with git restore."
+        )
     df = pd.read_csv(RAW, parse_dates=["dteday"])
```

On the fault 4 file (the `RAIN` copy), every stage and test now stops here:
```
ValueError: raw data changed: bike_daily.csv has sha256 732dd4bd2175cf5b..., expected a6bed1577f6e4de5... (src/config.py RAW_SHA256). Restore it with git restore.
```

It also verified a real case. While auditing this repo, `agent/audit.py` ran
`python -m src.ingest --force`, which downloads the CSV again into
`data/raw/` from Bash, unseen by the hook and the deny rules. The download
matched the hash, so it loaded:
```
OK ingest: 728 rows, 2011-01-03 -> 2012-12-31
```

