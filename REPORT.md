# Bikeshare demand: report

## Results

### Feature: `temp_delta_3`

`temp_delta_3` = temp[d] − mean(temp[d−3..d−1]), today's temperature relative
to the previous three calendar days. Same-day temperature is known at forecast
time, like the other weather columns. The three-day baseline is `.shift(1)` on
the gap-free calendar index, so it uses only earlier days. The test is
`test_temp_delta_3_uses_prior_calendar_days` in `tests/test_leakage.py`.

Train is 2011 (348 days), validation H1 2012 (182 days), test H2 2012 (184
days). The feature was chosen and kept on validation alone. Test scores were
read only for this final table. "After" rows are `reports/metrics.json` as
printed by `/metrics-table`. "Before" rows are the same pipeline at the
starter commit.

| Model | Features | valid R² | valid MAE | test R² | test MAE |
|---|---|---|---|---|---|
| baseline_mean | before | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| baseline_mean | after | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| baseline_persistence | before | 0.4507 | 900.6 | 0.5099 | 840.1 |
| baseline_persistence | after | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | before | 0.6475 | 832.9 | 0.5833 | 947.1 |
| ridge | after | 0.6843 | 767.4 | 0.6271 | 873.1 |
| hgb | before | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| hgb | after | 0.1722 | 1335.7 | -0.5648 | 2021.1 |

I kept the feature. On validation, ridge, the best model, gains 0.0368 R²,
above the 0.01 noise threshold for a 182-day set. hgb loses 0.0463, and that
loss stays in the table. `metrics.json` also carries `baseline_roll7`
(last week's mean; valid 0.5945, test 0.4796), added for Exercise 4.

## Headline finding

Ridge is the best model, and only narrowly better than "yesterday again".
On the H2 2012 test set it scores R² 0.6271 (MAE 873.1) against persistence's
0.5099 (MAE 840.1): ahead on R², behind on MAE. Gradient boosting scores R²
-0.5648, yet still beats the mean baseline comfortably (-2.1441, MAE 2847.7 vs
2021.1). A negative R² here does not mean it loses to its baseline. It means
ridership grew about 65% from 2011 to 2012, and trees trained on 2011 cannot
predict counts above 2011's maximum, while ridge extrapolates. My feature,
`temp_delta_3`, raised ridge's validation R² from 0.6475 to 0.6843 (test 0.5833
to 0.6271) and lowered hgb's, and I kept it.

## Exercise 4: target transformation

hgb loses because 71 of 182 validation days exceed 2011's maximum count
(6043). A tree trained on 2011 cannot predict above that (its highest
validation prediction is 5565), so hgb is capped while ridge extrapolates.

The transformation is `diff_roll7`: fit `cnt − cnt_roll_7`, the departure from
last week's average, then add `cnt_roll_7` back. The target no longer carries
the level, so the cap stops mattering. `cnt_roll_7` is built with `.shift(1)`
and is known at prediction time. The code runs with
`python -m src.train --target diff_roll7` and writes
`reports/metrics_diff_roll7.json`. `make train` and `reports/metrics.json`
stay untransformed.

I chose the target on validation only. Validation R²:

| Target | ridge | hgb |
|---|---|---|
| none | 0.6843 | 0.1722 |
| log1p | -0.2184 | 0.0560 |
| diff_lag1 | 0.6839 | 0.6090 |
| **diff_roll7** | **0.6930** | **0.7393** |
| ratio_roll7 | 0.4916 | 0.7592 |
| logratio_roll7 | 0.5847 | 0.7552 |

`log1p` rescales the target but keeps the level, and it fails, which is what
the cap predicts. Every target measured relative to recent demand rescues hgb.
I picked `diff_roll7` over the top-scoring `ratio_roll7` because it improves
both models. `ratio_roll7` gains hgb 0.02 and costs ridge 0.19, so choosing it
would mean choosing whichever target makes hgb win.

I evaluated the test set once, on 2026-11-25 at 14:00, after making that
choice:

| Model | Valid R² | Test R² | Test MAE |
|---|---|---|---|
| Mean baseline | -0.9318 | -2.1441 | 2847.7 |
| Persistence (yesterday) | 0.4507 | 0.5099 | 840.1 |
| Last-week mean (`cnt_roll_7`) | 0.5945 | 0.4796 | 871.6 |
| ridge, untransformed | 0.6843 | 0.6271 | 873.1 |
| hgb, untransformed | 0.1722 | -0.5648 | 2021.1 |
| ridge, `diff_roll7` | 0.693 | 0.6335 | 858.7 |
| hgb, `diff_roll7` | 0.7393 | **0.6514** | **800.9** |

Is it defensible? As a model, yes. The transform follows from the cap rather
than from a search, it uses no future information, and it applies to both
models. The cap also predicted which targets would work (`log1p` fails,
targets relative to recent demand succeed), and that is what happened.

As evidence that hgb is the better model, no. I tried five targets on 182
validation days, and hgb's lead over ridge shrinks from 0.046 on validation to
0.018 on test, close to the 0.01 noise bar. Both transformed models beat
persistence on test by only about 0.12 to 0.14. The transform removes hgb's
handicap without making it the clear winner. The headline holds in a narrower
form: trees cannot extrapolate levels. Given a target without the level they
match ridge, and neither is far ahead of "yesterday again".

## Exercise 5: ridge vs persistence

| Metric | ridge | Persistence | Better |
|---|---|---|---|
| Test R² | 0.627 | 0.510 | ridge, by 0.117 |
| Test MAE, all 184 days | 873 | 840 | persistence, by 33 rides |
| Test RMSE | 1,063 | 1,219 | ridge |
| Test MAE, 37 big-swing days | 1,336 | 2,281 | ridge, by ~950 rides |
| Test MAE, other 147 days | 757 | 477 | persistence, by ~280 rides |
| Test median miss | 803 | 506 | persistence |
| Test days missed by > 2,000 rides | 7 | 17 | ridge |
| Validation MAE, all 182 days | 767 | 901 | ridge |
| Validation MAE, big-swing days | 1,187 | 2,341 | ridge |
| Validation days missed by > 2,000 rides | 7 | 18 | ridge |

Big-swing days are the top 20% by change from the day before: at least 1,373
rides on test, at least 1,560 on validation. Ridge numbers include
`temp_delta_3`.

I would deploy ridge. The two models fail in different ways: persistence is
very good on ordinary days and very bad when demand jumps, while ridge is a
little worse every day and much better on the jumps. On ordinary days ridge
gives up about 280 rides of accuracy, and on big-swing days it gains about
950. For a bike operator the expensive mistake is missing a swing: no bikes on
a warm day after a cold week, or crews sent out for a day that gets rained
out. Persistence's MAE win also comes from the test set alone; on validation,
ridge wins MAE too. The argument depends on one assumption. If every ride of
error costs the same, MAE is the right metric, and "yesterday again" is the
cheaper, simpler model to deploy.

## Attribution

Models: Claude Opus 5.5 (Anthropic, hosted; Claude Code's default) in Claude
Code for VS Code, for all exercises. Where the assignment needed the local
model, I switched to qwen3.5:9b via Ollama (64k context, built as
`bikeshare-coder`): the blind fault fixes and `agent/audit.py`.

The working split: I provided the ideas and the instructions, and the agent
wrote the code. Where the agent proposed an idea instead (the Exercise 4
target and the extra tests), the decisions list below says so and records
that I approved it.

Generated by the agent:
- The code for `temp_delta_3` in `src/features.py` and its test in `tests/test_leakage.py`, from my idea
- `src/train.py`: `TARGET_TRANSFORMS`, the `--target` flag, `baseline_roll7`
- `src/metrics_table.py` and `.claude/skills/metrics-table/SKILL.md`
- `src/config.py` raw-column sets and `RAW_SHA256`; the hash check in `src/ingest.py`
- Tests: `test_every_raw_column_is_classified`, `test_no_feature_is_a_renamed_leak`, `test_target_transform_round_trips`, and the named-date assertion
- The fixes for faults 1 and 2 and `reports/audit.jsonl` (qwen3.5:9b)

Decisions that were mine, each with its line in `docs/DECISIONS.md`:
- What to predict, what counts as leakage, how to split: kept the starter's choices (the starter lines)
- Baselines: mean and persistence, plus last week's mean once Exercise 4 needed it (`baseline_roll7`)
- The idea for `temp_delta_3` (today's temperature against the previous three days), and keeping it (the `temp_delta_3` line)
- `diff_roll7` is defensible as a model but not as proof hgb is better; I accepted the agent's proposal of the target and made the verdict (Exercise 4 line)
- Deploy ridge over persistence (Exercise 5 line)
- Approve the agent's proposals for the value-based leak test, the raw-column check and the hash check (their lines)
- Which skill to write, `metrics-table` (the `CLAUDE.md` line)

