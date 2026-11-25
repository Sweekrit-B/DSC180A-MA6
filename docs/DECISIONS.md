# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-25: Add `temp_delta_3` = temp[d] - mean(temp[d-3..d-1]). Available at d: same-day temp is known like other weather, and the baseline is shift(1) on the calendar index. Valid R^2: ridge 0.6475 -> 0.6843 (+0.037), hgb 0.2185 -> 0.1722 (-0.046, negative). Kept: ridge, the best model, gains above the 0.01 noise bar.
- 2026-11-25: Exercise 4, target `diff_roll7` (cnt − cnt_roll_7) behind `--target`; chosen on validation over ratio/log targets because it improves both models (ridge 0.684 -> 0.693, hgb 0.172 -> 0.739). Test touched once: hgb 0.651 vs ridge 0.634, a 0.018 lead, near the 0.01 noise bar. Verdict: defensible as a model (chosen from the 2011 cap, no future information, applied to both models), not as proof hgb is better. Not adopted as default; `make train` and metrics.json stay untransformed.
- 2026-11-25: Add `baseline_roll7` (predict last week's mean) to every run. It is the anchor of `diff_roll7`, so a transformed model must beat it, not just persistence. Valid R^2 0.5945, above persistence (0.4507).
