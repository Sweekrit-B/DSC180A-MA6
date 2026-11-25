---
name: metrics-table
description: Put a results table into REPORT.md straight from a metrics JSON, so the report matches it digit for digit. Use whenever model scores go into REPORT.md.
allowed-tools: Read, Edit, Bash(make train*), Bash(python -m src.metrics_table*)
---

# Metrics table: $ARGUMENTS

`$ARGUMENTS` is a metrics file, optionally followed by `--valid-only`.
With no file, use `reports/metrics.json`.

1. If the file is `reports/metrics.json` and the code under `src/` changed
   since it was written, run `make train` first.
2. Run `python -m src.metrics_table $ARGUMENTS`.
3. Paste the table into REPORT.md exactly as printed. Never retype, round or
   reorder a number. If REPORT.md already has a table from this file, compare
   every cell and list each mismatch instead of fixing it silently.
4. Use `--valid-only` while still choosing features or targets. Test columns
   go into the report only after the choice is final.
