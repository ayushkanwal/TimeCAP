# AgriWebb Data-Only Integration Plan (Implemented)

## Objective
Prepare AgriWebb data so it can be consumed by the existing TimeCAP pipeline without changing model internals.

## Locked Decisions
- Keep existing model behavior unchanged (classification-first training path).
- Build `7-day ADG` from interpolated daily weight.
- Convert ADG to classes with thresholds:
  - `0` decrease: `adg_7d < -0.10`
  - `1` neutral: `-0.10 <= adg_7d <= 0.10`
  - `2` increase: `adg_7d > 0.10`
- Use both `Bulk` and `Individual` rows.
- Use deterministic rule-based text summaries.
- Use chronological split `60/20/20`.
- Exclude treatments in v1 (not provided in current extract).

## Dedup Rule (Approved Amendment)
For each `(LIVESTOCK_ID, day)`:
1. Keep row with latest `OBSERVATION_DATE`.
2. If tied, keep row with latest `CREATION_DATE`.
3. If still tied, keep row with highest `RECORD_ID` (stable tie-break).

## Data Inputs
- `farm-info.csv`
- `paddock-info.csv`
- `paddock-moves.csv`
- `weather.csv`
- `weights.csv`

## Processing Outputs
Generated under `dataset/agriwebb/processed`:
- `x_time.npy` (`[N, 56, F]`, float32)
- `x_text.npy` (`[N]`, object/string)
- `y_class.npy` (`[N]`, int64)
- `y_adg_7d.npy` (`[N]`, float32)
- `sample_meta.parquet`
- `feature_names.json`
- `splits.json`
- `data_quality_report.json`

## Leakage Guards
- Features are based on data available at or before anchor day `t`.
- Target uses `t` to `t+7` interpolation only for label creation.
- No future (`> t`) features are used in windows.

## Split Policy
- Sort samples by anchor day.
- Chronological train/val/test = `60/20/20`.

## Script
Use `dataset/agriwebb/prepare_agriwebb.py` to build all artifacts.
