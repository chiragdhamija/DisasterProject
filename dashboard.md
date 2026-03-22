# Dashboard Metric Definitions

This file explains the four cards shown in the dashboard:

1. `Samples`
2. `Mean Hazard`
3. `Mean Risk (USD/day/sample)`
4. `EAL Total (USD/day)`

All definitions below are based on the current pipeline outputs:

- `NextDayWildFireSpr/data/interim/sample_risk_scores.csv`
- `NextDayWildFireSpr/data/interim/date_risk_summary.csv`
- `NextDayWildFireSpr/tools/fuse_risk_scores.py`

## 1) Per-sample quantities (computed first)

For each sample tile `i` on date `d`:

### Hazard

- Model outputs pixel probabilities for next-day fire.
- `hazard_prob_mean_i` = mean predicted probability across the tile.
- `hazard_index_i = clip(hazard_prob_mean_i, 0, 1)`.

### Exposure (monetary base)

- `asset_value_usd_i = acs_housing_units_i * acs_median_home_value_i`.
- If ACS values are invalid/missing (<= 0), `asset_value_usd_i` becomes `NaN`.

### Vulnerability

- `vulnerability_index_i` comes from SVI (`svi_rpl_themes`, clipped to `[0,1]`).
- `vulnerability_for_risk_i` = `vulnerability_index_i` with median fill for missing values.

### Risk and EAL (sample level)

- Primary risk formula used:
  - `risk_score_i = hazard_index_i * asset_value_usd_i * vulnerability_for_risk_i`
- EAL alias:
  - `risk_eal_usd_i = risk_score_i`

So in this pipeline, per-sample `risk_score` and `risk_eal_usd` are numerically equal.

## 2) Daily dashboard cards (aggregated by date)

For a selected date `d`, using all rows from `sample_risk_scores.csv` where `sample_date == d`:

### `Samples`

- `Samples_d = count(sample_id)`

This is the number of sample tiles available for that date.

### `Mean Hazard`

- `MeanHazard_d = mean(hazard_index_i)`

Unit: unitless probability-like index in `[0,1]`.

### `Mean Risk (USD/day/sample)`

- `MeanRisk_d = mean(risk_score_i)`

Unit: USD per sample for that day.

### `EAL Total (USD/day)`

- `EALTotal_d = sum(risk_eal_usd_i)`
- Since `risk_eal_usd_i = risk_score_i`, this is also:
  - `sum(risk_score_i)`

Unit: total expected loss across all samples for that day.

## 3) Mean Risk vs EAL Total

- `Mean Risk` is an average per sample.
- `EAL Total` is the total over all samples.

Conceptually:

- `EALTotal_d ~= MeanRisk_d * number_of_valid_risk_samples_d`

It may not equal `MeanRisk_d * Samples_d` exactly when some rows have missing `risk_score` (for example missing/invalid ACS asset inputs), because pandas mean/sum ignore `NaN` values.

## 4) Why `Samples` matters

- Higher `Samples` means daily metrics are estimated from more tiles and are more stable.
- Lower `Samples` means the daily aggregates can be more sensitive to a few tiles.

## 4.1) What is a sample tile?

A sample tile is one `64 x 64` spatial patch for a specific date.

Each tile carries:

- model input channels (weather, vegetation, terrain, previous fire, etc.),
- metadata (`sample_lon`, `sample_lat`, `sample_date`),
- label/prediction for wildfire spread in that patch.

So if the dashboard shows `Samples = 200` for a date, the day-level metrics are aggregated from 200 such `64 x 64` patches.

## 5) Map semantics (current)

- Base selected day (`t`): map shows **observed fire only** using ground-truth fire fraction.
- Next days (`t+1`, `t+2` in current window): map shows **predicted spread**.
- Trajectories are built from predicted-fire-positive samples only.

## 6) Dashboard Controls and Time Window

The dashboard works on a short window anchored at a selected base date.

### Base Date (Available)

- You choose a base date `t` from available sampled dates.
- The API returns a window (default horizon `2`): `[t, t+1, t+2]` in available-date order.

### Slider

- Moves within the loaded window dates.
- Each slider position updates map + cards + chart marker for that date.

### Play / Speed

- `Play` animates the slider through the current window.
- `Speed` changes animation interval (slow/normal/fast).

### Status Text

- Shows current load state (window/tract layer fetch messages).

## 7) Map View: Daily Spread + Trajectory

This mode combines point-level wildfire spread display and trajectory lines.

### Daily Spread points

- On base day (`t`): points are shown only where `gt_fire_frac > 0` (observed fire).
- On future days (`t+1`, `t+2`): points are shown only where `hazard_pred_fire_frac > 0` (predicted fire).

### Point styling

- Base day points:
  - color reflects observed fire fraction bins,
  - marker size scales with observed fire fraction.
- Predicted-day points:
  - color reflects risk bins (from `risk_score`),
  - marker size scales with predicted fire fraction.

### Trajectory lines

- Built from predicted-fire-positive points only.
- Per day, positive points are clustered to local centroids.
- Across days, nearest clusters are linked to form multiple trajectories.
- In the viewer, trajectories are displayed only for dates after base day.

## 8) Map View: Tract Risk Map

This mode shows choropleth risk at census-tract level for the selected date.

### Tract aggregation (per date, per GEOID)

- `risk_score_mean` = mean of sample `risk_score` inside tract.
- `hazard_index_mean` = mean sample hazard index inside tract.
- `risk_eal_usd_sum` = sum of sample EAL in tract.
- `samples` = number of samples mapped to that tract/date.

### Coloring

- Tracts are color-binned by `risk_score_mean` quantile-style breaks for that date.

## 9) Right Panel: Daily Snapshot + Risk Across Days

### Daily Snapshot cards

- `Samples`: total sample count on selected date.
- `Mean Hazard`: average `hazard_index` on selected date.
- `Mean Risk (USD/day/sample)`: average `risk_score` on selected date.
- `EAL Total (USD/day)`: sum of `risk_eal_usd` on selected date.

### Risk Across Days chart

- Line 1: mean risk per sample (USD millions).
- Line 2: mean hazard index.
- Line 3: EAL total (USD millions/day).
- Marker: currently selected day in the loaded window.

## 10) Important Interpretation Notes

- This dashboard is based on sampled `64 x 64` tiles, not a dense wall-to-wall raster over California.
- Dates shown in selector/window are dataset-available sampled dates.
- `EAL Total` can be much larger than `Mean Risk` because one is a sum and the other is an average.
