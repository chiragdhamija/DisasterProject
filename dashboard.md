# Dashboard Definitions and Calculations

This file documents what the current dashboard shows, how each number is computed, and what each map mode represents.

## 1) What the dashboard currently shows

Current right-panel cards:

1. `Mean Hazard`
2. `Total Risk (USD/day)`

Current chart:

1. `Total Risk Across Days` (line in USD millions/day)
2. `Selected Day` marker

## 2) Core per-sample quantities

All per-sample values come from `NextDayWildFireSpr/data/interim/sample_risk_scores.csv`.

For each sample tile `i` on date `d`:

- `hazard_prob_mean_i`: model mean next-day fire probability over the tile.
- `hazard_index_i = clip(hazard_prob_mean_i, 0, 1)`.
- `asset_value_usd_i = acs_housing_units_i * acs_median_home_value_i`.
- `vulnerability_for_risk_i`: SVI-based vulnerability (`svi_rpl_themes`, clipped and filled where needed in fusion).
- `risk_score_i = hazard_index_i * asset_value_usd_i * vulnerability_for_risk_i`.
- `risk_eal_usd_i = risk_score_i`.

So in this pipeline, per-sample `risk_score` and per-sample `risk_eal_usd` are numerically identical.

## 3) Daily metrics used by frontend

Daily rows come from `NextDayWildFireSpr/frontend/data/daily_risk_summary.json` (built from `date_risk_summary.csv`).

For selected day `d`:

- `Mean Hazard` card uses `hazard_index_mean_d = mean(hazard_index_i)` over samples on day `d`.
- `Total Risk (USD/day)` card uses `risk_eal_usd_sum_d = sum(risk_eal_usd_i)` over valid samples on day `d`.

Important detail:

- The total is a sum over valid risk rows. Rows with missing monetary inputs produce null risk and do not contribute to the sum.

## 4) Meaning of sample tile

A sample tile is one `64 x 64` spatial patch at one date.

Each tile has:

- hazard-model inputs (weather, terrain, vegetation, previous fire, metadata channels),
- location/date metadata (`sample_lon`, `sample_lat`, `sample_date`),
- ground-truth fire fraction and model-predicted fire fraction.

## 5) Map mode: Daily Spread + Trajectory

This mode combines spread points and trajectory lines.

Spread point rendering logic:

- Base day `t` (selected date): shows only observed fire points where `gt_fire_frac > 0`.
- Future window days (`t+1`, `t+2` by default horizon): shows only predicted spread points where `hazard_pred_fire_frac > 0`.

Point styling:

- Base day points are colored/sized by observed fire fraction.
- Future-day points are colored by risk bins and sized by predicted fire fraction.

Trajectory construction:

- Built only from predicted-fire-positive points.
- Daily points are clustered to centroids (`cluster_radius_deg = 0.30`).
- Centroids are linked across dates (`link_radius_deg = 0.90`) to create multiple trajectories.
- Frontend renders trajectory segments only for days after base date.

## 6) Map mode: Tract Risk Map

This mode shows tract-level choropleth for selected date.

Per-date, per-tract aggregation (from sample-level rows mapped by `GEOID`):

- `risk_score_mean`
- `hazard_index_mean`
- `risk_eal_usd_sum`
- `samples`

Choropleth color is based on `risk_score_mean` with quantile-style breaks for that date.

## 7) Time-window and API behavior

Dashboard uses the local API server (`serve_frontend_api.py`) so heavy data filtering happens in backend.

Endpoints:

1. `/api/meta`
2. `/api/window?date=YYYY-MM-DD&horizon=H`
3. `/api/tract-risk?date=YYYY-MM-DD`

Window logic:

- Default horizon is `2`, so window is three dates: `[t, t+1, t+2]`.
- `t+1` and `t+2` are next available sampled dates, not guaranteed strict calendar +1/+2 if dates are missing.

## 8) UI controls

- `Base Date`: anchor date `t` from available sampled dates.
- `Slider`: moves within loaded window dates.
- `Play` and `Speed`: animate slider across loaded window.
- `Map View`: switch between spread/trajectory and tract-risk choropleth.

## 9) Hazard model input channels (full forms)

These are the 15 channels used in hazard training/inference, matching the current hazard dataset:

1. `elevation`: elevation (m)
2. `th`: wind direction (degrees)
3. `vs`: wind speed at 10m (m/s)
4. `tmmn`: minimum temperature (K)
5. `tmmx`: maximum temperature (K)
6. `sph`: specific humidity (mass fraction)
7. `pr`: precipitation (mm, daily)
8. `pdsi`: Palmer Drought Severity Index
9. `NDVI`: Normalized Difference Vegetation Index
10. `population`: persons per km² (population density)
11. `erc`: Energy Release Component (fire danger index)
12. `PrevFireMask`: previous-day fire mask feature
13. `meta_lon_z`: z-normalized longitude
14. `meta_lat_z`: z-normalized latitude
15. `meta_day_of_year_z`: z-normalized day-of-year

Source catalogs for raw physical channels are listed in `report.md` (official Earth Engine dataset pages).

Direct catalog links:

- https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
- https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT
- https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP13A1
- https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Population_Density
- https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD14A1
