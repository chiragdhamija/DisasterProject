# Wildfire Spread and HEV Risk Mapping for California (2020)

## 1. Project Definition

### 1.1 Objective
This project builds a full pipeline to:
- Predict next-day wildfire spread hazard from geospatial/environmental inputs.
- Fuse hazard with Exposure and Vulnerability features.
- Produce daily and tract-level risk outputs for California.
- Serve an interactive dashboard for spread, trajectory, and risk maps.

Core risk framing used in implementation:

`Risk (USD/day) = Hazard x Exposure x Vulnerability`

Where in this codebase:
- Hazard = model-predicted fire probability proxy (`hazard_index = hazard_prob_mean`)
- Exposure = monetary asset value proxy per sample (`asset_value_usd`)
- Vulnerability = SVI-based damage fraction (`vulnerability_for_risk`)

### 1.2 Study Area
- Geography: California, USA
- Spatial boundary in export: state geometry from Earth Engine `TIGER/2018/States` filtered with `STATEFP=06`
- Frontend map bounds: approx California extent `[-124.55, 32.35]` to `[-114.05, 42.1]`

### 1.3 Time Scope
- Calendar year: `2020-01-01` to `2020-12-31`
- Final dataset coverage: all 366 days of leap year 2020
- No missing days in final manifest/sample index.

---

## 2. Final End-to-End Pipeline

The implemented pipeline (Option A final) is orchestrated by:
- `NextDayWildFireSpr/tools/run_option_a_export_ca_2020.sh` (Earth Engine export submission)
- `NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh` (local rebuild from downloaded TFRecords)

### 2.1 Stages
1. Earth Engine export of NDWS-style TFRecords with sample geospatial/time metadata.
2. Build reproducible sample manifest from downloaded TFRecords.
3. Build canonical hazard training pickles (`train/validation/test`).
4. Train U-Net hazard model.
5. Preprocess vector layers into common CRS (`EPSG:3310`).
6. Join samples to census tracts.
7. Build HEV per-sample features.
8. Run hazard inference on all samples.
9. Fuse hazard + HEV into monetary risk outputs.
10. Build frontend geospatial assets (points, trajectories, tract choropleth, daily summaries).
11. Serve frontend + API locally.

---

## 3. Data Inventory (Final)

## 3.1 Core Hazard Dataset (Generated)
### Raw mapped TFRecords
Directory: `NextDayWildFireSpr/data/ndws64_meta_ca`

Manifest summary (`sample_manifest_summary.json`, from `totals` object):
- Total rows: `61,064`
- Split rows: `train=50,142`, `eval=5,525`, `test=5,397`
- Files: `53`
- Missing required metadata: `0`
- Split mismatches: `0`

Date coverage:
- Min date: `2020-01-01`
- Max date: `2020-12-31`
- Unique dates: `366`
- Missing dates in 2020: `0`

Day-level split assignment:
- Train days: `296`
- Eval days: `35`
- Test days: `35`
- Dates are mutually exclusive across splits.

### Canonical hazard pickles
Directory: `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard`

Files:
- `train.data`, `train.labels`
- `validation.data`, `validation.labels`
- `test.data`, `test.labels`
- `channels_metadata.json`
- `sample_index.csv`

From `channels_metadata.json`:
- Tile size: `64`
- Channels: `15`
- Feature dtype: `float16`
- Label dtype: `uint8`
- Shapes:
  - Train: `(50142, 15, 64, 64)` labels `(50142, 64, 64)`
  - Validation: `(5525, 15, 64, 64)` labels `(5525, 64, 64)`
  - Test: `(5397, 15, 64, 64)` labels `(5397, 64, 64)`

Channel order and definitions (sourced):

| # | Channel | Definition (exact meaning used) | Origin |
|---:|---|---|---|
| 1 | `elevation` | Elevation (meters) from DEM | `USGS/SRTMGL1_003` band `elevation` |
| 2 | `th` | Wind direction (degrees) | `IDAHO_EPSCOR/GRIDMET` band `th` |
| 3 | `vs` | Wind velocity at 10m (m/s) | `IDAHO_EPSCOR/GRIDMET` band `vs` |
| 4 | `tmmn` | Minimum temperature (Kelvin) | `IDAHO_EPSCOR/GRIDMET` band `tmmn` |
| 5 | `tmmx` | Maximum temperature (Kelvin) | `IDAHO_EPSCOR/GRIDMET` band `tmmx` |
| 6 | `sph` | Specific humidity (mass fraction) | `IDAHO_EPSCOR/GRIDMET` band `sph` |
| 7 | `pr` | Precipitation amount (mm, daily total) | `IDAHO_EPSCOR/GRIDMET` band `pr` |
| 8 | `pdsi` | Palmer Drought Severity Index | `GRIDMET/DROUGHT` band `pdsi` |
| 9 | `NDVI` | Normalized Difference Vegetation Index | `NOAA/VIIRS/001/VNP13A1` band `NDVI` |
| 10 | `population` | Estimated number of persons per square kilometer | renamed from `CIESIN/GPWv411/GPW_Population_Density` band `population_density` |
| 11 | `erc` | Energy release component (NFDRS fire danger index) | `IDAHO_EPSCOR/GRIDMET` band `erc` |
| 12 | `PrevFireMask` | Previous-day fire mask feature (derived from MODIS fire mask classes; prior-day max window, then binarized in canonical build) | derived from `MODIS/006/MOD14A1` `FireMask` (class definitions aligned with `MODIS/061/MOD14A1`) |
| 13 | `meta_lon_z` | Z-score normalized sample longitude | derived metadata channel |
| 14 | `meta_lat_z` | Z-score normalized sample latitude | derived metadata channel |
| 15 | `meta_day_of_year_z` | Z-score normalized day-of-year from `sample_date` | derived metadata channel |

Definition sources (official Earth Engine catalog pages):
- SRTM elevation (`USGS/SRTMGL1_003`): https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- GRIDMET meteorology (`IDAHO_EPSCOR/GRIDMET`, for `pr`, `sph`, `th`, `tmmn`, `tmmx`, `vs`, `erc`): https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
- Drought index (`GRIDMET/DROUGHT`, `pdsi`): https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT
- VIIRS NDVI (`NOAA/VIIRS/001/VNP13A1`, `NDVI`): https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP13A1
- Population density (`CIESIN/GPWv411/GPW_Population_Density`, `population_density`): https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Population_Density
- MODIS FireMask classes (`MODIS/006/MOD14A1`, used in export; `MODIS/061/MOD14A1` successor): https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD14A1 and https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1

## 3.2 External Datasets (Actually Used)
All loaded from `Ext_Datasets/`.

1. `TIGER2020_CaliforniaTractsShapefile/tl_2020_06_tract.shp`
- Used for tract geometry and attributes:
  - `GEOID`, `NAMELSAD`, `ALAND`, `AWATER`
- Purpose in pipeline:
  - Assign each sample tile to a census tract (spatial identity key).
  - Provide tract polygons for tract-level risk mapping in dashboard outputs.

2. `CaliforniaRoads_InfraShapefile-CRS_-_Functional_Classification/CRS_-_Functional_Classification.shp`
- Used to compute nearest road proximity per sample:
  - `road_nearest_dist_m`
  - nearest `RouteID`, `F_System`
- Purpose in pipeline:
  - Add infrastructure proximity context to each sample.
  - Contribute to exposure modeling via road-proximity term in `exposure_index`.

3. `California_Historic_Fire_Perimeters_-6273763535668926275/California_Fire_Perimeters_(all).shp`
- Used for historical fire context features within buffer:
  - `past_fire_count_5y_10km`
  - `past_fire_acres_5y_10km`
  - `days_since_fire_min_5y_10km`
- Uses `ALARM_DATE`, `GIS_ACRES`, geometry.
- Purpose in pipeline:
  - Add local historical-burn context around each sample date/location (5-year lookback, 10 km buffer).
  - Retained for HEV analysis and future risk/hazard refinement.
  - Note: these fields are generated in HEV table but are not directly multiplied in final `risk_score = H x E x V`.

4. `acs_2020_exposure.json`
- ACS API extract fields:
  - `B01003_001E` -> population (`acs_population`)
  - `B25001_001E` -> housing units (`acs_housing_units`)
  - `B25077_001E` -> median home value (`acs_median_home_value`)
- Joined by GEOID built from (`state`,`county`,`tract`).
- Purpose in pipeline:
  - Provide tract socioeconomic exposure values (population, housing stock, median home value).
  - Supply monetary exposure proxy (`asset_value_usd`) used in final risk equation.

5. `SVI_2020_CaliforniaTract.csv`
- SVI fields used:
  - `RPL_THEMES` (primary vulnerability)
  - `RPL_THEME1`..`RPL_THEME4` (retained)
- Mapped to:
  - `svi_rpl_themes`, `svi_rpl_theme1`..`svi_rpl_theme4`
- Purpose in pipeline:
  - Provide tract-level vulnerability signal.
  - Used as primary vulnerability term in risk fusion (`vulnerability_index`, then `vulnerability_for_risk`).

## 3.3 External Dataset Present but Not Used in Final Pipeline
- `Ext_Datasets/CAL-FIRE_OFFICIAL_FIREPARAMS/fire24_1.gdb`
- It is present in storage but is not referenced by final scripts.

## 3.4 Generated Data Volume (Final Build Snapshot)
Measured from current workspace outputs:

- `NextDayWildFireSpr/data/ndws64_meta_ca`: `7,816,771,334` bytes (`~7.817 GB`, `~7.280 GiB`)
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard`: `7,764,842,113` bytes (`~7.765 GB`, `~7.232 GiB`)
- `NextDayWildFireSpr/data/interim`: `936,463,523` bytes (`~0.936 GB`, `~0.872 GiB`)
- `NextDayWildFireSpr/frontend/data`: `47,952,558` bytes (`~0.048 GB`, `~0.045 GiB`)

Total generated footprint across these output groups: `~16.566 GB` (`~15.429 GiB`).

---

## 4. Data Generation from Earth Engine

Script: `NextDayWildFireSpr/tools/ee_export_with_mapping.py`

### 4.1 EE Source Collections and Bands
Configured IDs:
- `USGS/SRTMGL1_003` -> `elevation`
- `NOAA/VIIRS/001/VNP13A1` -> `NDVI`
- `GRIDMET/DROUGHT` -> `pdsi`
- `IDAHO_EPSCOR/GRIDMET` -> `pr`, `sph`, `th`, `tmmn`, `tmmx`, `vs`, `erc`
- `MODIS/006/MOD14A1` -> `FireMask`
- `CIESIN/GPWv411/GPW_Population_Density` -> `population_density`

Notes:
- EE emitted deprecation warnings for VIIRS `001` and MODIS `006` during export submission.

### 4.2 Temporal Feature Construction
Per `window_start` (1-day prediction target):
- Drought window: lagged median over configured days.
- Vegetation window: lagged median.
- Weather window: lagged median.
- `PrevFireMask`: prior-day max fire mask.
- `FireMask`: current 1-day max fire mask.
- Detection mask for sampling: `detection = clamp(FireMask,6,7)-6`.

Concrete temporal ranges used per sample day `t`:
- Drought (`GRIDMET/DROUGHT`): median over `[t-5, t)`
- Vegetation (`NOAA/VIIRS/001/VNP13A1`): median over `[t-8, t)`
- Weather (`IDAHO_EPSCOR/GRIDMET`): median over `[t-2, t)`
- Previous fire (`MODIS/006/MOD14A1`): max over `[t-1, t)`
- Target fire label (`MODIS/006/MOD14A1`): max over `[t, t+1)`
- Elevation (`USGS/SRTMGL1_003`) and population (`CIESIN/GPWv411/GPW_Population_Density`) are static in this workflow.

### 4.3 Sampling Logic
- Tile kernel: `64 x 64`
- Sampling resolution: default 1000m (`sampling_scale`)
- Stratified sampling on `detection` class.
- For days with no positive fire pixels, script still samples negatives (`no_fire_samples_per_day`) so the day is retained.
- Final export date range used for full build: `2020-01-01` to `2021-01-01` (12 monthly submission windows).

### 4.4 Split Strategy
- Split at day level, not tile level.
- `split_days_into_train_eval_test(...)`:
  - shuffle day offsets with seed
  - eval ratio = `split_ratio`
  - test ratio = same as eval

### 4.5 Metadata Preservation (Critical)
Each exported sample includes:
- `sample_lon`
- `sample_lat`
- `sample_date`
- `start_day`
- `split`

This is the key mapping that enables later sample->tract join and date-windowed frontend playback.

---

## 5. Manifest and Canonical Hazard Dataset Build

## 5.1 Manifest Build
Script: `tools/build_sample_manifest.py`

Output columns:
- `sample_id`, `split`, `sample_date`, `start_day`, `sample_lon`, `sample_lat`, `source_file`, `record_index`

`sample_id` convention:
- `{split}_{source_file_stem}_{record_index:08d}`

Integrity checks:
- duplicate sample_id forbidden
- required metadata must exist if `--fail_on_missing_metadata`

## 5.2 Hazard Pickle Build
Script: `tools/build_hazard_pickles.py`

Input:
- mapped TFRecords + manifest

Output:
- pickled tensors for train/validation/test
- `channels_metadata.json`
- `sample_index.csv` mapping array row -> sample metadata

Label binarization (`_to_binary_fire_mask`):
- If values in `[-1,1]`: `>0` => fire
- Else categorical mask: `>=7` => fire

Fire-level handling note (discussed and implemented):
- MODIS categorical `FireMask` can contain non-binary class codes (for example values such as `3`, `5`, `7`, `8`, `9`).
- Final preprocessing rule is explicit and fixed:
  - `FireMask >= 7` -> fire (`1`)
  - `FireMask < 7` -> non-fire (`0`)

Location/time meta channels:
- `meta_lon_z`, `meta_lat_z`, `meta_day_of_year_z`
- z-normalized using train-split mean/std only

Train normalization constants (from metadata):
- `meta_lon_z`: mean `-119.7486`, std `2.4645`
- `meta_lat_z`: mean `37.3110`, std `2.5713`
- `meta_day_of_year_z`: mean `183.5329`, std `107.8623`

### 5.3 Additional Preprocessing Rules and Similar Decisions
- Required metadata enforcement:
  - Records missing `sample_date`, `sample_lon`, or `sample_lat` are excluded during manifest build.
- Deterministic identity:
  - `sample_id` is deterministic and used as the stable join key across manifest, canonical dataset, HEV table, hazard inference, and risk outputs.
- Split conversion:
  - Raw split names (`train`, `eval`, `test`) are mapped to canonical model splits (`train`, `validation`, `test`).
- Storage dtypes:
  - Features stored as `float16`; labels stored as `uint8` (volume/runtime tradeoff).
- Meta-channel scaling:
  - `meta_lon_z`, `meta_lat_z`, `meta_day_of_year_z` are normalized using train-only statistics, then broadcast into constant 64x64 channels per sample tile.
- Raw band scaling policy:
  - NDWS physical bands are used in their exported numeric scale (no additional global min-max normalization in `build_hazard_pickles.py`).
  - The only explicit thresholding/conversion is fire-mask binarization.
- Classification thresholds:
  - In training/evaluation metrics, logits are binarized at `0` (equivalent to probability threshold `0.5`).
  - In hazard inference export, probability threshold is `0.5` (`--threshold 0.5`) for predicted-fire pixel metrics.

---

## 6. Geospatial Processing and HEV Feature Engineering

## 6.1 CRS Standardization Decision
File: `tools/spatial_standards.py`
- Target CRS: `EPSG:3310` (`NAD83 / California Albers`)
- Distance unit: meter
- Area unit: square meter

This CRS is used consistently for:
- tract overlays
- nearest-road distance
- fire-history buffering/intersections

## 6.2 Vector Preprocessing
Script: `tools/preprocess_geospatial_layers.py`

Outputs (`data/interim/geospatial_3310/`):
- `tracts_3310.gpkg` (9129 rows)
- `roads_3310.gpkg` (779,834 rows)
- `fire_perimeters_3310.gpkg` (22,810 rows)

All layers:
- cleaned for invalid/empty geometry
- reprojected to `EPSG:3310`

## 6.3 Sample -> Tract Join
Script: `tools/build_sample_tract_join.py`

Method:
- Convert sample points (lon/lat) to EPSG:3310
- Spatial join with tract polygons (`intersects`)
- Optional nearest fallback exists in code but final summary shows all matched by intersects

Final summary:
- rows_total: `61,064`
- rows_matched: `61,064`
- match_rate: `1.0`
- match_method_counts: `intersects=61,064`

## 6.4 HEV Feature Table
Script: `tools/build_hev_features.py`
Output: `data/interim/sample_features_hev.csv` (`61,064` rows)

Derived fields include:
- Exposure base fields:
  - `acs_population`
  - `acs_housing_units`
  - `acs_median_home_value`
  - `tract_area_km2 = ALAND / 1e6`
  - `exposure_pop_density_km2 = acs_population / tract_area_km2`
  - `exposure_housing_density_km2 = acs_housing_units / tract_area_km2`
- Vulnerability fields:
  - `svi_rpl_themes` (+ theme1..4)
  - `vulnerability_index = svi_rpl_themes`
- Infrastructure proximity:
  - `road_nearest_dist_m`
  - `road_nearest_route_id`, `road_nearest_f_system`
- Fire history context (5-year lookback, 10km buffer):
  - `past_fire_count_5y_10km`
  - `past_fire_acres_5y_10km`
  - `days_since_fire_min_5y_10km`

Null-rate summary:
- `svi_rpl_themes_missing`: `0.02528` (~2.53%)
- `road_dist_missing`: `0.0`

---

## 7. Hazard Model

## 7.1 Model Family and Base Architecture
Training script: `NextDayWildFireSpr/trainModel-II.py`
Selected model: `U_Net` from `leejunhyun_unet_models.py`

Architecture details:
- Encoder channels: 64 -> 128 -> 256 -> 512 -> 1024
- Decoder with skip concatenations back to 64
- Output head: `1x1 conv` to single logit channel
- Input channels: 15
- Output: per-pixel logit map (`64x64`)

Important implementation note:
- In this specific `U_Net` class, `MaxPool2d(kernel_size=1, stride=2)` is used in encoder downsampling.

## 7.2 Model Input and Output Definition
Input tensor per sample:
- Shape: `(15, 64, 64)`
- Contains 12 NDWS physical channels + 3 metadata z-channels

Output tensor per sample:
- Shape: `(1, 64, 64)` logits
- Through sigmoid => fire probability per pixel for next-day mask

## 7.3 Training Data Preprocessing
In `datasets.py`:
- Data loaded from pickles.
- `good_indices` removes crops containing `-1` labels.
- Crop size is `64`, equal to tile size (effectively full tile use).
- Optional rotation and random flips available.

Final run used:
- `rotation_factor=1` (no rotational augmentation)
- `random_flip=True`
- `batch_size=32`
- `amp=True`

## 7.4 Loss and Optimization
Configured in `trainModel-II.py`:
- Primary criterion: `BCEWithLogitsLoss(pos_weight=10)`
- Also logged custom validation loss from `metrics.py`: `WBCE + 2*dice_loss`
- Optimizer: `AdamW`
- LR scheduler: `ReduceLROnPlateau` on validation F1
- Gradient clipping: `max_norm=1.0`

## 7.5 Final Training Run (Used for Inference)
Log: `NextDayWildFireSpr/train_hazard_option_a_full_stable.log`

Command used:
```bash
python trainModel-II.py \
  --epochs 8 \
  --batch_size 32 \
  --num_workers 0 \
  --rotation_factor 1 \
  --max_train_samples 0 \
  --max_val_samples 0 \
  --amp \
  --random_flip \
  --dataset_path data/next-day-wildfire-spread-ca-hazard \
  --channels_metadata data/next-day-wildfire-spread-ca-hazard/channels_metadata.json \
  --learning_rate 4e-4 \
  --weight_decay 1e-4 \
  --pos_weight 10 \
  --grad_clip 1.0 \
  --lr_factor 0.6 \
  --lr_patience 2 \
  --early_stop_patience 4 \
  --min_delta 0.0005
```

Run summary:
- Train samples: `50,142`
- Validation samples: `5,525`
- Epochs run: `8`
- Best saved model: `savedModels/model-U_Net-bestF1Score-Rank-0.weights`
- Logged best epoch index: `6` (0-based), i.e. epoch 7 by count
- Logged best F1: `0.529990662040049`

Epoch-wise validation (from `NextDayWildFireSpr/savedModels/val_metrics_history.pkl`):

| Epoch | Val Loss | IoU | Acc | F1 | AUC | Dice | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.1332 | 0.5462 | 0.9707 | 0.5082 | 0.7646 | 0.5602 | 0.5105 | 0.5528 |
| 2 | 1.8982 | 0.6201 | 0.9794 | 0.5152 | 0.7738 | 0.6337 | 0.5129 | 0.5509 |
| 3 | 1.8984 | 0.6693 | 0.9918 | 0.5265 | 0.8398 | 0.6826 | 0.5258 | 0.5426 |
| 4 | 2.1384 | 0.5552 | 0.9840 | 0.5236 | 0.8410 | 0.5698 | 0.5252 | 0.5606 |
| 5 | 2.3172 | 0.4717 | 0.9299 | 0.4917 | 0.7505 | 0.4917 | 0.5125 | 0.5484 |
| 6 | 2.1387 | 0.5928 | 0.9904 | 0.5292 | 0.8398 | 0.6072 | 0.5384 | 0.5599 |
| 7 | 2.5516 | 0.5647 | 0.9916 | 0.5300 | 0.8478 | 0.5791 | 0.5347 | 0.5537 |
| 8 | 2.4293 | 0.4161 | 0.8282 | 0.4479 | 0.7239 | 0.4479 | 0.5046 | 0.5153 |

---

## 8. Hazard Inference

Script: `tools/infer_hazard_scores.py`

Input:
- canonical pickled dataset
- channel metadata
- sample index
- trained model weights

Output:
- `data/interim/hazard_predictions.csv`
- `data/interim/hazard_predictions_summary.json`

Per-sample exported hazard fields:
- `hazard_prob_mean`
- `hazard_prob_p95`
- `hazard_prob_max`
- `hazard_pred_fire_frac`
- `gt_fire_frac`

Final inference summary:
- Rows: `61,064`
- Device: `cuda`
- Threshold: `0.5`
- Mean `hazard_prob_mean`: `0.00990`
- Mean predicted fire fraction: `0.00699`
- Mean GT fire fraction: `0.00289` (computed from CSV)

Pixel metrics (class-1 positive class):
- Precision: `0.2568`
- Recall: `0.6224`
- F1: `0.3636`
- IoU: `0.2222`
- Accuracy: `0.9937`
- Brier score: `0.00448`

Macro-like metrics (for comparability with training-style macro averages):
- Macro F1: `0.6802`
- Macro IoU: `0.6080`
- Class-0 F1: `0.9968`
- Class-1 F1: `0.3636`

Why macro and positive F1 differ strongly:
- Severe class imbalance (most pixels are no-fire).
- Class-0 dominates macro scores.

---

## 9. Risk Fusion Methodology (HEV)

Script: `tools/fuse_risk_scores.py`

## 9.1 Hazard component
- `hazard_index = clip(hazard_prob_mean, 0, 1)`

## 9.2 Exposure index (normalized, non-monetary composite)
Components are min-max scaled (some with `log1p`):
- `exp_pop_density_idx` from `exposure_pop_density_km2` (`log1p`)
- `exp_housing_density_idx` from `exposure_housing_density_km2` (`log1p`)
- `exp_home_value_idx` from `acs_median_home_value` (`log1p`)
- `exp_road_proximity_idx` from `1 / (1 + road_nearest_dist_m)`

Weighted composite:
`exposure_index = 0.35*pop + 0.25*housing + 0.25*home_value + 0.15*road_proximity`

## 9.3 Vulnerability component
- Primary: `vulnerability_index = svi_rpl_themes` clipped to `[0,1]`
- Missing handling for monetary risk calculation:
  - `vulnerability_for_risk = vulnerability_index.fillna(median)`
  - Median used: `0.5359`
  - Imputed rows: `1544`

## 9.4 Monetary exposure and risk
Monetary asset proxy:
- `asset_value_usd = acs_housing_units * acs_median_home_value`
- Non-positive values set to NaN

Primary risk formula in code:
- `risk_score = hazard_index * asset_value_usd * vulnerability_for_risk`
- Alias: `risk_eal_usd = risk_score`

Interpretation used in dashboard:
- `risk_eal_usd` is treated as Expected Annual Loss-style daily proxy for each sample-day.
- Daily total risk is sum over samples for that date.

## 9.5 Additional outputs retained for analysis
- `risk_score_weighted` (normalized weighted H/E/V composite, non-monetary)
- `expected_property_loss_usd_weighted`
- `expected_property_loss_usd_hev`
- `expected_population_affected`
- `expected_housing_units_affected`
- `risk_tier` via quantiles (`very_low`..`very_high`)

## 9.6 Risk fusion outputs
From `risk_fusion_summary.json`:
- Sample rows: `61,064`
- Tract rows: `1,098`
- Date rows: `366`
- Mean sample risk: `3,031,545.12`
- Total risk (`sum risk_eal_usd`): `175,132,361,314.19`
- Risk null rows: `3,294` (all due to null `asset_value_usd`)

---

## 10. Frontend Asset Build and Dashboard Pipeline

Script: `tools/build_frontend_assets.py`

Inputs:
- `sample_risk_scores.csv`
- `hazard_predictions.csv`
- `date_risk_summary.csv`
- `tract_risk_summary.csv`
- tract geometries (`tracts_3310.gpkg`)

Outputs (`NextDayWildFireSpr/frontend/data/`):
- `spread_points.geojson`
- `spread_daily_compact.json`
- `spread_trajectory.geojson`
- `spread_trajectory_compact.json`
- `daily_risk_summary.json`
- `tract_risk.geojson`
- `frontend_assets_summary.json`

From `frontend_assets_summary.json`:
- Spread points: `61,064`
- Trajectory source points: `10,355` (predicted-fire-positive only)
- Trajectory dates: `356`
- Multi trajectories: `1,461`
- Daily summary rows: `366`
- Tract map features: `1,048`
  - (tract summary had 1,098 rows; 50 had null `risk_score_mean` and were dropped when `keep_null_tracts=false`)

## 10.1 Trajectory construction logic
- Daily clusters from predicted-fire-positive samples only.
- Cluster radius: `0.30 deg`
- Inter-day link radius: `0.90 deg`
- Multi-trajectory linking generates parallel tracks (not single centroid only).

## 10.2 Dashboard serving model
Server: `tools/serve_frontend_api.py`

Provides:
- static frontend files
- API endpoints:
  - `/api/meta`
  - `/api/window?date=YYYY-MM-DD&horizon=2`
  - `/api/tract-risk?date=YYYY-MM-DD`

Important date-window behavior (Option B logic in API):
- Selected date returns next **available** dates in data index, not strict calendar interpolation.

## 10.3 Current dashboard behavior
- Base date map (t): shows **observed** fire only (`gt_fire_frac > 0`).
- Next dates (t+): show **predicted** spread samples (`hazard_pred_fire_frac > 0`).
- KPI cards currently show:
  - `Mean Hazard`
  - `Total Risk (USD/day)`
- Risk chart currently shows:
  - `Total Risk (USD millions/day)` + selected-day marker

---

## 11. Complete Methodology Mapping to User-Requested Flow

### 11.1 Geospatial processing
- Standardized all vectors to EPSG:3310.
- Cleaned geometries and preserved key attributes.
- Prepared tract/road/fire geospatial layers for distance and intersection operations.

### 11.2 Sample -> tract join
- Exact point-in-polygon attachment of each tile sample to tract GEOID.
- Produced tract metadata per sample (`ALAND`, `AWATER`, names, point coords in 3310).

### 11.3 Building HEV features
- Exposure from ACS + tract area + road proximity.
- Vulnerability from CDC/ATSDR SVI percentile ranks.
- Historical fire context from perimeter intersections in a 10 km buffer and 5-year lookback.

### 11.4 Hazard inference
- U-Net predicts per-pixel logits for next-day fire.
- Converted to probabilities and thresholded predictions.
- Aggregated sample-level hazard indices (`mean/p95/max fire probability`, `predicted fire fraction`).

### 11.5 Risk fusion
- Joined hazard and HEV on `sample_id`.
- Computed monetary risk as HxExV per sample.
- Aggregated to day and tract summaries.

### 11.6 Frontend asset building
- Exported both rich GeoJSON and compact JSON payloads.
- Built predicted-spread points, multi-trajectories, daily summaries, tract choropleth.

---

## 12. Inputs and Outputs of the Trained Hazard Model

## 12.1 Input (per sample)
- Tensor: `(15, 64, 64)`
- Channels:
  - 12 physical wildfire/environment channels
  - 3 metadata channels (`lon_z`, `lat_z`, `day_of_year_z`)

## 12.2 Output (per sample)
- Raw logits: `(1, 64, 64)`
- Sigmoid probabilities: fire probability map for next day
- Aggregated inference exports:
  - `hazard_prob_mean`, `hazard_prob_p95`, `hazard_prob_max`, `hazard_pred_fire_frac`

---

## 13. Final Result Artifacts

## 13.1 Model
- `NextDayWildFireSpr/savedModels/model-U_Net-bestF1Score-Rank-0.weights`
- `train_loss_history.pkl`
- `val_metrics_history.pkl`

## 13.2 Interim analytical outputs
- `data/interim/hazard_predictions.csv`
- `data/interim/sample_features_hev.csv`
- `data/interim/sample_risk_scores.csv`
- `data/interim/date_risk_summary.csv`
- `data/interim/tract_risk_summary.csv`
- summaries (`*_summary.json`)

## 13.3 Frontend outputs
- `frontend/data/spread_daily_compact.json`
- `frontend/data/spread_trajectory_compact.json`
- `frontend/data/daily_risk_summary.json`
- `frontend/data/tract_risk.geojson`
- etc.

---

## 14. Problems Faced and Resolutions (Observed During This Build)

1. Earth Engine project/auth setup friction
- Issue: project registration/auth mismatch and API enablement errors.
- Resolution: configured valid EE project and authenticated account, then reran exports.

2. Deprecation warnings in EE datasets
- VIIRS and MODIS dataset versions used by exporter are deprecated.
- Pipeline still runs, but future update should migrate IDs to latest EE catalogs.

3. Heavy preprocessing stage
- TFRecord decode + pickle writing is CPU-bound and long-running.
- TensorFlow attempted CUDA init on machines without visible TF CUDA support, causing warnings.
- Resolution: keep Stage-2 as CPU preprocessing and run with unbuffered logs.

4. Training instability / class imbalance sensitivity
- Fire pixels are sparse relative to no-fire, causing volatile positive-class metrics.
- Macro metrics and class-1 metrics can diverge significantly.
- Mitigation used:
  - `pos_weight=10`
  - regularization + scheduler + clipping
  - model selection by validation F1

5. Frontend performance challenges
- Full GeoJSON payloads caused slow rendering.
- Resolution:
  - compact JSON by date
  - backend API date windows
  - California bounds and map filtering

---

## 15. Drawbacks and Limitations

1. Label source/version limitations
- Export still relies on older EE MODIS/VIIRS asset IDs currently flagged deprecated.

2. Class imbalance remains strong
- Positive-class precision/F1 remains modest despite tuning.
- Accuracy is high but less informative due majority no-fire pixels.

3. Risk is a proxy, not full actuarial model
- `asset_value_usd` is approximated via `housing_units x median_home_value` at tract level.
- No explicit structure-type fragility curves or insurance model calibration.

4. Vulnerability simplification
- Vulnerability taken from tract-level SVI percentile and median-imputed when missing.

5. Spatial aggregation effects
- Sample tiles represent 64x64 km windows at 1 km internal resolution.
- Tract aggregation smooths local heterogeneity.

6. Trajectory visualization is analytical, not a physical fireline simulator
- Trajectories are cluster-linked centroids of predicted-positive samples.

---

## 16. Assumptions Taken

1. Hazard proxy
- `hazard_prob_mean` is used as sample-level hazard index in `[0,1]`.

2. Exposure monetary proxy
- `asset_value_usd = housing_units x median_home_value` approximates exposed asset stock.

3. Vulnerability proxy
- `svi_rpl_themes` interpreted as vulnerability damage fraction proxy in `[0,1]`.

4. Risk equation
- Primary daily monetary risk uses strict multiplicative form:
  - `risk_score = H x E x V`

5. Missing vulnerability
- Missing vulnerability is imputed by global median (`0.5359`).

6. Missing monetary exposure
- If asset value is missing/non-positive, risk is left null for that sample.

7. Date windowing for UI
- Dashboard uses next available dates (`t, t+1, t+2 available`) rather than filling absent dates.

8. Forecast horizon
- Hazard model is one-step next-day model; longer horizon visualization is achieved by stepping across available daily predictions.

---

## 17. Reproducibility (Concrete Commands)

From repository root:

1. Submit EE exports (full-year or filtered months):
```bash
EE_PROJECT=<your-project-id> bash NextDayWildFireSpr/tools/run_option_a_export_ca_2020.sh
```

2. After downloading TFRecords to `NextDayWildFireSpr/data/ndws64_meta_ca`, rebuild locally:
```bash
bash NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh
```

3. Manual training command used for final selected model:
```bash
cd NextDayWildFireSpr
python trainModel-II.py \
  --epochs 8 --batch_size 32 --num_workers 0 \
  --rotation_factor 1 --max_train_samples 0 --max_val_samples 0 \
  --amp --random_flip \
  --dataset_path data/next-day-wildfire-spread-ca-hazard \
  --channels_metadata data/next-day-wildfire-spread-ca-hazard/channels_metadata.json \
  --learning_rate 4e-4 --weight_decay 1e-4 --pos_weight 10 \
  --grad_clip 1.0 --lr_factor 0.6 --lr_patience 2 \
  --early_stop_patience 4 --min_delta 0.0005
```

4. Serve dashboard with backend API:
```bash
python NextDayWildFireSpr/tools/serve_frontend_api.py --host 127.0.0.1 --port 8080
```

---

## 18. What is Complete vs Future Work

Completed:
- Full 2020 California data generation with mapping metadata
- Canonical hazard dataset build
- Hazard model training + inference
- HEV feature construction
- Risk fusion outputs (sample/day/tract)
- Interactive dashboard + local API

Future improvements (recommended):
- Update deprecated EE source collections.
- Calibrate hazard thresholds and probability calibration.
- Add explicit uncertainty intervals.
- Add richer exposure inventories (building types, critical infrastructure classes).
- Add physical spread constraints for trajectory interpretation.

---

## 19. Final Conclusion

The project is implemented end-to-end with reproducible scripts and concrete outputs for California 2020. The final system generates daily next-day hazard predictions, fuses them with tract-level exposure/vulnerability to compute monetary risk, and serves operational map products (spread, trajectory, tract risk) through a local dashboard API stack.

All major values in this report are taken from the current code and generated artifacts in this workspace (`sample_manifest_summary.json`, `channels_metadata.json`, `*_summary.json`, training log/history files, and frontend asset summaries).

---

## 20. Appendix: Exact Data Schemas and Field Definitions

This section documents the exact field-level schemas from final generated artifacts.

### 20.1 `sample_manifest.csv` (8 columns)
Source: `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`

1. `sample_id`: deterministic sample key (`{split}_{source_file_stem}_{record_index:08d}`)
2. `split`: original TFRecord split (`train`/`eval`/`test`)
3. `sample_date`: date string (`YYYY-MM-DD`)
4. `start_day`: day offset from export `start_date`
5. `sample_lon`: sample point longitude (WGS84)
6. `sample_lat`: sample point latitude (WGS84)
7. `source_file`: source TFRecord filename
8. `record_index`: record index inside source TFRecord

### 20.2 `sample_index.csv` (10 columns)
Source: `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/sample_index.csv`

1. `output_split`: canonical split name (`train`/`validation`/`test`)
2. `split`: original manifest split (`train`/`eval`/`test`)
3. `array_index`: row index in canonical pickle arrays
4. `sample_id`
5. `sample_date`
6. `sample_lon`
7. `sample_lat`
8. `meta_day_of_year`: day-of-year integer
9. `source_file`
10. `record_index`

### 20.3 `sample_features_hev.csv` (34 columns)
Source: `NextDayWildFireSpr/data/interim/sample_features_hev.csv`

1. `sample_id`
2. `split`
3. `sample_date`
4. `start_day`
5. `sample_lon`
6. `sample_lat`
7. `source_file`
8. `record_index`
9. `GEOID`
10. `NAMELSAD`
11. `ALAND`
12. `AWATER`
13. `point_x_3310`
14. `point_y_3310`
15. `tract_match_method`
16. `tract_nearest_dist_m`
17. `acs_population`
18. `acs_housing_units`
19. `acs_median_home_value`
20. `svi_rpl_themes`
21. `svi_rpl_theme1`
22. `svi_rpl_theme2`
23. `svi_rpl_theme3`
24. `svi_rpl_theme4`
25. `tract_area_km2`
26. `exposure_pop_density_km2`
27. `exposure_housing_density_km2`
28. `road_nearest_dist_m`
29. `road_nearest_route_id`
30. `road_nearest_f_system`
31. `past_fire_count_5y_10km`
32. `past_fire_acres_5y_10km`
33. `days_since_fire_min_5y_10km`
34. `vulnerability_index`

### 20.4 `hazard_predictions.csv` (11 columns)
Source: `NextDayWildFireSpr/data/interim/hazard_predictions.csv`

1. `output_split`
2. `array_index`
3. `sample_id`
4. `sample_date`
5. `sample_lon`
6. `sample_lat`
7. `hazard_prob_mean`
8. `hazard_prob_p95`
9. `hazard_prob_max`
10. `hazard_pred_fire_frac`
11. `gt_fire_frac`

### 20.5 `sample_risk_scores.csv` (31 columns)
Source: `NextDayWildFireSpr/data/interim/sample_risk_scores.csv`

1. `sample_id`
2. `split`
3. `output_split`
4. `sample_date`
5. `sample_lon`
6. `sample_lat`
7. `GEOID`
8. `hazard_index`
9. `exposure_index`
10. `vulnerability_index`
11. `risk_score`
12. `risk_score_weighted`
13. `risk_hev_product`
14. `risk_tier`
15. `hazard_prob_mean`
16. `hazard_prob_p95`
17. `hazard_prob_max`
18. `hazard_pred_fire_frac`
19. `acs_population`
20. `acs_housing_units`
21. `asset_value_usd`
22. `risk_eal_usd`
23. `expected_property_loss_usd_weighted`
24. `expected_property_loss_usd_hev`
25. `expected_population_affected`
26. `expected_housing_units_affected`
27. `exposure_pop_density_km2`
28. `exposure_housing_density_km2`
29. `acs_median_home_value`
30. `road_nearest_dist_m`
31. `svi_rpl_themes`

### 20.6 `date_risk_summary.csv` (13 columns)
Source: `NextDayWildFireSpr/data/interim/date_risk_summary.csv`

1. `sample_date`
2. `samples`
3. `risk_score_mean`
4. `risk_score_weighted_mean`
5. `risk_hev_product_mean`
6. `hazard_index_mean`
7. `exposure_index_mean`
8. `vulnerability_index_mean`
9. `risk_eal_usd_sum`
10. `expected_property_loss_usd_weighted_sum`
11. `expected_property_loss_usd_hev_sum`
12. `expected_population_affected_sum`
13. `expected_housing_units_affected_sum`

### 20.7 `tract_risk_summary.csv` (15 columns)
Source: `NextDayWildFireSpr/data/interim/tract_risk_summary.csv`

1. `GEOID`
2. `samples`
3. `risk_score_mean`
4. `risk_score_weighted_mean`
5. `risk_hev_product_mean`
6. `risk_score_p90`
7. `hazard_index_mean`
8. `exposure_index_mean`
9. `vulnerability_index_mean`
10. `asset_value_usd_mean`
11. `risk_eal_usd_sum`
12. `expected_property_loss_usd_weighted_sum`
13. `expected_property_loss_usd_hev_sum`
14. `expected_population_affected_sum`
15. `expected_housing_units_affected_sum`

### 20.8 Core JSON schema top-level keys (final build)

1. `sample_manifest_summary.json`:
`input_dir`, `output_csv`, `totals`, `file_counts`
2. `channels_metadata.json`:
`source`, `tile_size`, `channel_names`, `num_channels`, `base_channels`, `meta_channels`, `data_dtype`, `label_dtype`, `meta_normalization`, `split_shapes`, `integrity`, `sample_index_csv`
3. `hazard_predictions_summary.json`:
`inputs`, `outputs`, `hazard_distribution`, `pixel_metrics`, `pixel_metrics_macro_like_training`, `per_split_metrics`
4. `risk_fusion_summary.json`:
`inputs`, `formula`, `fusion_weights_normalized`, `outputs`, `distributions`
5. `frontend_assets_summary.json`:
`inputs`, `outputs`, `counts`, `trajectory_weight_col`, `keep_null_tracts`, `simplify_tolerance`, `round_decimals`, `hazard_merge`, `map_extent`

### 20.9 Split naming consistency note

- Manifest and HEV tables use `split`: `train`/`eval`/`test`.
- Canonical hazard pickles and inference use `output_split`: `train`/`validation`/`test`.
- Mapping is deterministic and fixed:
`train -> train`, `eval -> validation`, `test -> test`.
