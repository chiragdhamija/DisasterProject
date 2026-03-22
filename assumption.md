# Assumptions and Preprocessing Log (Canonical Hazard Pipeline)

## Scope and Intent
- This document records the exact preprocessing and assumptions used for the **current canonical model pipeline**.
- Current model objective: **next-day wildfire hazard prediction**.
- Exposure and vulnerability are reserved for post-model risk fusion, not hazard-model training input.

## Canonical Data Sources Used for Model Training
- Mapped CA TFRecords: `NextDayWildFireSpr/data/ndws64_meta_ca`
- Sample manifest: `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`
- Built training dataset: `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard`

## Preprocessing Applied (Model Dataset)

### 1) California subset construction
- Source mapped TFRecords are filtered to California bounds before canonical training dataset generation.
- Split structure is preserved (`train`, `eval`, `test`).

### 2) Stable sample mapping
- A sample manifest is generated with deterministic sample identifiers.
- `sample_id` format used during build: `<split>_<source_file_stem>_<record_index>`.

### 3) Label and fire-history mask conversion to binary
- Both `FireMask` (target) and `PrevFireMask` (input channel) are converted to binary values.
- Conversion rule:
  - If source values are in `[-1, 1]`, fire = `value > 0`.
  - Otherwise (categorical MODIS-like values), fire = `value >= 7`.
- Final stored values: `0` (no fire), `1` (fire).

### 4) Feature channels included in canonical model
- Original NDWS-style channels (12):
  - `elevation`, `th`, `vs`, `tmmn`, `tmmx`, `sph`, `pr`, `pdsi`, `NDVI`, `population`, `erc`, `PrevFireMask`
- Metadata channels (3):
  - `meta_lon_z`, `meta_lat_z`, `meta_day_of_year_z`
- Total channels used by model: `15`.

### 5) Metadata feature engineering
- Metadata channels are derived from manifest values:
  - `sample_lon`
  - `sample_lat`
  - `sample_date` -> day of year (`1..366`)
- Standardization is performed using **train split statistics only**.
- Standardized values are broadcast to full tile shape (`64x64`) per sample.

### 6) Output tensor formats
- Input tensor shape: `(N, 15, 64, 64)` (`float32`)
- Label tensor shape: `(N, 64, 64)` (`float32`)
- Split naming in output pickles:
  - `train`
  - `validation` (from `eval`)
  - `test`

### 7) Index traceability
- `sample_index.csv` is written alongside pickles.
- It stores mapping from array index back to `sample_id`, date, lon/lat, source file, and TFRecord record index.

## Training-Time Assumptions
- Training script uses **all channels in the dataset** by default.
- No `selected_features` filtering is used in the canonical workflow.
- Model architecture is `U_Net` (`NextDayWildFireSpr/leejunhyun_unet_models.py`).
- Loss is `BCEWithLogitsLoss` with class weighting (`pos_weight` configurable).
- Optimizer is `AdamW` with configurable `learning_rate` and `weight_decay`.
- Learning rate uses `ReduceLROnPlateau` monitored on validation F1.
- Gradient clipping is applied (`--grad_clip`) to reduce unstable updates.
- Early stopping is enabled on validation F1 (`--early_stop_patience`, `--min_delta`).
- Training augmentation includes rotation; random horizontal/vertical flips are optional (`--random_flip`).
- Inference hazard probability is computed with `sigmoid(logits)`.
- Optional binary hazard map threshold is assumed to be `0.5` unless tuned later.

## Geospatial and External Data Assumptions (Risk Stage)
- Canonical projection for derived geospatial layers: `EPSG:3310`.
- `data/interim` is a derived workspace for geospatial/risk artifacts, not the direct model training dataset.
- Tract assignment for sample points:
  - Primary method: polygon intersection.
  - Fallback method: nearest tract for non-intersecting points.
- Risk fusion primary formula follows slide objective:
  - `risk_score = hazard_index * exposure_index * vulnerability_for_risk`
  - Exposure index is composed from scaled population density, housing density, median home value, and road proximity.
  - Vulnerability is sourced from SVI (`svi_rpl_themes`) and median-imputed when missing.
  - A weighted score is retained only as secondary comparison (`risk_score_weighted`).
- Monetary risk fields are currently **proxy estimates** (not insured-loss model outputs):
  - `asset_value_usd = acs_housing_units * acs_median_home_value`
  - `risk_eal_usd = hazard_index * asset_value_usd * vulnerability_for_risk`
  - `expected_property_loss_usd_hev = risk_eal_usd`
  - `expected_property_loss_usd_weighted = risk_score_weighted * asset_value_usd`
  - `expected_population_affected = risk_score * acs_population`
  - `expected_housing_units_affected = risk_score * acs_housing_units`
- ACS invalid/sentinel non-positive values are treated as missing before monetary calculations.

## What Is Explicitly NOT Used in Canonical Hazard Training
- External exposure/vulnerability feature channels are not included as model input.
- The intermediate dataset `next-day-wildfire-spread-hev` is considered non-canonical for final hazard training.

## Known Limitations / Practical Assumptions
- California subset filtering is done at sample level using spatial metadata; edge cases near state boundaries can exist.
- `PrevFireMask` and `FireMask` binarization assumes `>=7` is valid fire class for categorical mask variants.
- Base environmental channels are used as exported values; no additional per-channel z-normalization is applied in canonical builder beyond metadata channels.
- Nearest-tract fallback introduces an approximation for points outside tract polygons.

## Reproducibility Notes
- Canonical builder script: `NextDayWildFireSpr/tools/build_hazard_pickles.py`
- Canonical training script: `NextDayWildFireSpr/trainModel-II.py`
- Canonical channel definition source: `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/channels_metadata.json`
