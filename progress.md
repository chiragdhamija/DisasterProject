# Progress Log (Canonical Approach)

## Date
2026-03-22

## Objective
Build a **hazard-first wildfire pipeline**:
1. Train model only on wildfire/environmental + location/time inputs from mapped NDWS data.
2. Keep exposure/vulnerability for post-model risk fusion (not hazard training input).

## Option A (Full-Year Coverage) Update
- Implemented Option A exporter changes in:
  - `NextDayWildFireSpr/tools/ee_export_with_mapping.py`
- Key fixes applied:
  - Region control with default `--region ca` (California-only sampling).
  - Daily split window default `--split_window_days 1` (no 8-day omission pattern).
  - No-fire day retention via `--no_fire_samples_per_day`.
  - More robust detection-count handling (`bestEffort`, fallback sampling on reduce errors).
- Added runnable scripts:
  - `NextDayWildFireSpr/tools/run_option_a_export_ca_2020.sh`
  - `NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh`
- Added runbook shortcuts in `run.md` for Option A export + full rebuild.

## Step-2 Stability Fix (Full-Year 2020 Scale)
- Observed failure mode on full-year manifest (`61,064` samples, `50,142` train): Step 2 could terminate mid-write with only partial outputs (for example `train.data` at `0` bytes) due high memory pressure.
- Implemented low-memory rebuild changes:
  - `NextDayWildFireSpr/tools/build_hazard_pickles.py`
    - Processes splits sequentially (`train` then `validation` then `test`) instead of holding all splits in RAM.
    - Added storage dtype controls: `--data_dtype` (`float32|float16`) and `--label_dtype` (`float32|float16|uint8`).
    - Defaulted to `float16` features + `uint8` labels to reduce peak memory.
    - Added per-split allocation logging and atomic temp-file writes (`*.tmp -> final`).
  - `NextDayWildFireSpr/tools/infer_hazard_scores.py`
    - Reduced peak RAM by casting to `float32` per batch during inference (instead of full split copy).
  - `NextDayWildFireSpr/trainModel-II.py`
    - Explicitly casts input tensors to `float32` before forward pass so training remains stable with float16-stored pickles.
  - `NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh`
    - Exposes `STEP2_DATA_DTYPE` and `STEP2_LABEL_DTYPE` env controls (defaults: `float16`, `uint8`).
- Smoke validation completed on a subset (`3040` samples): pickles + metadata + index all generated correctly with zero parse/integrity errors.

## Training Throughput Controls (Implemented)
- Added fast-mode controls for large full-year training in:
  - `NextDayWildFireSpr/trainModel-II.py`
  - `NextDayWildFireSpr/datasets.py`
- New CLI options:
  - `--rotation_factor` (`1..4`) to control 4x rotation expansion.
  - `--max_train_samples` to cap base train samples before rotation.
  - `--max_val_samples` to cap validation samples.
  - `--amp` for CUDA mixed precision speedup.
- Dataset internals updated:
  - Optional deterministic subsampling of `good_indices`.
  - Rotation pipeline now supports reduced rotation sets; `rotation_factor=1` disables extra rotations.
- Smoke test passed with fast settings (`rotation_factor=1`, capped train/val):
  - Training started immediately and completed 1 epoch successfully.

## Metric Stability / Imbalance Note
- Verified severe class imbalance in full-year canonical labels:
  - train fire-pixel fraction: `0.002628` (~`0.26%`)
  - validation fire-pixel fraction: `0.003970`
  - test fire-pixel fraction: `0.004164`
- Patched `NextDayWildFireSpr/metrics.py` (`mean_iou`) to avoid `NaN` when a class union is zero in a batch (returns stable value instead of dividing by zero).

## Canonical Implemented Pipeline

### 1) CA Mapped Dataset Preparation
- Built CA-only mapped TFRecord subset:
  - `NextDayWildFireSpr/data/ndws64_meta_ca`
- Generated sample manifest:
  - `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`
  - `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest_summary.json`
- Validated schema and metadata mapping (`sample_lon`, `sample_lat`, `sample_date`, `split`, `start_day`).

### 2) Geospatial Standardization (for risk stage)
- Standardized layers to `EPSG:3310`:
  - `NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg`
  - `NextDayWildFireSpr/data/interim/geospatial_3310/roads_3310.gpkg`
  - `NextDayWildFireSpr/data/interim/geospatial_3310/fire_perimeters_3310.gpkg`

### 3) Canonical Hazard Training Dataset (Current)
- Built with script:
  - `NextDayWildFireSpr/tools/build_hazard_pickles.py`
- Output dataset:
  - `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard`
  - Files: `train/validation/test .data + .labels`
  - Metadata: `channels_metadata.json`
  - Index map: `sample_index.csv`
- Input channels (`15` total):
  - Original NDWS bands (`12`): `elevation, th, vs, tmmn, tmmx, sph, pr, pdsi, NDVI, population, erc, PrevFireMask`
  - Metadata channels (`3`): `meta_lon_z, meta_lat_z, meta_day_of_year_z`
- Labels:
  - `FireMask` binarized to `0/1`
- Shapes:
  - `train: (2334, 15, 64, 64)`
  - `validation: (158, 15, 64, 64)`
  - `test: (291, 15, 64, 64)`

### 4) Training Code Alignment
- Updated `NextDayWildFireSpr/trainModel-II.py`:
  - Default dataset path now points to canonical hazard dataset.
  - Uses **all channels** by default (no feature subset workflow required).
  - Auto device fallback (CUDA if available, else CPU).
- Updated `NextDayWildFireSpr/metrics.py`:
  - Safe binary handling for AUC path.

### 5) Readiness and Sanity
- Patched `NextDayWildFireSpr/tools/sanity_check_readiness.py` for `Ext_Datasets` layout.
- Current sanity result: `PASS=10, WARN=1, FAIL=0`
  - Warning is expected legacy archive schema warning.

## Documentation Added
- `pipeline.md` (model I/O, architecture, pipeline flow).
- `dataset.md` (exact dataset inventory including `Ext_Datasets`).

## Current Status
- Canonical hazard-model training data and code are ready.
- Next active step: run full training on `next-day-wildfire-spread-ca-hazard`, then generate hazard outputs for risk fusion.
- `data/interim` has been recreated clean from scratch (only canonical files retained):
  - `geospatial_3310/*`
  - `sample_tract_join.csv`
  - `sample_tract_join_summary.json`

## Latest Training Run (Canonical Hazard, 5 Epochs)
- Command run:
  - `python trainModel-II.py --epochs 5`
- Dataset used:
  - `data/next-day-wildfire-spread-ca-hazard`
- Channel config:
  - `15` channels (12 NDWS + 3 metadata channels), all used.
- Validation trajectory:
  - Epoch 1: `F1=0.4983`, `AUC=0.9261`, `IoU=0.4967`
  - Epoch 2: `F1=0.6294`, `AUC=0.9405`, `IoU=0.5676`
  - Epoch 3: `F1=0.6518`, `AUC=0.9435`, `IoU=0.5885`  <-- best F1
  - Epoch 4: `F1=0.6208`, `AUC=0.9453`, `IoU=0.5618`
  - Epoch 5: `F1=0.6275`, `AUC=0.9489`, `IoU=0.5716`
- Best checkpoint:
  - `Best epoch: 2` (0-based index, i.e. epoch 3)
  - `Best F1 score: 0.6518347859382629`

## Overfitting Mitigation Update (Implemented)
- Updated trainer (`NextDayWildFireSpr/trainModel-II.py`) with:
  - `AdamW` optimizer (`--learning_rate`, `--weight_decay`)
  - `ReduceLROnPlateau` scheduler on validation F1 (`--lr_factor`, `--lr_patience`)
  - Gradient clipping (`--grad_clip`)
  - Early stopping on validation F1 (`--early_stop_patience`, `--min_delta`)
  - Configurable `--pos_weight` for class imbalance
- Updated dataset augmentation (`NextDayWildFireSpr/datasets.py`):
  - Added training-only random horizontal/vertical flips in `RotatedWildfireDataset`
  - Flip augmentation is now optional via `--random_flip` (default off)
  - Existing 0/90/180/270 rotation augmentation retained
- Sanity checked:
  - Syntax compile passed for modified files.
  - Zero-epoch dry run completed with canonical dataset and channel mapping.

## Latest Regularized Run (8 Epoch Budget, Early Stop at 5)
- Command profile:
  - `learning_rate=8e-4`, `weight_decay=3e-4`, `pos_weight=5`, `grad_clip=1.0`
  - `lr_factor=0.5`, `lr_patience=1`, `early_stop_patience=3`, `min_delta=0.001`
- Outcome:
  - Early stopping triggered at epoch 5.
  - Best epoch: 1 (0-based, i.e. epoch 2)
  - Best F1: `0.6350`
- Interpretation:
  - Overfitting control works (late-epoch overtraining prevented),
  - but this setting underperforms the prior best canonical run (`F1=0.6518`), so regularization is currently too strong.

## Latest Tuned Run (10 Epoch Budget, Early Stop at 6)
- Command profile:
  - `learning_rate=7e-4`, `weight_decay=1e-4`, `pos_weight=4`, `grad_clip=1.0`
  - `lr_factor=0.6`, `lr_patience=2`, `early_stop_patience=4`, `min_delta=0.0005`
  - `random_flip=False`
- Outcome:
  - Early stopping triggered at epoch 6.
  - Best epoch: 1 (0-based, i.e. epoch 2)
  - Best F1: `0.6507`
  - Best IoU: `0.5860`
  - Best AUC: `0.9317`
- Interpretation:
  - This recovered most of the previous performance while still preventing late-epoch degradation.
  - It remains marginally below historical best (`F1=0.6518`), difference `0.0011`.
  - Practical decision: treat current checkpoint as acceptable and proceed to hazard inference + risk fusion.

## Phase 2 Implemented: Hazard Inference + Risk Fusion

### A) Hazard Inference Script
- Added script:
  - `NextDayWildFireSpr/tools/infer_hazard_scores.py`
- What it does:
  - Loads canonical dataset pickles (`train/validation/test`)
  - Loads trained `U_Net` checkpoint
  - Runs inference and writes per-sample hazard outputs:
    - `hazard_prob_mean`, `hazard_prob_p95`, `hazard_prob_max`, `hazard_pred_fire_frac`
  - Joins prediction rows to `sample_index.csv` metadata (`sample_id`, date, lon/lat)
  - Writes global pixel metric summary at chosen threshold

### B) HEV Feature Table Build
- Executed:
  - `NextDayWildFireSpr/tools/build_hev_features.py`
- Output:
  - `NextDayWildFireSpr/data/interim/sample_features_hev.csv` (`2783` rows)
  - `NextDayWildFireSpr/data/interim/sample_features_hev_summary.json`

### C) Risk Fusion Script
- Added script:
  - `NextDayWildFireSpr/tools/fuse_risk_scores.py`
- Fusion logic:
  - Hazard index from `hazard_prob_mean` (`0..1`)
  - Exposure index from scaled exposure features:
    - population density, housing density, median home value, road proximity
  - Vulnerability index from `svi_rpl_themes`
  - Final primary risk score (slide-aligned):
    - `risk_score = hazard * exposure * vulnerability`
  - Weighted score retained as secondary comparison:
    - `risk_score_weighted = 0.5 * hazard + 0.3 * exposure + 0.2 * vulnerability`
  - Risk tier classification into 5 quantile bands:
    - `very_low`, `low`, `moderate`, `high`, `very_high`

### D) Outputs Generated (Current)
- Hazard:
  - `NextDayWildFireSpr/data/interim/hazard_predictions.csv` (`2783` rows)
  - `NextDayWildFireSpr/data/interim/hazard_predictions_summary.json`
- Risk:
  - `NextDayWildFireSpr/data/interim/sample_risk_scores.csv` (`2783` rows)
  - `NextDayWildFireSpr/data/interim/tract_risk_summary.csv` (`334` tracts)
  - `NextDayWildFireSpr/data/interim/date_risk_summary.csv` (`199` dates)
  - `NextDayWildFireSpr/data/interim/risk_fusion_summary.json`

### E) Current Phase-2 Metrics Snapshot
- Hazard inference summary:
  - `rows=2783`, `threshold=0.5`
  - positive-class pixel `F1=0.2358`, `IoU=0.1336`, `Accuracy=0.9480`, `Brier=0.0395`
  - macro (training-like) across all splits: `macro_F1=0.6044`, `macro_IoU=0.5406`
  - validation split macro (training-comparable): `macro_F1=0.6508`, `macro_IoU=0.5836`
  - Note: low positive-class F1 is expected under heavy class imbalance; training logs use macro-style metrics.
- Risk summary:
  - `risk_score_mean=0.01793`, `risk_score_p95=0.05232`
  - tier counts balanced by quantile design

### F) Monetary Risk Outputs (Implemented)
- Added explicit economic-impact fields in risk fusion (`tools/fuse_risk_scores.py`):
  - `asset_value_usd = acs_housing_units * acs_median_home_value`
  - `risk_eal_usd = hazard * asset_value_usd * vulnerability`
  - `expected_property_loss_usd_hev = risk_eal_usd`
  - `expected_property_loss_usd_weighted = risk_score_weighted * asset_value_usd` (secondary)
  - `expected_population_affected = risk_score * acs_population`
  - `expected_housing_units_affected = risk_score * acs_housing_units`
- These fields are now present in:
  - `sample_risk_scores.csv`
  - `tract_risk_summary.csv` (sum aggregations)
  - `date_risk_summary.csv` (sum aggregations)
- Current totals (`risk_fusion_summary.json`):
  - `risk_eal_usd_total = 27,320,949,901.72`
  - `expected_property_loss_usd_weighted_total = 394,742,449,515.56` (secondary)
  - `expected_population_affected_total = 171,751.97`
  - `expected_housing_units_affected_total = 84,256.97`

## Next Steps
1. Build minimal frontend for California risk map visualization (sample + tract layers).
2. Add scenario analysis view (for example reduced vulnerability case and delta-EAL).
3. Finalize report figures/tables from `sample_risk_scores`, `tract_risk_summary`, and `date_risk_summary`.

## Phase 3 Implemented: Frontend Visualization

### A) Frontend Asset Builder
- Added/updated:
  - `NextDayWildFireSpr/tools/build_frontend_assets.py`
- Generated map-ready artifacts:
  - `NextDayWildFireSpr/frontend/data/spread_points.geojson`
  - `NextDayWildFireSpr/frontend/data/spread_trajectory.geojson`
  - `NextDayWildFireSpr/frontend/data/daily_risk_summary.json`
  - `NextDayWildFireSpr/frontend/data/tract_risk.geojson`
  - `NextDayWildFireSpr/frontend/data/frontend_assets_summary.json`
- Optimization applied:
  - Tract layer now exports only tracts with risk values (`334` features).
  - Geometry simplification supported via `--simplify_tolerance`; current run used `0.0002`.
  - Tract GeoJSON reduced from ~`119MB` to ~`3.9MB`.

### B) Frontend UI (Implemented)
- Added:
  - `NextDayWildFireSpr/frontend/index.html`
  - `NextDayWildFireSpr/frontend/styles.css`
  - `NextDayWildFireSpr/frontend/app.js`
- UI capabilities:
  - California map with **daily predicted spread points** (risk-colored).
  - **Day slider + play/pause animation** to step through wildfire progression.
  - **Spread trajectory** line/centroid progression across days.
  - Switchable **tract risk choropleth** view.
  - Daily KPI cards (`samples`, `hazard`, `risk`, `EAL`).
  - Time-series chart for risk/hazard/EAL across dates.

### C) Reproducibility Docs Updated
- Updated:
  - `run.md`
- Added explicit commands for:
  - frontend asset generation
  - local frontend server startup

### D) Frontend Performance Optimization (California-only)
- Issue addressed:
  - Browser lag/hang due to heavier frontend parsing and unrestricted map extent.
- Changes implemented:
  - Added compact backend assets:
    - `frontend/data/spread_daily_compact.json`
    - `frontend/data/spread_trajectory_compact.json`
  - Frontend now uses compact spread/trajectory files instead of parsing full GeoJSON first.
  - Tract risk layer is lazy-loaded only when user switches to `Tract Risk Map`.
  - Map extent is now constrained to California bounds only.
  - Default tract geometry simplification increased to `0.001` in asset builder.
  - Added local backend API server:
    - `NextDayWildFireSpr/tools/serve_frontend_api.py`
    - Serves static frontend plus endpoints:
      - `/api/meta`
      - `/api/window?date=YYYY-MM-DD&horizon=2`
      - `/api/tract-risk?date=YYYY-MM-DD`
  - Frontend now uses a **calendar base date (`t`)** and loads only `t..t+2` window from backend.
- Result:
  - Total frontend data footprint reduced and initial render path is faster.

### E) Option-B Date Mapping (Implemented)
- Frontend now uses only dataset-available dates for base-date selection.
- `/api/window` now returns `t, t+1, t+2` as next available sampled dates (not missing calendar dates).
- This prevents empty-map confusion on missing dates (for example unavailable 2020-09-09 samples).
