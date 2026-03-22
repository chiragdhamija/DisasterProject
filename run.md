# Reproducibility Runbook

This file documents how to run the full wildfire pipeline end-to-end and reproduce the current results.

## 1) Environment

Run from project root:

```bash
cd "/home/chirag/Desktop/SEM 8/Disaster/DisasterProject/Wildfire-Spread-Prediction"
```

Use the existing virtual env:

```bash
source .venv/bin/activate
```

## 1.1) Option A (Full-Year 2020 CA) quick scripts

### A) Submit Earth Engine exports (monthly, CA-only, daily)

```bash
EE_PROJECT=disaster-490916 \
bash NextDayWildFireSpr/tools/run_option_a_export_ca_2020.sh
```

After tasks finish, download all generated `train_*.tfrecord(.gz)`, `eval_*.tfrecord(.gz)`, `test_*.tfrecord(.gz)` files
from Drive/GCS into:

`NextDayWildFireSpr/data/ndws64_meta_ca`

### B) Rebuild full local pipeline from downloaded files

```bash
bash NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh
```

Headless/background variant (recommended for long Step 2 so VS Code closing does not stop run):

```bash
nohup env STEP2_DATA_DTYPE=float16 STEP2_LABEL_DTYPE=uint8 PYTHONUNBUFFERED=1 \
  bash NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh \
  > option_a_rebuild.log 2>&1 &

tail -f option_a_rebuild.log
```

## 2) Required Inputs (Must Exist)

### 2.1 Core mapped NDWS data

- `NextDayWildFireSpr/data/ndws64_meta_ca/train_*.tfrecord.gz`
- `NextDayWildFireSpr/data/ndws64_meta_ca/eval_*.tfrecord.gz`
- `NextDayWildFireSpr/data/ndws64_meta_ca/test_*.tfrecord.gz`
- `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`

### 2.2 External datasets

- `Ext_Datasets/acs_2020_exposure.json`
- `Ext_Datasets/SVI_2020_CaliforniaTract.csv`
- `Ext_Datasets/TIGER2020_CaliforniaTractsShapefile/*`
- `Ext_Datasets/CaliforniaRoads_InfraShapefile-CRS_-_Functional_Classification/*`
- `Ext_Datasets/California_Historic_Fire_Perimeters_-6273763535668926275/*`

## 3) (Optional) Build CA subset + manifest from full mapped export

If `ndws64_meta_ca` is already ready, skip this section.

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/build_ca_subset.py \
  --input_dir NextDayWildFireSpr/data/ndws64_meta_full_complete_US_GEE \
  --output_dir NextDayWildFireSpr/data/ndws64_meta_ca \
  --overwrite

.venv/bin/python NextDayWildFireSpr/tools/build_sample_manifest.py \
  --input_dir NextDayWildFireSpr/data/ndws64_meta_ca \
  --output_csv NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv \
  --summary_json NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest_summary.json
```

### Outputs

- `NextDayWildFireSpr/data/ndws64_meta_ca/*.tfrecord.gz`
- `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`
- `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest_summary.json`

## 4) Build canonical hazard training pickles

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/build_hazard_pickles.py \
  --tfrecord_dir NextDayWildFireSpr/data/ndws64_meta_ca \
  --manifest_csv NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv \
  --output_dir NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard \
  --metadata_json NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/channels_metadata.json \
  --sample_index_csv NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/sample_index.csv \
  --tile_size 64 \
  --data_dtype float16 \
  --label_dtype uint8
```

For live logs in non-interactive runs:

```bash
env PYTHONUNBUFFERED=1 .venv/bin/python -u NextDayWildFireSpr/tools/build_hazard_pickles.py ...
```

### Outputs

- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/train.data`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/train.labels`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/validation.data`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/validation.labels`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/test.data`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/test.labels`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/channels_metadata.json`
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/sample_index.csv`

## 5) Train hazard model (canonical command used)

Run from `NextDayWildFireSpr`:

### Command

```bash
cd NextDayWildFireSpr

python trainModel-II.py \
  --epochs 10 \
  --learning_rate 7e-4 \
  --weight_decay 1e-4 \
  --pos_weight 4 \
  --grad_clip 1.0 \
  --lr_factor 0.6 \
  --lr_patience 2 \
  --early_stop_patience 4 \
  --min_delta 0.0005 \
  2>&1 | tee train_hazard_10ep.log
```

### Outputs

- `NextDayWildFireSpr/savedModels/model-U_Net-bestF1Score-Rank-0.weights`
- `NextDayWildFireSpr/savedModels/train_loss_history.pkl`
- `NextDayWildFireSpr/savedModels/val_metrics_history.pkl`
- `NextDayWildFireSpr/train_hazard_10ep.log`

## 6) Build geospatial interim layers + tract join

Run from project root:

### Command

```bash
cd "/home/chirag/Desktop/SEM 8/Disaster/DisasterProject/Wildfire-Spread-Prediction"

.venv/bin/python NextDayWildFireSpr/tools/preprocess_geospatial_layers.py \
  --ext_root Ext_Datasets \
  --output_dir NextDayWildFireSpr/data/interim/geospatial_3310

.venv/bin/python NextDayWildFireSpr/tools/build_sample_tract_join.py \
  --manifest NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv \
  --tracts NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg \
  --output_csv NextDayWildFireSpr/data/interim/sample_tract_join.csv \
  --summary_json NextDayWildFireSpr/data/interim/sample_tract_join_summary.json
```

### Outputs

- `NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg`
- `NextDayWildFireSpr/data/interim/geospatial_3310/roads_3310.gpkg`
- `NextDayWildFireSpr/data/interim/geospatial_3310/fire_perimeters_3310.gpkg`
- `NextDayWildFireSpr/data/interim/geospatial_3310/geospatial_preprocess_summary.json`
- `NextDayWildFireSpr/data/interim/sample_tract_join.csv`
- `NextDayWildFireSpr/data/interim/sample_tract_join_summary.json`

## 7) Build HEV features

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/build_hev_features.py \
  --sample_tract_csv NextDayWildFireSpr/data/interim/sample_tract_join.csv \
  --roads NextDayWildFireSpr/data/interim/geospatial_3310/roads_3310.gpkg \
  --fire NextDayWildFireSpr/data/interim/geospatial_3310/fire_perimeters_3310.gpkg \
  --acs_json Ext_Datasets/acs_2020_exposure.json \
  --svi_csv Ext_Datasets/SVI_2020_CaliforniaTract.csv \
  --output_csv NextDayWildFireSpr/data/interim/sample_features_hev.csv \
  --summary_json NextDayWildFireSpr/data/interim/sample_features_hev_summary.json
```

### Outputs

- `NextDayWildFireSpr/data/interim/sample_features_hev.csv`
- `NextDayWildFireSpr/data/interim/sample_features_hev_summary.json`

## 8) Run hazard inference

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/infer_hazard_scores.py \
  --dataset_dir NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard \
  --channels_metadata NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/channels_metadata.json \
  --sample_index_csv NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/sample_index.csv \
  --weights NextDayWildFireSpr/savedModels/model-U_Net-bestF1Score-Rank-0.weights \
  --split all \
  --threshold 0.5 \
  --batch_size 64 \
  --output_csv NextDayWildFireSpr/data/interim/hazard_predictions.csv \
  --summary_json NextDayWildFireSpr/data/interim/hazard_predictions_summary.json
```

### Outputs

- `NextDayWildFireSpr/data/interim/hazard_predictions.csv`
- `NextDayWildFireSpr/data/interim/hazard_predictions_summary.json`

## 9) Fuse risk (`R = H × E × V`) + EAL proxy

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/fuse_risk_scores.py \
  --hazard_csv NextDayWildFireSpr/data/interim/hazard_predictions.csv \
  --hev_csv NextDayWildFireSpr/data/interim/sample_features_hev.csv \
  --output_sample_csv NextDayWildFireSpr/data/interim/sample_risk_scores.csv \
  --output_tract_csv NextDayWildFireSpr/data/interim/tract_risk_summary.csv \
  --output_date_csv NextDayWildFireSpr/data/interim/date_risk_summary.csv \
  --summary_json NextDayWildFireSpr/data/interim/risk_fusion_summary.json
```

### Outputs

- `NextDayWildFireSpr/data/interim/sample_risk_scores.csv`
- `NextDayWildFireSpr/data/interim/tract_risk_summary.csv`
- `NextDayWildFireSpr/data/interim/date_risk_summary.csv`
- `NextDayWildFireSpr/data/interim/risk_fusion_summary.json`

## 10) Important interim files to keep for report/demo

- `NextDayWildFireSpr/data/interim/hazard_predictions_summary.json`
- `NextDayWildFireSpr/data/interim/risk_fusion_summary.json`
- `NextDayWildFireSpr/data/interim/sample_risk_scores.csv`
- `NextDayWildFireSpr/data/interim/tract_risk_summary.csv`
- `NextDayWildFireSpr/data/interim/date_risk_summary.csv`
- `NextDayWildFireSpr/savedModels/model-U_Net-bestF1Score-Rank-0.weights`
- `NextDayWildFireSpr/train_hazard_10ep.log`

## 11) Build frontend map assets (spread + trajectory + risk map)

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/build_frontend_assets.py \
  --sample_risk_csv NextDayWildFireSpr/data/interim/sample_risk_scores.csv \
  --hazard_csv NextDayWildFireSpr/data/interim/hazard_predictions.csv \
  --date_summary_csv NextDayWildFireSpr/data/interim/date_risk_summary.csv \
  --tract_risk_csv NextDayWildFireSpr/data/interim/tract_risk_summary.csv \
  --tracts_gpkg NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg \
  --output_dir NextDayWildFireSpr/frontend/data \
  --summary_json NextDayWildFireSpr/frontend/data/frontend_assets_summary.json \
  --trajectory_weight_col risk_score \
  --simplify_tolerance 0.001 \
  --round_decimals 5
```

### Outputs

- `NextDayWildFireSpr/frontend/data/spread_daily_compact.json`
- `NextDayWildFireSpr/frontend/data/spread_trajectory_compact.json`
- `NextDayWildFireSpr/frontend/data/spread_points.geojson`
- `NextDayWildFireSpr/frontend/data/spread_trajectory.geojson`
- `NextDayWildFireSpr/frontend/data/daily_risk_summary.json`
- `NextDayWildFireSpr/frontend/data/tract_risk.geojson`
- `NextDayWildFireSpr/frontend/data/frontend_assets_summary.json`

Notes:
- Frontend now loads compact spread/trajectory JSON first.
- Tract risk GeoJSON is loaded lazily only when `Tract Risk Map` is selected.
- Map extent is restricted to California in frontend.
- Trajectory payload now includes:
  - legacy `centroids` (single weighted centroid path),
  - `trajectories` (multi-track, cluster-linked paths) used by the UI.
- Base date map is rendered from `gt_fire_frac` (observed fire only); future days use predicted spread (`hazard_pred_fire_frac`).

## 12) Run frontend locally

### Command

```bash
.venv/bin/python NextDayWildFireSpr/tools/serve_frontend_api.py \
  --host 127.0.0.1 \
  --port 8080 \
  --horizon_default 2
```

Open:

- `http://localhost:8080`

API endpoints (served by same process):

- `/api/meta`
- `/api/window?date=YYYY-MM-DD&horizon=2` (uses next **available sampled dates**)
- `/api/tract-risk?date=YYYY-MM-DD`

Option B note:
- Frontend base-date selector now lists only dates available in the dataset.
- If dates are missing in the underlying sampled data, they will not appear in selector.

Frontend files used:

- `NextDayWildFireSpr/frontend/index.html`
- `NextDayWildFireSpr/frontend/app.js`
- `NextDayWildFireSpr/frontend/styles.css`
- `NextDayWildFireSpr/frontend/data/*` (generated by step 11)

## 13) Notes on metric interpretation

- Training log F1/IoU are macro-like metrics (class 0 + class 1).
- Inference summary includes:
  - `pixel_metrics` (class-1 positive-fire metrics)
  - `pixel_metrics_macro_like_training` (comparable to training-style reporting)
  - `per_split_metrics` (train/validation/test separately)
- Daily Snapshot cards in frontend:
  - `Samples`: number of sample tiles on that date (`date_risk_summary.samples`)
  - `Mean Hazard`: mean `hazard_index` on that date
  - `Mean Risk (USD/day/sample)`: mean per-sample `risk_score` on that date
  - `EAL Total (USD/day)`: sum of `risk_eal_usd` on that date
- Current risk fusion formula in code (`fuse_risk_scores.py`):
  - `risk_score = hazard_index * asset_value_usd * vulnerability_for_risk`
  - `risk_eal_usd = risk_score`
