# Progress Log (Canonical Approach)

## Date
2026-03-22

## Objective
Build a **hazard-first wildfire pipeline**:
1. Train model only on wildfire/environmental + location/time inputs from mapped NDWS data.
2. Keep exposure/vulnerability for post-model risk fusion (not hazard training input).

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

## Next Steps
1. Re-run with milder regularization and flips off by default (new `--random_flip` flag is optional).
2. Freeze best hazard model and generate hazard probabilities per sample/tile.
3. Build exposure and vulnerability indices from external layers.
4. Fuse `Hazard + Exposure + Vulnerability` into final California risk map.
5. Build minimal frontend for map visualization/reporting.
