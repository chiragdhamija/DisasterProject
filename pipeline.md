# Wildfire Hazard Pipeline (Canonical)

## Objective
Predict **next-day wildfire hazard** (fire spread probability) from wildfire/environmental inputs.

Final risk map is produced in a separate step:
- Hazard from model output
- Exposure from external socioeconomic/infrastructure layers
- Vulnerability from SVI/ACS indicators

Primary risk fusion follows:
`Risk = Hazard × Exposure × Vulnerability` (fusion done after hazard inference).

## Model Input
Current training dataset:
- `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard`
- Source: mapped CA TFRecords (`ndws64_meta_ca`) + sample metadata channels

Input tensor per sample:
- Shape: `(C, 64, 64)`
- Channels (`C=15`):
1. `elevation`
2. `th`
3. `vs`
4. `tmmn`
5. `tmmx`
6. `sph`
7. `pr`
8. `pdsi`
9. `NDVI`
10. `population`
11. `erc`
12. `PrevFireMask` (binary)
13. `meta_lon_z` (z-normalized lon, broadcast)
14. `meta_lat_z` (z-normalized lat, broadcast)
15. `meta_day_of_year_z` (z-normalized day-of-year, broadcast)

Target label per sample:
- Shape: `(64, 64)`
- Field: `FireMask` (binary 0/1)

## Model Output
- Architecture: `U_Net` from `NextDayWildFireSpr/leejunhyun_unet_models.py`
- Output shape: `(1, 64, 64)` logits per sample
- Inference probability map: `sigmoid(logits)`

## Final Model I/O (Exact)
- Model input at train/inference:
  - Tensor shape: `(N, 15, 64, 64)` (`float32`)
  - `N` is batch size.
  - 15 channels = 12 NDWS environmental/fire channels + 3 metadata channels (`meta_lon_z`, `meta_lat_z`, `meta_day_of_year_z`).
- Model output at inference:
  - Raw logits: `(N, 1, 64, 64)`
  - Hazard probability: `sigmoid(logits)` -> `(N, 1, 64, 64)`
  - Optional binary hazard mask: `probability >= threshold` (commonly `0.5`).
- Trained model artifact:
  - Saved best-weight file in `NextDayWildFireSpr/savedModels/`
  - Default filename pattern: `model-U_Net-bestF1Score-Rank-<rank>.weights`

## Training Setup
File: `NextDayWildFireSpr/trainModel-II.py`

Key behavior:
- Uses **all channels** from dataset (no feature subset flag required)
- Auto device: CUDA if available, else CPU fallback
- Loss: `BCEWithLogitsLoss` (with positive class weighting)
- Optimizer: `AdamW` (learning rate + weight decay configurable)
- LR policy: `ReduceLROnPlateau` on validation F1
- Stabilization: gradient clipping (`max_norm` configurable)
- Overfitting control: early stopping on validation F1 (`patience` + `min_delta`)
- Training-only augmentation: rotation (existing) + optional random horizontal/vertical flips (`--random_flip`)
- Validation metrics: Loss, IoU, Accuracy, F1, AUC, Dice, Precision, Recall

## Data Construction
Builder script:
- `NextDayWildFireSpr/tools/build_hazard_pickles.py`

What it does:
- Reads mapped TFRecords (`train/eval/test`)
- Binarizes `FireMask` and `PrevFireMask`
- Adds location/time channels from manifest metadata
- Writes pickles:
  - `train.data`, `train.labels`
  - `validation.data`, `validation.labels`
  - `test.data`, `test.labels`
- Writes metadata:
  - `channels_metadata.json`
  - `sample_index.csv` (array index to sample_id/date/lon/lat mapping)

## Risk Mapping Stage (Post-Training)
1. Run hazard inference to get hazard probability/index.
   - Script: `NextDayWildFireSpr/tools/infer_hazard_scores.py`
   - Output: `data/interim/hazard_predictions.csv`
2. Build HEV feature table from external datasets.
   - Script: `NextDayWildFireSpr/tools/build_hev_features.py`
   - Output: `data/interim/sample_features_hev.csv`
3. Fuse hazard + exposure + vulnerability into risk score.
   - Script: `NextDayWildFireSpr/tools/fuse_risk_scores.py`
   - Primary formula: `risk_score = hazard_index * exposure_index * vulnerability_for_risk`
   - Monetary EAL proxy: `risk_eal_usd = hazard_index * asset_value_usd * vulnerability_for_risk`
   - Outputs:
     - `data/interim/sample_risk_scores.csv`
     - `data/interim/tract_risk_summary.csv`
     - `data/interim/date_risk_summary.csv`
   - Includes monetary/impact proxy fields:
      - `asset_value_usd`
      - `risk_eal_usd`
      - `expected_property_loss_usd_weighted`
      - `expected_property_loss_usd_hev`
      - `expected_population_affected`
      - `expected_housing_units_affected`
4. Use tract/date/sample outputs for reporting and frontend map layers.
