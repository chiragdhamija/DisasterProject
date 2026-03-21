# Wildfire Hazard Pipeline (Canonical)

## Objective
Predict **next-day wildfire hazard** (fire spread probability) from wildfire/environmental inputs.

Final risk map is produced in a separate step:
- Hazard from model output
- Exposure from external socioeconomic/infrastructure layers
- Vulnerability from SVI/ACS indicators

`Risk = f(Hazard, Exposure, Vulnerability)` (fusion done after hazard inference).

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
1. Run hazard inference to get hazard probability map/index.
2. Compute exposure index from external datasets.
3. Compute vulnerability index from external datasets.
4. Fuse into composite risk score and classify tiers for reporting/visualization.
