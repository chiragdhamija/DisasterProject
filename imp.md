Great, you’re in good shape now.

I validated your new mapped export:
- Folder: `NextDayWildFireSpr/data/ndws64_meta_full`
- Files: `8` (`train=6`, `eval=1`, `test=1`)
- Schema: all required metadata fields present
- Metadata: valid (`sample_lon`, `sample_lat`, `sample_date`, `start_day`, `split`)

Important finding:
- Only about **20.38%** samples are inside California (`2827 / 13874`).

Since your external datasets are California-only, the next step should be:

1. Create a **CA-only filtered dataset** from `ndws64_meta_full` into a new folder (keep split names).  
2. Validate CA-only schema + counts.  
3. Generate a mapping table (`sample_id, split, date, lon, lat`) for joins.  
4. Start feature engineering joins (roads/fire/SVI/ACS) on this CA-only set.  
5. Update training pipeline to use enhanced channels.

If you want, I can do step 1 now and create `NextDayWildFireSpr/data/ndws64_meta_ca/` for you.



Implemented. I started execution and kept a running log in [progress.md](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/progress.md).

What is done now:

1. California subset + manifest pipeline
- Added [build_ca_subset.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/build_ca_subset.py)
- Added [build_sample_manifest.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/build_sample_manifest.py)
- Built `ndws64_meta_ca` and manifest (`2783` CA samples)

2. Geospatial preprocessing + tract mapping + HEV features
- Added [preprocess_geospatial_layers.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/preprocess_geospatial_layers.py)
- Added [build_sample_tract_join.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/build_sample_tract_join.py)
- Added [build_hev_features.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/build_hev_features.py)
- Outputs:
  - `NextDayWildFireSpr/data/interim/geospatial_3310/*`
  - `sample_tract_join.csv` with 100% mapping (`intersects=2630`, `nearest=153`)
  - `sample_features_hev.csv` (`2783` rows, 34 columns)

3. Model-integration block started
- Added [build_enhanced_pickles.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/build_enhanced_pickles.py)
- Created enhanced dataset:
  - `train: (2334, 21, 64, 64)`
  - `validation: (158, 21, 64, 64)`
  - `test: (291, 21, 64, 64)`
- Channel metadata saved at:
  - `NextDayWildFireSpr/data/next-day-wildfire-spread-hev/channels_metadata.json`

4. Training code made dynamic for channel count
- Updated [trainModel-II.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/trainModel-II.py)
- Added args:
  - `--dataset_path`
  - `--channels_metadata`
  - `--selected_features`
- `U_Net` input channels now come from selected channel list (not hardcoded 12)

5. Sanity tool fixed for your current folder layout
- Updated [sanity_check_readiness.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/sanity_check_readiness.py) to support `Ext_Datasets/`
- Re-run result: `PASS=10, WARN=1, FAIL=0` (warning is expected legacy archive schema warning)

6. Git hygiene
- Updated [.gitignore](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/.gitignore) for `__pycache__/` and `*.pyc`.

Next immediate implementation block:
1. Run enhanced smoke training (`1-2` epochs) on `next-day-wildfire-spread-hev`.
2. Build risk fusion outputs (hazard/exposure/vulnerability/composite) and tract rankings.
3. Then build the minimal frontend for California risk map visualization.


 confirmed the core issue: both FireMask and PrevFireMask in your TFRecords are categorical codes (3,5,8,9), not binary. I’m patching the converter to binarize these correctly (>=7 => fire) and adding a defensive AUC fix so training won’t crash again.



 Done. `interim` has been recreated cleanly.

Current `interim` now contains only:
- `geospatial_3310/fire_perimeters_3310.gpkg`
- `geospatial_3310/roads_3310.gpkg`
- `geospatial_3310/tracts_3310.gpkg`
- `geospatial_3310/geospatial_preprocess_summary.json`
- `sample_tract_join.csv`
- `sample_tract_join_summary.json`

Run this full canonical sequence from now on:

```bash
cd "/home/chirag/Desktop/SEM 8/Disaster/DisasterProject/Wildfire-Spread-Prediction"
```

1. Rebuild interim (clean)
```bash
rm -rf NextDayWildFireSpr/data/interim && mkdir -p NextDayWildFireSpr/data/interim

.venv/bin/python NextDayWildFireSpr/tools/preprocess_geospatial_layers.py \
  --ext_root Ext_Datasets \
  --output_dir NextDayWildFireSpr/data/interim/geospatial_3310

.venv/bin/python NextDayWildFireSpr/tools/build_sample_tract_join.py \
  --manifest NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv \
  --tracts NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg \
  --output_csv NextDayWildFireSpr/data/interim/sample_tract_join.csv \
  --summary_json NextDayWildFireSpr/data/interim/sample_tract_join_summary.json
```

2. Rebuild canonical hazard training dataset
```bash
.venv/bin/python NextDayWildFireSpr/tools/build_hazard_pickles.py \
  --tfrecord_dir NextDayWildFireSpr/data/ndws64_meta_ca \
  --manifest_csv NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv \
  --output_dir NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard \
  --metadata_json NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/channels_metadata.json \
  --sample_index_csv NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard/sample_index.csv \
  --tile_size 64
```

3. Train (all channels by default, no selected-features flag)
```bash
cd NextDayWildFireSpr

python trainModel-II.py --epochs 10 2>&1 | tee train_hazard_10ep.log
```

4. Share summary
```bash
grep -E "Using all channels|num_input_channels|Validation -|F1 Score|Best epoch|Best F1 score|Traceback|RuntimeError" train_hazard_10ep.log
```



Phase 2 is done and executed end-to-end.

1. Implemented hazard inference script: [infer_hazard_scores.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/infer_hazard_scores.py)
2. Implemented risk fusion script: [fuse_risk_scores.py](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/tools/fuse_risk_scores.py)
3. Updated docs/logs: [progress.md](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/progress.md), [pipeline.md](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/pipeline.md), [assumption.md](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/assumption.md)

Generated outputs:
1. Hazard predictions: [hazard_predictions.csv](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/hazard_predictions.csv)
2. Hazard summary: [hazard_predictions_summary.json](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/hazard_predictions_summary.json)
3. HEV features: [sample_features_hev.csv](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/sample_features_hev.csv)
4. Final sample risk: [sample_risk_scores.csv](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/sample_risk_scores.csv)
5. Tract risk summary: [tract_risk_summary.csv](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/tract_risk_summary.csv)
6. Date risk summary: [date_risk_summary.csv](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/date_risk_summary.csv)
7. Risk fusion summary: [risk_fusion_summary.json](/home/chirag/Desktop/SEM%208/Disaster/DisasterProject/Wildfire-Spread-Prediction/NextDayWildFireSpr/data/interim/risk_fusion_summary.json)

Rerun commands anytime:
```bash
cd NextDayWildFireSpr

../.venv/bin/python tools/build_hev_features.py
../.venv/bin/python tools/infer_hazard_scores.py --split all --batch_size 64 --weights savedModels/model-U_Net-bestF1Score-Rank-0.weights
../.venv/bin/python tools/fuse_risk_scores.py
```

Next phase ready: I can now build the minimal frontend to visualize California sample/tract risk maps from these CSV outputs.





Exactly. You are right.

With your current full-year dataset, fire pixels are very rare:

- `train fire fraction ≈ 0.2628%`
- `validation fire fraction ≈ 0.3970%`
- `test fire fraction ≈ 0.4164%`

So a trivial “always no-fire” model already gets about:

- `99.74%` train accuracy
- `99.60%` validation accuracy
- `99.58%` test accuracy

So `accuracy` is not useful here.

What matters:
1. `class-1 (fire) F1`
2. `class-1 IoU`
3. `class-1 recall`
4. `PR-AUC` (better than ROC-AUC for imbalance)

Your reported `F1 ~0.5209` is macro-like (includes class 0), so it can still look decent while fire-class performance is weak.

Most important fixes:
1. Increase `--pos_weight` (current `4` is too low for this imbalance).
2. Keep full dataset, but optimize for fire-class metrics.
3. Tune threshold on validation (don’t lock at `0.5`).

If you want, next full-dataset run with same setup but stronger imbalance handling:

```bash
python trainModel-II.py \
  --epochs 6 \
  --batch_size 32 \
  --num_workers 0 \
  --rotation_factor 1 \
  --max_train_samples 0 \
  --max_val_samples 0 \
  --amp \
  --dataset_path data/next-day-wildfire-spread-ca-hazard \
  --channels_metadata data/next-day-wildfire-spread-ca-hazard/channels_metadata.json \
  --learning_rate 7e-4 \
  --weight_decay 1e-4 \
  --pos_weight 20 \
  --grad_clip 1.0 \
  --lr_factor 0.6 \
  --lr_patience 2 \
  --early_stop_patience 4 \
  --min_delta 0.0005 \
  2>&1 | tee train_hazard_option_a_full_pos20.log
```

I can also patch training logs to print explicit fire-class-only metrics every epoch so we stop relying on misleading accuracy/macro-only numbers.