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