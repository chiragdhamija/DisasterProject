# Progress Log

## 2026-03-22

### Session Start
- Objective confirmed: start implementation from plan and maintain a running log.
- Current focus block:
  1. Build California-only subset from mapped TFRecords.
  2. Build sample manifest for reproducible joins.
  3. Validate generated artifacts.

### Completed
- Reviewed implementation notes (`plan.md`, `imp.md`) and confirmed immediate execution block.
- Implemented CA subset builder script:
  - `NextDayWildFireSpr/tools/build_ca_subset.py`
  - Purpose: filter mapped TFRecords to California bounds while preserving split/file structure.
  - Output summary file: `NextDayWildFireSpr/data/ndws64_meta_ca/ca_subset_summary.json`
- Implemented sample manifest builder script:
  - `NextDayWildFireSpr/tools/build_sample_manifest.py`
  - Purpose: generate reproducible sample table for all joins.
  - Output files:
    - `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`
    - `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest_summary.json`

### Executed Commands
- `.venv/bin/python NextDayWildFireSpr/tools/build_ca_subset.py --input_dir NextDayWildFireSpr/data/ndws64_meta_full --output_dir NextDayWildFireSpr/data/ndws64_meta_ca --strict_metadata --overwrite`
- `.venv/bin/python NextDayWildFireSpr/tools/build_sample_manifest.py --input_dir NextDayWildFireSpr/data/ndws64_meta_ca --output_csv NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv --summary_json NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest_summary.json --fail_on_missing_metadata`

### Validation Results
- CA subset totals:
  - `records_total=13874`
  - `records_kept=2783`
  - `records_dropped_outside_ca=11091`
  - `records_missing_coords=0`
  - `split_mismatches=0`
  - `ca_keep_ratio=0.200591`
- Manifest totals:
  - `rows=2783`
  - `split_counts(train=2334, eval=158, test=291, unknown=0)`
  - `missing_required_metadata=0`
  - `split_mismatches=0`
- Extra strict check:
  - All manifest points within CA bounds (`out_of_bounds=0`).
  - `wc -l sample_manifest.csv = 2784` (header + 2783 rows).

### Current Dataset Status
- Source mapped dataset: `NextDayWildFireSpr/data/ndws64_meta_full` (US-wide with metadata).
- Active modeling dataset for HEV joins: `NextDayWildFireSpr/data/ndws64_meta_ca` (California-only).
- Reproducibility key established: `sample_manifest.csv` (`sample_id`, split, date, lon, lat, source file, record index).

### Next Implementation Blocks
1. Reproject and clean vector layers to canonical CRS (`EPSG:3310`) into `data/interim`.
2. Build tract attachment pipeline (point-in-polygon: sample -> tract GEOID) using manifest lon/lat.
3. Build Exposure features per sample (ACS + road proximity/density + optional infrastructure proxies).
4. Build Vulnerability features per sample (SVI + ACS normalization/themes).
5. Build Hazard-history enrichments with time-safe windows (`<= sample_date` only).
6. Generate model-ready tensors/channels + normalization metadata.
7. Refactor training pipeline to config-driven input channels and run baseline vs HEV experiments.
8. Produce hazard/exposure/vulnerability/composite risk maps for California.
9. Final stage: build a minimal frontend to visualize maps, tract ranking, and per-layer toggles.

### Completed (Execution Block 2)
- Implemented geospatial preprocessing script:
  - `NextDayWildFireSpr/tools/preprocess_geospatial_layers.py`
  - Reprojects and cleans tracts/roads/fire perimeters to canonical `EPSG:3310`.
  - Outputs:
    - `NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg`
    - `NextDayWildFireSpr/data/interim/geospatial_3310/roads_3310.gpkg`
    - `NextDayWildFireSpr/data/interim/geospatial_3310/fire_perimeters_3310.gpkg`
    - `NextDayWildFireSpr/data/interim/geospatial_3310/geospatial_preprocess_summary.json`
- Implemented sample-to-tract join script:
  - `NextDayWildFireSpr/tools/build_sample_tract_join.py`
  - Includes nearest-tract fallback for points not intersecting tract polygons.
  - Outputs:
    - `NextDayWildFireSpr/data/interim/sample_tract_join.csv`
    - `NextDayWildFireSpr/data/interim/sample_tract_join_summary.json`
- Implemented HEV feature table script:
  - `NextDayWildFireSpr/tools/build_hev_features.py`
  - Builds per-sample Exposure + Vulnerability + time-safe Hazard-history features.
  - Outputs:
    - `NextDayWildFireSpr/data/interim/sample_features_hev.csv`
    - `NextDayWildFireSpr/data/interim/sample_features_hev_summary.json`

### Executed Commands (Block 2)
- `.venv/bin/python NextDayWildFireSpr/tools/preprocess_geospatial_layers.py --ext_root Ext_Datasets --output_dir NextDayWildFireSpr/data/interim/geospatial_3310`
- `.venv/bin/python NextDayWildFireSpr/tools/build_sample_tract_join.py --manifest NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv --tracts NextDayWildFireSpr/data/interim/geospatial_3310/tracts_3310.gpkg --output_csv NextDayWildFireSpr/data/interim/sample_tract_join.csv --summary_json NextDayWildFireSpr/data/interim/sample_tract_join_summary.json`
- `.venv/bin/python NextDayWildFireSpr/tools/build_hev_features.py --sample_tract_csv NextDayWildFireSpr/data/interim/sample_tract_join.csv --roads NextDayWildFireSpr/data/interim/geospatial_3310/roads_3310.gpkg --fire NextDayWildFireSpr/data/interim/geospatial_3310/fire_perimeters_3310.gpkg --acs_json Ext_Datasets/acs_2020_exposure.json --svi_csv Ext_Datasets/SVI_2020_CaliforniaTract.csv --output_csv NextDayWildFireSpr/data/interim/sample_features_hev.csv --summary_json NextDayWildFireSpr/data/interim/sample_features_hev_summary.json --fire_lookback_years 5 --fire_buffer_m 10000`

### Validation Results (Block 2)
- Geospatial standardized layers:
  - `tracts=9129`, `roads=779834`, `fire_perimeters=22810`, all in `EPSG:3310`.
- Sample-to-tract mapping:
  - `rows_total=2783`, `rows_matched=2783`, `rows_unmatched=0`, `match_rate=1.0`.
  - `tract_match_method`: `intersects=2630`, `nearest=153`.
- HEV feature table:
  - `rows=2783`, `cols=34`, `sample_id duplicates=0`.
  - Missing rates:
    - `geoid_missing=0`
    - `acs_population_missing=0`
    - `svi_rpl_themes_missing=1 row`
    - `road_nearest_dist_m_missing=0`
  - Fire-history feature range:
    - `past_fire_count_5y_10km`: min `0`, max `23`.

### Updated Next Blocks
1. Build model-ready channel tensor generation from `sample_features_hev.csv` + NDWS base channels.
2. Refactor training loader/model config for dynamic channel counts and run first enhanced smoke training.
3. Create risk fusion pipeline outputs: hazard map, exposure index, vulnerability index, composite risk index.
4. Build minimal frontend at final stage to visualize California maps and rankings.

### Completed (Execution Block 3)
- Implemented enhanced tensor builder:
  - `NextDayWildFireSpr/tools/build_enhanced_pickles.py`
  - Combines NDWS tile channels + normalized HEV scalar channels (broadcast to 64x64) into model-ready pickles.
  - Output dataset folder:
    - `NextDayWildFireSpr/data/next-day-wildfire-spread-hev`
    - Files: `train/validation/test .data + .labels`
    - Metadata: `channels_metadata.json`
- Refactored training script for dynamic channel support:
  - Updated `NextDayWildFireSpr/trainModel-II.py`
  - Added args:
    - `--dataset_path`
    - `--channels_metadata`
    - `--selected_features` (comma-separated)
  - Model input channels now inferred from selected feature list (not hardcoded to 12).

### Executed Commands (Block 3)
- `.venv/bin/python NextDayWildFireSpr/tools/build_enhanced_pickles.py --tfrecord_dir NextDayWildFireSpr/data/ndws64_meta_ca --hev_csv NextDayWildFireSpr/data/interim/sample_features_hev.csv --output_dir NextDayWildFireSpr/data/next-day-wildfire-spread-hev --metadata_json NextDayWildFireSpr/data/next-day-wildfire-spread-hev/channels_metadata.json --tile_size 64`
- `.venv/bin/python -m py_compile NextDayWildFireSpr/trainModel-II.py`
- Loader smoke test executed by importing `trainModel-II.py` and calling `create_data_loaders(...)` with enhanced dataset path + metadata.

### Validation Results (Block 3)
- Enhanced pickled dataset shapes:
  - `train.data=(2334, 21, 64, 64)`, `train.labels=(2334, 64, 64)`
  - `validation.data=(158, 21, 64, 64)`, `validation.labels=(158, 64, 64)`
  - `test.data=(291, 21, 64, 64)`, `test.labels=(291, 64, 64)`
- Integrity checks from builder:
  - `missing_in_hev=0`
  - `parse_errors=0`
- Data loader smoke test:
  - `num_input_channels=21`
  - first train batch shape: `X=(64, 21, 64, 64)`, `y=(64, 1, 64, 64)`

### Tooling Fixes (During Block 3)
- Patched `NextDayWildFireSpr/tools/sanity_check_readiness.py` to resolve external dataset paths from `Ext_Datasets/` (with backward-compatible fallback to repo root layout).
- Re-ran readiness check after patch:
  - `PASS=10`, `WARN=1`, `FAIL=0`
  - Remaining warning is expected for legacy archive schema check (archive TFRecords are base NDWS format without metadata fields).

### Remaining Blocks (Current)
1. Run enhanced smoke training (1-2 epochs) and baseline comparison on aligned split.
2. Build risk fusion outputs (`hazard`, `exposure`, `vulnerability`, `composite`) and tract-level rankings.
3. Export map artifacts for California and wire a minimal frontend for visualization.
