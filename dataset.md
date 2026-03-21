# Dataset Inventory (Implemented)

## A. Core Model Training Data (Used in Current Hazard Model)

### 1) Mapped NDWS CA TFRecords
- Path: `NextDayWildFireSpr/data/ndws64_meta_ca`
- Source pipeline: Earth Engine export with sample metadata mapping
- Files:
  - `train_ndws64_meta_*.tfrecord.gz`
  - `eval_ndws64_meta_*.tfrecord.gz`
  - `test_ndws64_meta_*.tfrecord.gz`
- Metadata fields per sample: `sample_lon`, `sample_lat`, `sample_date`, `start_day`, `split`

### 2) Sample Manifest
- Path: `NextDayWildFireSpr/data/ndws64_meta_ca/sample_manifest.csv`
- Role: stable sample-level mapping table
- Key columns: `sample_id`, `split`, `sample_date`, `sample_lon`, `sample_lat`, `source_file`, `record_index`

### 3) Canonical Hazard Pickled Dataset
- Path: `NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard`
- Built by: `NextDayWildFireSpr/tools/build_hazard_pickles.py`
- Files:
  - `train.data`, `train.labels`
  - `validation.data`, `validation.labels`
  - `test.data`, `test.labels`
  - `channels_metadata.json`
  - `sample_index.csv`
- Channel composition:
  - Original NDWS bands (12)
  - Location/time metadata channels (3): `meta_lon_z`, `meta_lat_z`, `meta_day_of_year_z`
- Labels:
  - `FireMask` binarized to 0/1

## B. External Datasets (Ext_Datasets)

These are used for risk mapping and reporting layers, not for canonical hazard-model training inputs.

### 1) ACS Exposure Table
- Path: `Ext_Datasets/acs_2020_exposure.json`
- Fields used: `B01003_001E`, `B25001_001E`, `B25077_001E`, tract identifiers

### 2) CDC SVI (California Tracts)
- Path: `Ext_Datasets/SVI_2020_CaliforniaTract.csv`
- Fields used: `RPL_THEMES`, `RPL_THEME1`, `RPL_THEME2`, `RPL_THEME3`, `RPL_THEME4`, tract IDs

### 3) California Census Tracts (TIGER/Line 2020)
- Path: `Ext_Datasets/TIGER2020_CaliforniaTractsShapefile/*`
- Geometry/ID base for tract joins and aggregations

### 4) California Roads Infrastructure Shapefile
- Path: `Ext_Datasets/CaliforniaRoads_InfraShapefile-CRS_-_Functional_Classification/*`
- Used for proximity/density-style exposure context

### 5) California Historic Fire Perimeters
- Path: `Ext_Datasets/California_Historic_Fire_Perimeters_-6273763535668926275/*`
- Used for historical hazard context and map overlays

### 6) CAL FIRE Official Fire Geodatabase
- Path: `Ext_Datasets/CAL-FIRE_OFFICIAL_FIREPARAMS/fire24_1.gdb`
- Supplementary official perimeter/event reference

## C. Derived Interim Geospatial Layers

### EPSG:3310 Standardized Layers
- Path: `NextDayWildFireSpr/data/interim/geospatial_3310`
- Built by: `NextDayWildFireSpr/tools/preprocess_geospatial_layers.py`
- Outputs:
  - `tracts_3310.gpkg`
  - `roads_3310.gpkg`
  - `fire_perimeters_3310.gpkg`

## D. Deprecated / Non-Canonical Training Variant
- Path: `NextDayWildFireSpr/data/next-day-wildfire-spread-hev`
- Note: this was an intermediate HEV-channel training variant and is not the canonical hazard-only model dataset.
