**Complete Implementation Plan (From Now to Final Risk Mapping)**

1. Finalize scope and risk definition.  
Set target output as `Risk = Hazard × Exposure × Vulnerability` and lock spatial unit (`64x64 grid` + tract aggregation) and temporal unit (`next-day`).

2. Freeze baseline hazard benchmark.  
Keep current original NDWS pipeline unchanged as baseline reference (metrics, model weights, config) so all improvements are measured against it.

3. Validate and register mapped export as source-of-truth.  
Use your new `ndws64_meta_full` TFRecords as master input, validate schema and metadata fields, and record dataset version/date.

4. Build California-only modeling subset.  
Filter mapped samples to California bounds and keep split integrity (`train/eval/test` unchanged); save as new dataset artifact.

5. Create sample manifest for reproducibility.  
Generate per-sample table: `sample_id, split, sample_date, lon, lat, source_file, record_index`; this becomes the join key for all H/E/V pipelines.

6. Standardize all external geospatial layers to canonical CRS.  
Use `EPSG:3310` for roads, fire perimeters, tract geometries, and derived layers; persist cleaned/reprojected layers in `data/interim`.

7. Build exposure feature pipeline.  
Engineer exposure layers (population/housing density, road proximity/density, optional critical infrastructure proxies) aligned to sample location/date.

8. Build vulnerability feature pipeline.  
Join SVI + ACS to tracts, compute normalized vulnerability indicators/themes, and produce per-sample vulnerability features.

9. Build hazard enhancement feature pipeline.  
Add wildfire-history context (recency, burn frequency, nearby historic perimeter intensity/proximity) with strict temporal leakage control (`<= sample_date` only).

10. Convert engineered features to model-ready channels.  
Rasterize or window-extract engineered features to same `64x64` geometry as hazard samples; compute channel stats and normalization metadata.

11. Create enhanced training dataset artifact.  
Produce final tensors/records for `train/eval/test` with original hazard channels + engineered H/E/V channels, plus schema manifest.

12. Refactor training code for dynamic input channels.  
Update dataset loader/model init so channel count is config-driven (not fixed at 12), and add experiment config files.

13. Run model experiments.  
Run: baseline, enhanced-all, and ablations (`hazard-only`, `hazard+exposure`, `hazard+vulnerability`, `full HEV`); track IoU/F1/AUC and calibration.

14. Build risk fusion pipeline.  
Generate three maps:  
`H`: model fire probability,  
`E`: normalized exposure index,  
`V`: normalized vulnerability index,  
then compute composite `R` and classify risk tiers.

15. Validate risk outputs.  
Perform technical validation (schema/range/nulls), spatial sanity (hotspots plausible), temporal sanity, split leakage checks, and sensitivity analysis on risk weights/normalization.

16. Produce final deliverables.  
Deliver reproducible scripts, trained model, hazard map, exposure map, vulnerability map, composite risk map, tract/county rankings, and full report with methodology + limitations.

17. Optional deployment layer.  
Add a single command pipeline to run daily inference and export map artifacts for operations.

**Immediate next execution block**
1. Build California-only subset + manifest.  
2. Implement exposure/vulnerability feature engineering on that subset.  
3. Update training code to dynamic channels and run first enhanced smoke training.  

You do not need a third-party “enhanced dataset” now; your mapped export + your own external layers is sufficient for full H-E-V completion.