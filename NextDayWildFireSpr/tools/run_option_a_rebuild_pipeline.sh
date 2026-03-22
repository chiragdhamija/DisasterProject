#!/usr/bin/env bash
set -euo pipefail

# Rebuilds the full local pipeline after Option A TFRecords are downloaded:
# 1) Manifest
# 2) Hazard dataset pickles
# 3) Train hazard model
# 4) Geospatial + HEV features
# 5) Hazard inference
# 6) Risk fusion
# 7) Frontend assets
#
# Usage (from repo root):
#   bash NextDayWildFireSpr/tools/run_option_a_rebuild_pipeline.sh
#
# Optional env overrides:
#   RUN_TRAIN=1
#   EPOCHS=10
#   RAW_DIR=NextDayWildFireSpr/data/ndws64_meta_ca
#   HAZARD_DIR=NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard
#   GPU_ID=0
#   TRAIN_BATCH_SIZE=64
#   INFER_BATCH_SIZE=64
#   REQUIRE_CUDA=1
#   STEP2_DATA_DTYPE=float16
#   STEP2_LABEL_DTYPE=uint8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
EPOCHS="${EPOCHS:-10}"
GPU_ID="${GPU_ID:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-64}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STEP2_DATA_DTYPE="${STEP2_DATA_DTYPE:-float16}"
STEP2_LABEL_DTYPE="${STEP2_LABEL_DTYPE:-uint8}"

if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi

RAW_DIR="${RAW_DIR:-NextDayWildFireSpr/data/ndws64_meta_ca}"
RAW_MANIFEST="${RAW_DIR}/sample_manifest.csv"
RAW_SUMMARY="${RAW_DIR}/sample_manifest_summary.json"

HAZARD_DIR="${HAZARD_DIR:-NextDayWildFireSpr/data/next-day-wildfire-spread-ca-hazard}"
CHANNELS_JSON="${HAZARD_DIR}/channels_metadata.json"
SAMPLE_INDEX="${HAZARD_DIR}/sample_index.csv"

INTERIM_DIR="${INTERIM_DIR:-NextDayWildFireSpr/data/interim}"
GEO_DIR="${INTERIM_DIR}/geospatial_3310"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

if ! compgen -G "${RAW_DIR}/train_*.tfrecord*" > /dev/null; then
  echo "[ERROR] Missing train TFRecords in ${RAW_DIR}"
  exit 1
fi

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  if ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import torch,sys
ok=torch.cuda.is_available() and torch.cuda.device_count()>0
print(f"[CUDA CHECK] available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} compiled_cuda={torch.version.cuda}")
sys.exit(0 if ok else 1)
PY
  then
    echo "[ERROR] CUDA not available in this runtime for GPU_ID=${GPU_ID}."
    echo "[HINT] Fix NVIDIA driver/CUDA first or run with REQUIRE_CUDA=0 for CPU fallback."
    exit 2
  fi
fi

echo "[STEP 1/9] Build sample manifest"
env TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" NextDayWildFireSpr/tools/build_sample_manifest.py \
  --input_dir "$RAW_DIR" \
  --output_csv "$RAW_MANIFEST" \
  --summary_json "$RAW_SUMMARY" \
  --fail_on_missing_metadata

echo "[STEP 2/9] Build canonical hazard pickles"
echo "[INFO] Step 2 is CPU-bound preprocessing (TFRecord decode + pickle write); GPU does not speed this stage."
env TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u NextDayWildFireSpr/tools/build_hazard_pickles.py \
  --tfrecord_dir "$RAW_DIR" \
  --manifest_csv "$RAW_MANIFEST" \
  --output_dir "$HAZARD_DIR" \
  --metadata_json "$CHANNELS_JSON" \
  --sample_index_csv "$SAMPLE_INDEX" \
  --tile_size 64 \
  --data_dtype "$STEP2_DATA_DTYPE" \
  --label_dtype "$STEP2_LABEL_DTYPE"

echo "[STEP 3/9] Coverage quick check"
export COVERAGE_SAMPLE_INDEX="$SAMPLE_INDEX"
"$PYTHON_BIN" - <<'PY'
import csv
import os
from datetime import date, timedelta
path = os.environ["COVERAGE_SAMPLE_INDEX"]
rows = list(csv.DictReader(open(path)))
dates = sorted({r["sample_date"] for r in rows if r.get("sample_date")})
if not dates:
    raise SystemExit("[ERROR] No sample dates found in sample_index.csv")
start = date(2020, 1, 1)
end = date(2020, 12, 31)
all_days = []
d = start
while d <= end:
    all_days.append(d.isoformat())
    d += timedelta(days=1)
missing = [d for d in all_days if d not in set(dates)]
print(f"[COVERAGE] rows={len(rows)} unique_dates={len(dates)} min={dates[0]} max={dates[-1]} missing_days_2020={len(missing)}")
PY

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[STEP 4/9] Train hazard model (GPU ${GPU_ID})"
  (
    cd NextDayWildFireSpr
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" trainModel-II.py \
      --epochs "$EPOCHS" \
      --batch_size "$TRAIN_BATCH_SIZE" \
      --dataset_path "data/next-day-wildfire-spread-ca-hazard" \
      --channels_metadata "data/next-day-wildfire-spread-ca-hazard/channels_metadata.json" \
      --learning_rate 7e-4 \
      --weight_decay 1e-4 \
      --pos_weight 4 \
      --grad_clip 1.0 \
      --lr_factor 0.6 \
      --lr_patience 2 \
      --early_stop_patience 4 \
      --min_delta 0.0005 \
      2>&1 | tee "train_hazard_option_a_${EPOCHS}ep.log"
  )
else
  echo "[STEP 4/9] Skipping training (RUN_TRAIN=${RUN_TRAIN})"
fi

echo "[STEP 5/9] Preprocess external geospatial layers"
"$PYTHON_BIN" NextDayWildFireSpr/tools/preprocess_geospatial_layers.py \
  --ext_root Ext_Datasets \
  --output_dir "$GEO_DIR"

echo "[STEP 6/9] Attach samples to tracts"
"$PYTHON_BIN" NextDayWildFireSpr/tools/build_sample_tract_join.py \
  --manifest "$RAW_MANIFEST" \
  --tracts "${GEO_DIR}/tracts_3310.gpkg" \
  --output_csv "${INTERIM_DIR}/sample_tract_join.csv" \
  --summary_json "${INTERIM_DIR}/sample_tract_join_summary.json"

echo "[STEP 7/9] Build HEV feature table"
"$PYTHON_BIN" NextDayWildFireSpr/tools/build_hev_features.py \
  --sample_tract_csv "${INTERIM_DIR}/sample_tract_join.csv" \
  --roads "${GEO_DIR}/roads_3310.gpkg" \
  --fire "${GEO_DIR}/fire_perimeters_3310.gpkg" \
  --acs_json Ext_Datasets/acs_2020_exposure.json \
  --svi_csv Ext_Datasets/SVI_2020_CaliforniaTract.csv \
  --output_csv "${INTERIM_DIR}/sample_features_hev.csv" \
  --summary_json "${INTERIM_DIR}/sample_features_hev_summary.json"

echo "[STEP 8/9] Run hazard inference"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" NextDayWildFireSpr/tools/infer_hazard_scores.py \
  --dataset_dir "$HAZARD_DIR" \
  --channels_metadata "$CHANNELS_JSON" \
  --sample_index_csv "$SAMPLE_INDEX" \
  --weights NextDayWildFireSpr/savedModels/model-U_Net-bestF1Score-Rank-0.weights \
  --split all \
  --threshold 0.5 \
  --batch_size "$INFER_BATCH_SIZE" \
  --output_csv "${INTERIM_DIR}/hazard_predictions.csv" \
  --summary_json "${INTERIM_DIR}/hazard_predictions_summary.json"

echo "[STEP 9/9] Fuse risk + build frontend assets"
"$PYTHON_BIN" NextDayWildFireSpr/tools/fuse_risk_scores.py \
  --hazard_csv "${INTERIM_DIR}/hazard_predictions.csv" \
  --hev_csv "${INTERIM_DIR}/sample_features_hev.csv" \
  --output_sample_csv "${INTERIM_DIR}/sample_risk_scores.csv" \
  --output_tract_csv "${INTERIM_DIR}/tract_risk_summary.csv" \
  --output_date_csv "${INTERIM_DIR}/date_risk_summary.csv" \
  --summary_json "${INTERIM_DIR}/risk_fusion_summary.json"

"$PYTHON_BIN" NextDayWildFireSpr/tools/build_frontend_assets.py \
  --sample_risk_csv "${INTERIM_DIR}/sample_risk_scores.csv" \
  --date_summary_csv "${INTERIM_DIR}/date_risk_summary.csv" \
  --tract_risk_csv "${INTERIM_DIR}/tract_risk_summary.csv" \
  --tracts_gpkg "${GEO_DIR}/tracts_3310.gpkg" \
  --output_dir NextDayWildFireSpr/frontend/data \
  --summary_json NextDayWildFireSpr/frontend/data/frontend_assets_summary.json \
  --round_decimals 4 \
  --simplify_tolerance 0.003

echo "[DONE] Option A rebuild completed."
