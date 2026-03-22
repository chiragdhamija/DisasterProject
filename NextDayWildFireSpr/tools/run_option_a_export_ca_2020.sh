#!/usr/bin/env bash
set -euo pipefail

# Submits Earth Engine export tasks for full-year 2020 California coverage
# using the Option A exporter settings (CA-only, daily split window, no-fire day sampling).
#
# Required:
# - EE auth configured for project (earthengine authenticate + set_project)
# - TFRecord exports to Drive/GCS enabled
#
# Usage:
#   EE_PROJECT=your-project-id bash NextDayWildFireSpr/tools/run_option_a_export_ca_2020.sh
#
# Optional env overrides:
#   EXPORT_DEST=drive|gcs
#   GCS_BUCKET=<bucket-name>           # required only when EXPORT_DEST=gcs
#   EXPORT_FOLDER=wildfire_ndws_ca2020_full
#   PREFIX_BASE=ndws64_meta_ca2020_full
#   MONTHS=202008,202009               # optional month filter (YYYYMM, comma-separated)
#   DRY_RUN=1                          # 1 = no export tasks; print split plan only

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EE_PROJECT="${EE_PROJECT:-}"
EXPORT_DEST="${EXPORT_DEST:-drive}"
EXPORT_FOLDER="${EXPORT_FOLDER:-wildfire_ndws_ca2020_full}"
PREFIX_BASE="${PREFIX_BASE:-ndws64_meta_ca2020_full}"
SAMPLING_SCALE="${SAMPLING_SCALE:-1000}"
KERNEL_SIZE="${KERNEL_SIZE:-64}"
SPLIT_RATIO="${SPLIT_RATIO:-0.1}"
SPLIT_WINDOW_DAYS="${SPLIT_WINDOW_DAYS:-1}"
SAMPLING_RATIO="${SAMPLING_RATIO:-0}"
SAMPLING_LIMIT_PER_CALL="${SAMPLING_LIMIT_PER_CALL:-60}"
NO_FIRE_SAMPLES_PER_DAY="${NO_FIRE_SAMPLES_PER_DAY:-60}"
NUM_SAMPLES_PER_FILE="${NUM_SAMPLES_PER_FILE:-2000}"
SEED="${SEED:-123}"
DRY_RUN="${DRY_RUN:-0}"
MONTHS="${MONTHS:-}"

if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

if [[ -z "$EE_PROJECT" ]]; then
  echo "[ERROR] Set EE_PROJECT before running this script."
  echo "Example: EE_PROJECT=disaster-490916 bash $0"
  exit 1
fi

EXTRA_ARGS=()
if [[ "$EXPORT_DEST" == "gcs" ]]; then
  if [[ -z "${GCS_BUCKET:-}" ]]; then
    echo "[ERROR] EXPORT_DEST=gcs requires GCS_BUCKET"
    exit 1
  fi
  EXTRA_ARGS+=(--bucket "$GCS_BUCKET")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

WINDOWS=(
  "2020-01-01 2020-02-01"
  "2020-02-01 2020-03-01"
  "2020-03-01 2020-04-01"
  "2020-04-01 2020-05-01"
  "2020-05-01 2020-06-01"
  "2020-06-01 2020-07-01"
  "2020-07-01 2020-08-01"
  "2020-08-01 2020-09-01"
  "2020-09-01 2020-10-01"
  "2020-10-01 2020-11-01"
  "2020-11-01 2020-12-01"
  "2020-12-01 2021-01-01"
)

for window in "${WINDOWS[@]}"; do
  read -r START_DATE END_DATE <<<"$window"
  MONTH_TAG="${START_DATE:0:7}"
  MONTH_KEY="${MONTH_TAG//-/}"

  if [[ -n "$MONTHS" ]]; then
    case ",${MONTHS}," in
      *",${MONTH_KEY},"*) ;;
      *) continue ;;
    esac
  fi

  PREFIX="${PREFIX_BASE}_${MONTH_KEY}"

  echo "[RUN] month=${MONTH_TAG} prefix=${PREFIX}"
  "$PYTHON_BIN" NextDayWildFireSpr/tools/ee_export_with_mapping.py \
    --export_dest "$EXPORT_DEST" \
    "${EXTRA_ARGS[@]}" \
    --folder "$EXPORT_FOLDER" \
    --prefix "$PREFIX" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --region ca \
    --kernel_size "$KERNEL_SIZE" \
    --sampling_scale "$SAMPLING_SCALE" \
    --sampling_ratio "$SAMPLING_RATIO" \
    --sampling_limit_per_call "$SAMPLING_LIMIT_PER_CALL" \
    --no_fire_samples_per_day "$NO_FIRE_SAMPLES_PER_DAY" \
    --split_window_days "$SPLIT_WINDOW_DAYS" \
    --eval_split_ratio "$SPLIT_RATIO" \
    --num_samples_per_file "$NUM_SAMPLES_PER_FILE" \
    --seed "$SEED" \
    --ee_project "$EE_PROJECT"
done

echo "[DONE] Option A export submission complete."
echo "[NEXT] Download all generated TFRecord files into NextDayWildFireSpr/data/ndws64_meta_ca"
