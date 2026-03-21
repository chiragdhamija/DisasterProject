#!/usr/bin/env python3
"""Build canonical hazard-model pickles from mapped CA TFRecords.

This creates model inputs using only:
- Original NDWS wildfire/environmental bands
- Location/time metadata channels derived from sample metadata

No external exposure/vulnerability features are used for model training.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception:
    tf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"

DEFAULT_TFRECORD_DIR = PROJECT_DIR / "data" / "ndws64_meta_ca"
DEFAULT_MANIFEST_CSV = PROJECT_DIR / "data" / "ndws64_meta_ca" / "sample_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "next-day-wildfire-spread-ca-hazard"
DEFAULT_METADATA_JSON = DEFAULT_OUTPUT_DIR / "channels_metadata.json"
DEFAULT_SAMPLE_INDEX_CSV = DEFAULT_OUTPUT_DIR / "sample_index.csv"

BASE_CHANNELS = [
    # Keep ordering aligned with current training code convention.
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
]

META_CHANNELS = [
    "meta_lon_z",
    "meta_lat_z",
    "meta_day_of_year_z",
]

LABEL_KEY = "FireMask"
SPLIT_TO_OUTPUT = {"train": "train", "eval": "validation", "test": "test"}


def _compression_for(path: Path) -> str:
    return "GZIP" if path.suffix == ".gz" else ""


def _iter_split_files(tfrecord_dir: Path, split: str) -> list[Path]:
    files = sorted(tfrecord_dir.glob(f"{split}_*.tfrecord"))
    files += sorted(tfrecord_dir.glob(f"{split}_*.tfrecord.gz"))
    return sorted({f.resolve(): f for f in files}.values(), key=lambda p: p.name)


def _read_array_feature(example: "tf.train.Example", key: str, tile_size: int) -> np.ndarray:
    feat = example.features.feature.get(key)
    if feat is None:
        raise KeyError(f"Missing key in TFRecord example: {key}")

    if feat.float_list.value:
        values = np.asarray(feat.float_list.value, dtype=np.float32)
    elif feat.int64_list.value:
        values = np.asarray(feat.int64_list.value, dtype=np.float32)
    else:
        raise ValueError(f"Feature {key} has no float/int values")

    expected = tile_size * tile_size
    if values.size != expected:
        raise ValueError(f"Feature {key} expected {expected} values, got {values.size}")
    return values.reshape((tile_size, tile_size))


def _to_binary_fire_mask(arr: np.ndarray) -> np.ndarray:
    """Converts masks to binary 0/1.

    Supports:
    - NDWS-style masks in [-1, 0, 1]
    - MODIS categorical masks where >=7 indicates fire.
    """
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    if finite.min() >= -1.0 and finite.max() <= 1.0:
        return (arr > 0).astype(np.float32)

    return (arr >= 7.0).astype(np.float32)


def _day_of_year(date_text: str) -> int:
    d = dt.datetime.strptime(date_text, "%Y-%m-%d").date()
    return d.timetuple().tm_yday


def _prepare_manifest(manifest_csv: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    manifest = pd.read_csv(manifest_csv)
    required = {
        "sample_id",
        "split",
        "sample_date",
        "sample_lon",
        "sample_lat",
        "source_file",
        "record_index",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if manifest["sample_id"].duplicated().any():
        dup = int(manifest["sample_id"].duplicated().sum())
        raise ValueError(f"Manifest has duplicate sample_id rows: {dup}")

    manifest["sample_lon"] = pd.to_numeric(manifest["sample_lon"], errors="coerce")
    manifest["sample_lat"] = pd.to_numeric(manifest["sample_lat"], errors="coerce")
    manifest["sample_date"] = pd.to_datetime(manifest["sample_date"], errors="coerce")
    manifest["meta_day_of_year"] = manifest["sample_date"].dt.dayofyear
    manifest["sample_date"] = manifest["sample_date"].dt.strftime("%Y-%m-%d")

    for col in ["sample_lon", "sample_lat", "meta_day_of_year"]:
        if manifest[col].isna().any():
            bad = int(manifest[col].isna().sum())
            raise ValueError(f"Manifest has missing values in {col}: {bad}")

    stats: dict[str, dict[str, float]] = {}
    train_mask = manifest["split"] == "train"
    for raw_col, z_col in [
        ("sample_lon", "meta_lon_z"),
        ("sample_lat", "meta_lat_z"),
        ("meta_day_of_year", "meta_day_of_year_z"),
    ]:
        vals = manifest.loc[train_mask, raw_col].astype(float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        manifest[z_col] = (manifest[raw_col].astype(float) - mean) / std
        stats[z_col] = {
            "source_column": raw_col,
            "train_mean": mean,
            "train_std": std,
        }

    manifest = manifest.set_index("sample_id", drop=False)
    return manifest, stats


def build_dataset(
    tfrecord_dir: Path,
    manifest_csv: Path,
    output_dir: Path,
    metadata_json: Path,
    sample_index_csv: Path,
    tile_size: int,
) -> int:
    if tf is None:
        print("[ERROR] Missing dependency: tensorflow")
        return 2

    if not tfrecord_dir.exists():
        print(f"[ERROR] TFRecord directory not found: {tfrecord_dir}")
        return 2
    if not manifest_csv.exists():
        print(f"[ERROR] Manifest CSV not found: {manifest_csv}")
        return 2

    manifest, meta_stats = _prepare_manifest(manifest_csv)

    channel_names = BASE_CHANNELS + META_CHANNELS
    num_channels = len(channel_names)

    split_counts = {
        out_split: int((manifest["split"] == in_split).sum())
        for in_split, out_split in SPLIT_TO_OUTPUT.items()
    }

    features = {
        split: np.zeros((split_counts[split], num_channels, tile_size, tile_size), dtype=np.float32)
        for split in ("train", "validation", "test")
    }
    labels = {
        split: np.zeros((split_counts[split], tile_size, tile_size), dtype=np.float32)
        for split in ("train", "validation", "test")
    }
    write_idx = {"train": 0, "validation": 0, "test": 0}

    sample_index_rows: list[dict[str, object]] = []
    missing_in_manifest = 0
    parse_errors = 0

    for in_split, out_split in SPLIT_TO_OUTPUT.items():
        files = _iter_split_files(tfrecord_dir, in_split)
        if not files:
            print(f"[WARN] No TFRecord files found for split={in_split}")
            continue

        for in_file in files:
            ds = tf.data.TFRecordDataset(str(in_file), compression_type=_compression_for(in_file))
            for record_index, raw_record in enumerate(ds):
                sample_id = f"{in_split}_{in_file.stem}_{record_index:08d}"
                if sample_id not in manifest.index:
                    missing_in_manifest += 1
                    continue

                row = manifest.loc[sample_id]
                try:
                    ex = tf.train.Example()
                    ex.ParseFromString(bytes(raw_record.numpy()))

                    base_stack: list[np.ndarray] = []
                    for key in BASE_CHANNELS:
                        arr = _read_array_feature(ex, key, tile_size)
                        if key == "PrevFireMask":
                            arr = _to_binary_fire_mask(arr)
                        base_stack.append(arr)

                    lon_channel = np.full((tile_size, tile_size), float(row["meta_lon_z"]), dtype=np.float32)
                    lat_channel = np.full((tile_size, tile_size), float(row["meta_lat_z"]), dtype=np.float32)
                    doy_channel = np.full((tile_size, tile_size), float(row["meta_day_of_year_z"]), dtype=np.float32)

                    x = np.stack(base_stack + [lon_channel, lat_channel, doy_channel], axis=0).astype(np.float32)
                    y = _to_binary_fire_mask(_read_array_feature(ex, LABEL_KEY, tile_size)).astype(np.float32)
                except Exception:
                    parse_errors += 1
                    continue

                i = write_idx[out_split]
                if i >= features[out_split].shape[0]:
                    continue
                features[out_split][i] = x
                labels[out_split][i] = y
                write_idx[out_split] += 1

                sample_index_rows.append(
                    {
                        "output_split": out_split,
                        "split": row["split"],
                        "array_index": i,
                        "sample_id": row["sample_id"],
                        "sample_date": row["sample_date"],
                        "sample_lon": float(row["sample_lon"]),
                        "sample_lat": float(row["sample_lat"]),
                        "meta_day_of_year": int(row["meta_day_of_year"]),
                        "source_file": row["source_file"],
                        "record_index": int(row["record_index"]),
                    }
                )

    # Trim by actual written row counts.
    for split in ("train", "validation", "test"):
        features[split] = features[split][: write_idx[split]]
        labels[split] = labels[split][: write_idx[split]]

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train.data").open("wb") as f:
        pickle.dump(features["train"], f)
    with (output_dir / "train.labels").open("wb") as f:
        pickle.dump(labels["train"], f)

    with (output_dir / "validation.data").open("wb") as f:
        pickle.dump(features["validation"], f)
    with (output_dir / "validation.labels").open("wb") as f:
        pickle.dump(labels["validation"], f)

    with (output_dir / "test.data").open("wb") as f:
        pickle.dump(features["test"], f)
    with (output_dir / "test.labels").open("wb") as f:
        pickle.dump(labels["test"], f)

    with sample_index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_split",
                "split",
                "array_index",
                "sample_id",
                "sample_date",
                "sample_lon",
                "sample_lat",
                "meta_day_of_year",
                "source_file",
                "record_index",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_index_rows)

    metadata = {
        "source": {
            "tfrecord_dir": str(tfrecord_dir),
            "manifest_csv": str(manifest_csv),
        },
        "tile_size": tile_size,
        "channel_names": channel_names,
        "num_channels": num_channels,
        "base_channels": BASE_CHANNELS,
        "meta_channels": META_CHANNELS,
        "meta_normalization": meta_stats,
        "split_shapes": {
            split: {
                "data": list(features[split].shape),
                "labels": list(labels[split].shape),
            }
            for split in ("train", "validation", "test")
        },
        "integrity": {
            "missing_in_manifest": missing_in_manifest,
            "parse_errors": parse_errors,
            "sample_index_rows": len(sample_index_rows),
        },
        "sample_index_csv": str(sample_index_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote hazard dataset pickles to: {output_dir}")
    print(f"[DONE] Wrote channels metadata: {metadata_json}")
    print(f"[DONE] Wrote sample index mapping: {sample_index_csv}")
    for split in ("train", "validation", "test"):
        print(f"[INFO] {split}: data={features[split].shape}, labels={labels[split].shape}")
    print(
        "[INFO] integrity:"
        f" missing_in_manifest={missing_in_manifest},"
        f" parse_errors={parse_errors},"
        f" sample_index_rows={len(sample_index_rows)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord_dir", type=Path, default=DEFAULT_TFRECORD_DIR)
    parser.add_argument("--manifest_csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata_json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--sample_index_csv", type=Path, default=DEFAULT_SAMPLE_INDEX_CSV)
    parser.add_argument("--tile_size", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return build_dataset(
        tfrecord_dir=args.tfrecord_dir,
        manifest_csv=args.manifest_csv,
        output_dir=args.output_dir,
        metadata_json=args.metadata_json,
        sample_index_csv=args.sample_index_csv,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
