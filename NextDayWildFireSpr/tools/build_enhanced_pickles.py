#!/usr/bin/env python3
"""Build enhanced pickled tensors by combining NDWS base tiles with HEV features."""

from __future__ import annotations

import argparse
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
DEFAULT_HEV_CSV = PROJECT_DIR / "data" / "interim" / "sample_features_hev.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "next-day-wildfire-spread-hev"
DEFAULT_METADATA_JSON = DEFAULT_OUTPUT_DIR / "channels_metadata.json"

BASE_CHANNELS = [
    # Keep base order aligned with current training script default.
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

DEFAULT_EXTRA_CHANNELS = [
    "exposure_pop_density_km2",
    "exposure_housing_density_km2",
    "acs_median_home_value",
    "svi_rpl_themes",
    "road_nearest_dist_m",
    "tract_nearest_dist_m",
    "past_fire_count_5y_10km",
    "past_fire_acres_5y_10km",
    "days_since_fire_min_5y_10km",
]

LABEL_KEY = "FireMask"
SPLIT_MAP = {"train": "train", "eval": "validation", "test": "test"}


def _compression_for(path: Path) -> str:
    return "GZIP" if path.suffix == ".gz" else ""


def _iter_split_files(tfrecord_dir: Path, split: str) -> list[Path]:
    files: list[Path] = []
    files.extend(sorted(tfrecord_dir.glob(f"{split}_*.tfrecord")))
    files.extend(sorted(tfrecord_dir.glob(f"{split}_*.tfrecord.gz")))
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
        raise ValueError(f"Feature {key} does not contain float_list/int64_list values")

    expected = tile_size * tile_size
    if values.size != expected:
        raise ValueError(f"Feature {key} expected {expected} values, got {values.size}")
    return values.reshape((tile_size, tile_size))


def _prepare_hev_table(
    hev_csv: Path,
    extra_channels: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    table = pd.read_csv(hev_csv)
    required = {"sample_id", "split"}
    missing_required = sorted(required - set(table.columns))
    if missing_required:
        raise ValueError(f"HEV table missing required columns: {missing_required}")

    missing_extra = sorted(set(extra_channels) - set(table.columns))
    if missing_extra:
        raise ValueError(f"HEV table missing requested extra channels: {missing_extra}")

    if table["sample_id"].duplicated().any():
        dup = int(table["sample_id"].duplicated().sum())
        raise ValueError(f"HEV table contains duplicate sample_id values: {dup}")

    stats: dict[str, dict[str, float]] = {}
    for col in extra_channels:
        table[col] = pd.to_numeric(table[col], errors="coerce")
        train_vals = table.loc[table["split"] == "train", col].dropna()
        mean = float(train_vals.mean()) if not train_vals.empty else 0.0
        std = float(train_vals.std(ddof=0)) if not train_vals.empty else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0

        z_col = f"z_{col}"
        table[z_col] = (table[col].fillna(mean) - mean) / std
        stats[col] = {
            "train_mean": mean,
            "train_std": std,
            "fill_value": mean,
        }

    table["output_split"] = table["split"].map(SPLIT_MAP)
    return table, stats


def build_pickles(
    tfrecord_dir: Path,
    hev_csv: Path,
    output_dir: Path,
    metadata_json: Path,
    tile_size: int,
    extra_channels: list[str],
) -> int:
    if tf is None:
        print("[ERROR] Missing dependency: tensorflow")
        return 2

    if not tfrecord_dir.exists():
        print(f"[ERROR] TFRecord directory not found: {tfrecord_dir}")
        return 2
    if not hev_csv.exists():
        print(f"[ERROR] HEV CSV not found: {hev_csv}")
        return 2

    table, norm_stats = _prepare_hev_table(hev_csv, extra_channels=extra_channels)
    table = table.set_index("sample_id", drop=False)

    channel_names = BASE_CHANNELS + [f"z_{c}" for c in extra_channels]
    num_channels = len(channel_names)

    split_counts = {
        split: int((table["output_split"] == split).sum())
        for split in ("train", "validation", "test")
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

    missing_in_hev = 0
    parse_errors = 0

    for split in ("train", "eval", "test"):
        files = _iter_split_files(tfrecord_dir, split)
        if not files:
            print(f"[WARN] No TFRecord files found for split={split}")
            continue

        for in_file in files:
            dataset = tf.data.TFRecordDataset(str(in_file), compression_type=_compression_for(in_file))
            for record_index, raw_record in enumerate(dataset):
                sample_id = f"{split}_{in_file.stem}_{record_index:08d}"

                if sample_id not in table.index:
                    missing_in_hev += 1
                    continue

                row = table.loc[sample_id]
                output_split = row["output_split"]
                if output_split not in ("train", "validation", "test"):
                    continue

                try:
                    example = tf.train.Example()
                    example.ParseFromString(bytes(raw_record.numpy()))

                    base_stack = [
                        _read_array_feature(example, key, tile_size)
                        for key in BASE_CHANNELS
                    ]
                    extra_stack = [
                        np.full(
                            (tile_size, tile_size),
                            float(row[f"z_{col}"]),
                            dtype=np.float32,
                        )
                        for col in extra_channels
                    ]
                    x = np.stack(base_stack + extra_stack, axis=0).astype(np.float32)
                    y = _read_array_feature(example, LABEL_KEY, tile_size).astype(np.float32)
                except Exception:
                    parse_errors += 1
                    continue

                i = write_idx[output_split]
                if i >= features[output_split].shape[0]:
                    # Extra safety in case counts from HEV table do not match data.
                    continue
                features[output_split][i] = x
                labels[output_split][i] = y
                write_idx[output_split] += 1

    # Trim to actual written rows.
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

    metadata = {
        "source": {
            "tfrecord_dir": str(tfrecord_dir),
            "hev_csv": str(hev_csv),
        },
        "tile_size": tile_size,
        "channel_names": channel_names,
        "num_channels": num_channels,
        "base_channels": BASE_CHANNELS,
        "extra_channels": extra_channels,
        "normalization": norm_stats,
        "split_shapes": {
            split: {
                "data": list(features[split].shape),
                "labels": list(labels[split].shape),
            }
            for split in ("train", "validation", "test")
        },
        "integrity": {
            "missing_in_hev": missing_in_hev,
            "parse_errors": parse_errors,
        },
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote enhanced dataset pickles to: {output_dir}")
    print(f"[DONE] Wrote channel metadata: {metadata_json}")
    for split in ("train", "validation", "test"):
        print(f"[INFO] {split}: data={features[split].shape}, labels={labels[split].shape}")
    print(f"[INFO] integrity: missing_in_hev={missing_in_hev}, parse_errors={parse_errors}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord_dir", type=Path, default=DEFAULT_TFRECORD_DIR)
    parser.add_argument("--hev_csv", type=Path, default=DEFAULT_HEV_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata_json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--tile_size", type=int, default=64)
    parser.add_argument(
        "--extra_channels",
        nargs="+",
        default=DEFAULT_EXTRA_CHANNELS,
        help="Scalar HEV columns to z-normalize and broadcast as extra channels.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return build_pickles(
        tfrecord_dir=args.tfrecord_dir,
        hev_csv=args.hev_csv,
        output_dir=args.output_dir,
        metadata_json=args.metadata_json,
        tile_size=args.tile_size,
        extra_channels=args.extra_channels,
    )


if __name__ == "__main__":
    raise SystemExit(main())
