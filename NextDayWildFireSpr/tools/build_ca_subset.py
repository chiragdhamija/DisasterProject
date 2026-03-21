#!/usr/bin/env python3
"""Build a California-only subset from mapped NDWS TFRecords.

This script filters TFRecord examples using sample lon/lat metadata added by
`ee_export_with_mapping.py`, while preserving split/file structure.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


try:
    import tensorflow as tf
except Exception:
    tf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"

DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "ndws64_meta_full"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "ndws64_meta_ca"

# California approx bounds in WGS84.
DEFAULT_BOUNDS = {
    "lon_min": -124.48,
    "lon_max": -114.13,
    "lat_min": 32.45,
    "lat_max": 42.05,
}

_SPLIT_RE = re.compile(r"^(train|eval|test)_")


@dataclass
class FileStats:
    file: str
    split: str
    total: int = 0
    kept: int = 0
    dropped_outside_ca: int = 0
    missing_coords: int = 0
    split_mismatches: int = 0


def _split_from_filename(name: str) -> str:
    match = _SPLIT_RE.match(name)
    return match.group(1) if match else "unknown"


def _compression_for(path: Path) -> str:
    return "GZIP" if path.suffix == ".gz" else ""


def _read_float_feature(example: "tf.train.Example", key: str) -> float | None:
    feat = example.features.feature.get(key)
    if feat is None:
        return None
    if feat.float_list.value:
        return float(feat.float_list.value[0])
    if feat.int64_list.value:
        return float(feat.int64_list.value[0])
    if feat.bytes_list.value:
        try:
            return float(feat.bytes_list.value[0].decode("utf-8"))
        except Exception:
            return None
    return None


def _read_text_feature(example: "tf.train.Example", key: str) -> str | None:
    feat = example.features.feature.get(key)
    if feat is None:
        return None
    if feat.bytes_list.value:
        try:
            return feat.bytes_list.value[0].decode("utf-8")
        except Exception:
            return None
    if feat.int64_list.value:
        return str(feat.int64_list.value[0])
    if feat.float_list.value:
        return str(feat.float_list.value[0])
    return None


def _iter_input_files(input_dir: Path) -> list[Path]:
    patterns = [
        "train_*.tfrecord",
        "train_*.tfrecord.gz",
        "eval_*.tfrecord",
        "eval_*.tfrecord.gz",
        "test_*.tfrecord",
        "test_*.tfrecord.gz",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(input_dir.glob(pat)))
    # Deduplicate while preserving sorted order.
    unique = sorted({f.resolve(): f for f in files}.values(), key=lambda p: p.name)
    return unique


def _is_in_bounds(
    lon: float,
    lat: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> bool:
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def build_subset(
    input_dir: Path,
    output_dir: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    strict_metadata: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    if tf is None:
        print("[ERROR] Missing dependency: tensorflow")
        return 2

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 2

    files = _iter_input_files(input_dir)
    if not files:
        print(f"[ERROR] No TFRecord files found in: {input_dir}")
        return 2

    if output_dir.exists() and not overwrite and not dry_run:
        print(f"[ERROR] Output directory already exists: {output_dir}")
        print("Use --overwrite to replace it.")
        return 2

    if output_dir.exists() and overwrite and not dry_run:
        for p in output_dir.glob("*"):
            if p.is_file():
                p.unlink()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[FileStats] = []
    missing_total = 0

    for in_file in files:
        split = _split_from_filename(in_file.name)
        stats = FileStats(file=in_file.name, split=split)

        compression = _compression_for(in_file)
        dataset = tf.data.TFRecordDataset(str(in_file), compression_type=compression)

        writer = None
        if not dry_run:
            out_file = output_dir / in_file.name
            options = tf.io.TFRecordOptions(compression_type=_compression_for(out_file))
            writer = tf.io.TFRecordWriter(str(out_file), options=options)

        for index, raw_record in enumerate(dataset):
            _ = index
            stats.total += 1
            serialized = bytes(raw_record.numpy())

            example = tf.train.Example()
            example.ParseFromString(serialized)

            lon = _read_float_feature(example, "sample_lon")
            lat = _read_float_feature(example, "sample_lat")
            if lon is None or lat is None:
                stats.missing_coords += 1
                continue

            split_from_feature = _read_text_feature(example, "split")
            if split_from_feature and split != "unknown" and split_from_feature != split:
                stats.split_mismatches += 1

            if _is_in_bounds(
                lon=lon,
                lat=lat,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            ):
                stats.kept += 1
                if writer is not None:
                    writer.write(serialized)
            else:
                stats.dropped_outside_ca += 1

        if writer is not None:
            writer.close()

        all_stats.append(stats)
        missing_total += stats.missing_coords
        print(
            "[FILE]"
            f" {in_file.name}"
            f" total={stats.total}"
            f" kept={stats.kept}"
            f" outside={stats.dropped_outside_ca}"
            f" missing_coords={stats.missing_coords}"
            f" split_mismatches={stats.split_mismatches}"
        )

    total = sum(s.total for s in all_stats)
    kept = sum(s.kept for s in all_stats)
    dropped = sum(s.dropped_outside_ca for s in all_stats)
    mismatches = sum(s.split_mismatches for s in all_stats)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "bounds": {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        },
        "totals": {
            "records_total": total,
            "records_kept": kept,
            "records_dropped_outside_ca": dropped,
            "records_missing_coords": missing_total,
            "split_mismatches": mismatches,
            "ca_keep_ratio": (kept / total) if total else 0.0,
        },
        "files": [asdict(s) for s in all_stats],
    }

    print("[SUMMARY]")
    print(json.dumps(summary["totals"], indent=2))

    if strict_metadata and missing_total > 0:
        print("[ERROR] Missing coordinates found while --strict_metadata is enabled.")
        return 1

    if not dry_run:
        summary_path = output_dir / "ca_subset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote filtered dataset to: {output_dir}")
        print(f"[DONE] Wrote summary: {summary_path}")
    else:
        print("[DONE] Dry run only; no files written.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lon_min", type=float, default=DEFAULT_BOUNDS["lon_min"])
    parser.add_argument("--lon_max", type=float, default=DEFAULT_BOUNDS["lon_max"])
    parser.add_argument("--lat_min", type=float, default=DEFAULT_BOUNDS["lat_min"])
    parser.add_argument("--lat_max", type=float, default=DEFAULT_BOUNDS["lat_max"])
    parser.add_argument(
        "--strict_metadata",
        action="store_true",
        help="Fail if any record is missing sample_lon/sample_lat.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output_dir.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Scan only; do not write output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return build_subset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        strict_metadata=args.strict_metadata,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
