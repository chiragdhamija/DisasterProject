#!/usr/bin/env python3
"""Build a reproducible sample manifest from mapped NDWS TFRecords."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


try:
    import tensorflow as tf
except Exception:
    tf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"

DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "ndws64_meta_ca"
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_DIR / "sample_manifest.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_INPUT_DIR / "sample_manifest_summary.json"

_SPLIT_RE = re.compile(r"^(train|eval|test)_")


def _compression_for(path: Path) -> str:
    return "GZIP" if path.suffix == ".gz" else ""


def _split_from_filename(name: str) -> str:
    match = _SPLIT_RE.match(name)
    return match.group(1) if match else "unknown"


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
    unique = sorted({f.resolve(): f for f in files}.values(), key=lambda p: p.name)
    return unique


def build_manifest(
    input_dir: Path,
    output_csv: Path,
    summary_json: Path,
    fail_on_missing_metadata: bool,
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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "sample_id",
        "split",
        "sample_date",
        "start_day",
        "sample_lon",
        "sample_lat",
        "source_file",
        "record_index",
    ]

    split_counts: dict[str, int] = {"train": 0, "eval": 0, "test": 0, "unknown": 0}
    file_counts: dict[str, int] = {}
    missing_required = 0
    split_mismatches = 0
    sample_ids: set[str] = set()

    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        for in_file in files:
            split_from_name = _split_from_filename(in_file.name)
            compression = _compression_for(in_file)
            dataset = tf.data.TFRecordDataset(str(in_file), compression_type=compression)

            count_this_file = 0
            for record_index, raw_record in enumerate(dataset):
                example = tf.train.Example()
                example.ParseFromString(bytes(raw_record.numpy()))

                split_feature = _read_text_feature(example, "split")
                split = split_feature or split_from_name
                if split not in split_counts:
                    split = "unknown"

                if split_feature and split_from_name != "unknown" and split_feature != split_from_name:
                    split_mismatches += 1

                sample_date = _read_text_feature(example, "sample_date")
                start_day = _read_text_feature(example, "start_day")
                lon = _read_float_feature(example, "sample_lon")
                lat = _read_float_feature(example, "sample_lat")

                if sample_date is None or lon is None or lat is None:
                    missing_required += 1
                    continue

                sample_id = f"{split}_{in_file.stem}_{record_index:08d}"
                if sample_id in sample_ids:
                    print(f"[ERROR] Duplicate sample_id generated: {sample_id}")
                    return 1
                sample_ids.add(sample_id)

                writer.writerow(
                    [
                        sample_id,
                        split,
                        sample_date,
                        "" if start_day is None else start_day,
                        f"{lon:.6f}",
                        f"{lat:.6f}",
                        in_file.name,
                        record_index,
                    ]
                )

                split_counts[split] += 1
                count_this_file += 1

            file_counts[in_file.name] = count_this_file
            print(f"[FILE] {in_file.name} manifest_rows={count_this_file}")

    total_rows = sum(file_counts.values())
    summary = {
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "totals": {
            "rows": total_rows,
            "split_counts": split_counts,
            "files": len(file_counts),
            "missing_required_metadata": missing_required,
            "split_mismatches": split_mismatches,
        },
        "file_counts": file_counts,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[SUMMARY]")
    print(json.dumps(summary["totals"], indent=2))
    print(f"[DONE] Wrote manifest: {output_csv}")
    print(f"[DONE] Wrote summary: {summary_json}")

    if fail_on_missing_metadata and missing_required > 0:
        print("[ERROR] Missing required metadata found while --fail_on_missing_metadata is enabled.")
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--fail_on_missing_metadata",
        action="store_true",
        help="Fail if required metadata fields are missing in any record.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return build_manifest(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        fail_on_missing_metadata=args.fail_on_missing_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
