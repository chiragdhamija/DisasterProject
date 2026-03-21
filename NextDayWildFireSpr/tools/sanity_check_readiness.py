#!/usr/bin/env python3
"""Sanity checks for wildfire data readiness and required fixes."""

from __future__ import annotations

import csv
import json
import pickle
import re
import struct
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"
EXT_DATASETS_DIR = REPO_ROOT / "Ext_Datasets"


def _resolve_external_path(*parts: str) -> Path:
    """Resolves external dataset paths with Ext_Datasets-first fallback."""
    ext_path = EXT_DATASETS_DIR.joinpath(*parts)
    if ext_path.exists():
        return ext_path
    if ext_path.suffix == "":
        if any((Path(str(ext_path) + ext)).exists() for ext in (".shp", ".shx", ".dbf", ".prj")):
            return ext_path
    return REPO_ROOT.joinpath(*parts)


def _status_line(status: str, title: str, details: str) -> str:
    return f"[{status}] {title}: {details}"


def _read_dbf_schema(path: Path):
    with path.open("rb") as f:
        header = f.read(32)
        num_records = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        rec_len = struct.unpack("<H", header[10:12])[0]
        fields = []
        offset = 1
        while True:
            marker = f.read(1)
            if not marker or marker == b"\r":
                break
            descriptor = marker + f.read(31)
            name = descriptor[:11].split(b"\x00", 1)[0].decode("latin1").strip()
            flen = descriptor[16]
            fields.append((name, offset, offset + flen))
            offset += flen
    return num_records, header_len, rec_len, fields


def _iter_dbf_records(path: Path, field_names: list[str]):
    n, header_len, rec_len, fields = _read_dbf_schema(path)
    index = {name: (a, b) for name, a, b in fields}
    selected = [(name, *index[name]) for name in field_names]
    with path.open("rb") as f:
        f.seek(header_len)
        for _ in range(n):
            rec = f.read(rec_len)
            if len(rec) < rec_len:
                break
            if rec[:1] == b"*":
                continue
            yield {
                name: rec[a:b].decode("latin1", "ignore").strip()
                for name, a, b in selected
            }


def check_archive_tfrecords(lines: list[str]) -> bool:
    archive = PROJECT_DIR / "archive"
    files = sorted(archive.glob("next_day_wildfire_spread_*.tfrecord"))
    ok = True
    if len(files) != 19:
        lines.append(_status_line("FAIL", "Archive TFRecords", f"expected 19, found {len(files)}"))
        return False

    counts = {"train": 0, "eval": 0, "test": 0}
    pat = re.compile(r"next_day_wildfire_spread_(train|eval|test)_\d+\.tfrecord$")
    for p in files:
        m = pat.match(p.name)
        if not m:
            ok = False
            continue
        counts[m.group(1)] += 1

    expected = {"train": 15, "eval": 2, "test": 2}
    if counts != expected or not ok:
        lines.append(_status_line("FAIL", "Archive TFRecords", f"counts={counts}, expected={expected}"))
        return False

    lines.append(_status_line("PASS", "Archive TFRecords", f"counts={counts}"))
    return True


def check_pickled_dataset(lines: list[str]) -> bool:
    base = PROJECT_DIR / "data" / "next-day-wildfire-spread"
    required = [f"{split}.{ext}" for split in ("train", "validation", "test") for ext in ("data", "labels")]
    missing = [x for x in required if not (base / x).exists()]
    if missing:
        lines.append(_status_line("FAIL", "Pickled dataset", f"missing files: {missing}"))
        return False

    try:
        for split in ("train", "validation", "test"):
            with (base / f"{split}.data").open("rb") as f:
                x = pickle.load(f)
            with (base / f"{split}.labels").open("rb") as f:
                y = pickle.load(f)
            if len(x.shape) != 4 or x.shape[1] != 12:
                lines.append(_status_line("FAIL", "Pickled dataset", f"{split}.data shape={x.shape}"))
                return False
            if len(y.shape) != 3 or y.shape[0] != x.shape[0]:
                lines.append(_status_line("FAIL", "Pickled dataset", f"{split}.labels shape={y.shape}, data shape={x.shape}"))
                return False
    except Exception as exc:
        lines.append(_status_line("FAIL", "Pickled dataset", f"load error: {exc}"))
        return False

    lines.append(_status_line("PASS", "Pickled dataset", "train/validation/test tensors are readable and shaped correctly"))
    return True


def check_svi_and_tract_join(lines: list[str]) -> bool:
    svi_path = _resolve_external_path("SVI_2020_CaliforniaTract.csv")
    tract_dbf = _resolve_external_path(
        "TIGER2020_CaliforniaTractsShapefile", "tl_2020_06_tract.dbf"
    )
    if not svi_path.exists() or not tract_dbf.exists():
        lines.append(_status_line("FAIL", "SVI/Tract join", "missing SVI CSV or tract DBF"))
        return False

    svi_fips = set()
    with svi_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "FIPS" not in reader.fieldnames:
            lines.append(_status_line("FAIL", "SVI/Tract join", "FIPS column not found"))
            return False
        for row in reader:
            fips = row["FIPS"].strip()
            if fips:
                svi_fips.add(fips)

    bad_fips = [x for x in svi_fips if len(x) != 11 or not x.isdigit()]
    if bad_fips:
        lines.append(_status_line("FAIL", "SVI/Tract join", f"invalid SVI FIPS rows: {len(bad_fips)}"))
        return False

    tract_geoids = {rec["GEOID"] for rec in _iter_dbf_records(tract_dbf, ["GEOID"]) if rec["GEOID"]}
    svi_not_in_tract = svi_fips - tract_geoids
    tract_not_in_svi = tract_geoids - svi_fips
    if svi_not_in_tract:
        lines.append(_status_line("FAIL", "SVI/Tract join", f"SVI GEOIDs missing in tract layer: {len(svi_not_in_tract)}"))
        return False

    lines.append(
        _status_line(
            "PASS",
            "SVI/Tract join",
            f"SVI rows={len(svi_fips)}, tracts={len(tract_geoids)}, tract-only={len(tract_not_in_svi)} (expected special tracts)",
        )
    )
    return True


def check_acs_json(lines: list[str]) -> bool:
    acs_path = _resolve_external_path("acs_2020_exposure.json")
    if not acs_path.exists():
        lines.append(_status_line("FAIL", "ACS JSON", "acs_2020_exposure.json missing"))
        return False

    try:
        with acs_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        lines.append(_status_line("FAIL", "ACS JSON", f"read error: {exc}"))
        return False

    if not isinstance(data, list) or len(data) < 2:
        lines.append(_status_line("FAIL", "ACS JSON", "unexpected format; expected list with header + rows"))
        return False

    header = data[0]
    required = {"NAME", "B01003_001E", "B25001_001E", "B25077_001E", "state", "county", "tract"}
    if not isinstance(header, list) or not required.issubset(set(header)):
        lines.append(_status_line("FAIL", "ACS JSON", f"missing required columns: {sorted(required - set(header))}"))
        return False

    idx = {name: header.index(name) for name in required}
    bad_geoids = 0
    for row in data[1:]:
        if not isinstance(row, list) or len(row) < len(header):
            bad_geoids += 1
            continue
        geoid = f"{row[idx['state']]}{row[idx['county']]}{row[idx['tract']]}"
        if len(geoid) != 11 or not geoid.isdigit():
            bad_geoids += 1

    if bad_geoids > 0:
        lines.append(_status_line("FAIL", "ACS JSON", f"invalid rows={bad_geoids}"))
        return False

    lines.append(_status_line("PASS", "ACS JSON", f"rows={len(data)-1}, schema valid"))
    return True


def check_vector_layers(lines: list[str]) -> bool:
    layers = [
        (
            "Fire Perimeters",
            _resolve_external_path(
                "California_Historic_Fire_Perimeters_-6273763535668926275",
                "California_Fire_Perimeters_(all)",
            ),
        ),
        (
            "Roads",
            _resolve_external_path(
                "CaliforniaRoads_InfraShapefile-CRS_-_Functional_Classification",
                "CRS_-_Functional_Classification",
            ),
        ),
        (
            "Tracts",
            _resolve_external_path(
                "TIGER2020_CaliforniaTractsShapefile", "tl_2020_06_tract"
            ),
        ),
    ]
    ok = True
    for label, stem in layers:
        missing = [ext for ext in (".shp", ".shx", ".dbf", ".prj") if not (Path(str(stem) + ext)).exists()]
        if missing:
            lines.append(_status_line("FAIL", f"{label} layer", f"missing {missing}"))
            ok = False
        else:
            lines.append(_status_line("PASS", f"{label} layer", "all shapefile components present"))
    return ok


def check_crs_decision(lines: list[str]) -> bool:
    standards_file = PROJECT_DIR / "tools" / "spatial_standards.py"
    if not standards_file.exists():
        lines.append(_status_line("FAIL", "CRS standard", "spatial_standards.py not found"))
        return False

    text = standards_file.read_text(encoding="utf-8")
    if "TARGET_CRS_EPSG = 3310" not in text:
        lines.append(_status_line("FAIL", "CRS standard", "TARGET_CRS_EPSG is not set to 3310"))
        return False

    roads_prj = _resolve_external_path(
        "CaliforniaRoads_InfraShapefile-CRS_-_Functional_Classification",
        "CRS_-_Functional_Classification.prj",
    ).read_text(encoding="utf-8", errors="ignore")
    fire_prj = _resolve_external_path(
        "California_Historic_Fire_Perimeters_-6273763535668926275",
        "California_Fire_Perimeters_(all).prj",
    ).read_text(encoding="utf-8", errors="ignore")
    tract_prj = _resolve_external_path(
        "TIGER2020_CaliforniaTractsShapefile", "tl_2020_06_tract.prj"
    ).read_text(encoding="utf-8", errors="ignore")

    source_signals = []
    source_signals.append("roads:3857-like" if "Mercator_Auxiliary_Sphere" in roads_prj else "roads:other")
    source_signals.append("fire:3857-like" if "Mercator_Auxiliary_Sphere" in fire_prj else "fire:other")
    source_signals.append("tract:4269-like" if "North_American_1983" in tract_prj and "PROJCS" not in tract_prj else "tract:other")

    lines.append(_status_line("PASS", "CRS standard", f"target=EPSG:3310, source_crs={source_signals}"))
    return True


def check_mapping_export_patch(lines: list[str]) -> bool:
    patch_file = PROJECT_DIR / "tools" / "ee_export_with_mapping.py"
    if not patch_file.exists():
        lines.append(_status_line("FAIL", "Mapping exporter patch", "ee_export_with_mapping.py not found"))
        return False

    text = patch_file.read_text(encoding="utf-8")
    required_snippets = [
        "geometries=True",
        "sample_lon",
        "sample_lat",
        "sample_date",
        "start_day",
        "split",
    ]
    missing = [s for s in required_snippets if s not in text]
    if missing:
        lines.append(_status_line("FAIL", "Mapping exporter patch", f"missing snippets: {missing}"))
        return False

    lines.append(_status_line("PASS", "Mapping exporter patch", "metadata fields + geometry preservation are present"))
    return True


def check_runtime_dependencies(lines: list[str]) -> bool:
    # Informational warning only; user said remaining must-haves will be handled.
    try:
        import importlib

        mods = [
            "ee",
            "pandas",
            "geopandas",
            "fiona",
            "pyogrio",
            "shapely",
            "pyproj",
            "rasterio",
            "rtree",
        ]
        missing = []
        for m in mods:
            try:
                importlib.import_module(m)
            except Exception:
                missing.append(m)
        if missing:
            lines.append(_status_line("WARN", "Geo dependencies", f"missing in current interpreter: {missing}"))
        else:
            lines.append(_status_line("PASS", "Geo dependencies", "all required geospatial libs installed"))
        return True
    except Exception as exc:
        lines.append(_status_line("WARN", "Geo dependencies", f"dependency check skipped: {exc}"))
        return True


def check_tfrecord_feature_schema(lines: list[str]) -> bool:
    # Informational check: confirms current archive has no geo/time fields yet.
    try:
        import tensorflow as tf
    except Exception:
        lines.append(_status_line("WARN", "TFRecord schema", "tensorflow unavailable; skipped"))
        return True

    files = sorted((PROJECT_DIR / "archive").glob("next_day_wildfire_spread_train_*.tfrecord"))
    if not files:
        lines.append(_status_line("FAIL", "TFRecord schema", "no train TFRecords found"))
        return False

    raw_ds = tf.data.TFRecordDataset(str(files[0]))
    ex = next(iter(raw_ds))
    proto = tf.train.Example()
    proto.ParseFromString(ex.numpy())
    keys = sorted(proto.features.feature.keys())

    expected_base = {
        "elevation",
        "pdsi",
        "pr",
        "sph",
        "th",
        "tmmn",
        "tmmx",
        "vs",
        "erc",
        "population",
        "NDVI",
        "PrevFireMask",
        "FireMask",
    }
    missing = sorted(expected_base - set(keys))
    has_meta = any(k in keys for k in ("sample_lon", "sample_lat", "sample_date"))

    if missing:
        lines.append(_status_line("FAIL", "TFRecord schema", f"missing base keys: {missing}"))
        return False
    if has_meta:
        lines.append(_status_line("PASS", "TFRecord schema", "contains base + metadata fields"))
    else:
        lines.append(_status_line("WARN", "TFRecord schema", "contains base fields only (expected until re-export with mapping patch)"))
    return True


def main() -> int:
    lines: list[str] = []
    checks = [
        check_archive_tfrecords,
        check_pickled_dataset,
        check_svi_and_tract_join,
        check_acs_json,
        check_vector_layers,
        check_crs_decision,
        check_mapping_export_patch,
        check_runtime_dependencies,
        check_tfrecord_feature_schema,
    ]

    ok = True
    for check_fn in checks:
        ok = check_fn(lines) and ok

    print("Sanity Check Report")
    print("===================")
    for line in lines:
        print(line)

    fail_count = sum(1 for x in lines if x.startswith("[FAIL]"))
    warn_count = sum(1 for x in lines if x.startswith("[WARN]"))
    pass_count = sum(1 for x in lines if x.startswith("[PASS]"))
    print("-------------------")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")

    return 1 if not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
