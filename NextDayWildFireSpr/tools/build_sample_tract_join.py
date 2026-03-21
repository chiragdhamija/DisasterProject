#!/usr/bin/env python3
"""Attach each sample from manifest to a California census tract (GEOID)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from spatial_standards import TARGET_CRS


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"

DEFAULT_MANIFEST = PROJECT_DIR / "data" / "ndws64_meta_ca" / "sample_manifest.csv"
DEFAULT_TRACTS = PROJECT_DIR / "data" / "interim" / "geospatial_3310" / "tracts_3310.gpkg"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data" / "interim" / "sample_tract_join.csv"
DEFAULT_SUMMARY_JSON = PROJECT_DIR / "data" / "interim" / "sample_tract_join_summary.json"


def _read_tracts(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".gpkg":
        tracts = gpd.read_file(path, layer="tracts")
    else:
        tracts = gpd.read_file(path)

    if "GEOID" not in tracts.columns:
        raise ValueError("Tracts layer does not contain GEOID column")

    keep = [c for c in ["GEOID", "NAMELSAD", "ALAND", "AWATER", "geometry"] if c in tracts.columns]
    tracts = tracts[keep].copy()
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

    if tracts.crs is None:
        raise ValueError("Tracts layer CRS is missing")
    if str(tracts.crs) != TARGET_CRS:
        tracts = tracts.to_crs(TARGET_CRS)

    tracts = tracts[tracts.geometry.notna()].copy()
    tracts = tracts[~tracts.geometry.is_empty].copy()
    return tracts


def build_join(
    manifest_path: Path,
    tracts_path: Path,
    output_csv: Path,
    summary_json: Path,
    assign_nearest_fallback: bool,
) -> int:
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 2
    if not tracts_path.exists():
        print(f"[ERROR] Tracts layer not found: {tracts_path}")
        return 2

    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "split": str, "source_file": str})
    required_cols = {"sample_id", "sample_lon", "sample_lat", "sample_date", "split", "source_file", "record_index"}
    missing_cols = sorted(required_cols - set(manifest.columns))
    if missing_cols:
        print(f"[ERROR] Manifest missing columns: {missing_cols}")
        return 2

    manifest["sample_lon"] = pd.to_numeric(manifest["sample_lon"], errors="coerce")
    manifest["sample_lat"] = pd.to_numeric(manifest["sample_lat"], errors="coerce")
    manifest["sample_date"] = pd.to_datetime(manifest["sample_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    points = gpd.GeoDataFrame(
        manifest[["sample_id"]].copy(),
        geometry=gpd.points_from_xy(manifest["sample_lon"], manifest["sample_lat"]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    points["point_x_3310"] = points.geometry.x
    points["point_y_3310"] = points.geometry.y

    tracts = _read_tracts(tracts_path)

    joined = gpd.sjoin(
        points[["sample_id", "point_x_3310", "point_y_3310", "geometry"]],
        tracts,
        how="left",
        predicate="intersects",
    )

    joined = joined.sort_values(["sample_id", "index_right"], kind="mergesort")
    joined = joined.drop_duplicates(subset=["sample_id"], keep="first")
    joined["tract_match_method"] = "intersects"
    joined["tract_nearest_dist_m"] = 0.0

    # Fallback for points outside tract polygons (for example offshore grid cells):
    # attach nearest tract while preserving explicit match provenance.
    unmatched_ids = joined.loc[joined["GEOID"].isna(), "sample_id"].tolist()
    if assign_nearest_fallback and unmatched_ids:
        unmatched_points = points[points["sample_id"].isin(unmatched_ids)].copy()
        nearest = gpd.sjoin_nearest(
            unmatched_points[["sample_id", "point_x_3310", "point_y_3310", "geometry"]],
            tracts,
            how="left",
            distance_col="tract_nearest_dist_m",
        )
        nearest = nearest.sort_values(["sample_id", "tract_nearest_dist_m"], kind="mergesort")
        nearest = nearest.drop_duplicates(subset=["sample_id"], keep="first")
        nearest["tract_match_method"] = "nearest"

        keep_cols = [c for c in joined.columns if c in nearest.columns]
        nearest = nearest[keep_cols]
        joined = joined[~joined["sample_id"].isin(unmatched_ids)]
        joined = pd.concat([joined, nearest], ignore_index=True)
        joined = joined.sort_values(["sample_id"], kind="mergesort")
        joined = joined.drop_duplicates(subset=["sample_id"], keep="first")

    attach_cols = [
        c
        for c in [
            "sample_id",
            "GEOID",
            "NAMELSAD",
            "ALAND",
            "AWATER",
            "point_x_3310",
            "point_y_3310",
            "tract_match_method",
            "tract_nearest_dist_m",
        ]
        if c in joined.columns
    ]
    attached = joined[attach_cols].copy()

    result = manifest.merge(attached, on="sample_id", how="left")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    unmatched = int(result["GEOID"].isna().sum())
    summary = {
        "manifest": str(manifest_path),
        "tracts": str(tracts_path),
        "output_csv": str(output_csv),
        "rows_total": int(len(result)),
        "rows_matched": int(len(result) - unmatched),
        "rows_unmatched": unmatched,
        "match_rate": float((len(result) - unmatched) / len(result)) if len(result) else 0.0,
        "match_method_counts": result["tract_match_method"].value_counts(dropna=False).to_dict(),
        "split_counts": result["split"].value_counts(dropna=False).to_dict(),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote sample-tract join: {output_csv}")
    print(f"[DONE] Wrote summary: {summary_json}")
    print(f"[INFO] Match rate: {summary['match_rate']:.4f} ({summary['rows_matched']}/{summary['rows_total']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tracts", type=Path, default=DEFAULT_TRACTS)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--no_nearest_fallback",
        action="store_true",
        help="Disable nearest-tract assignment for unmatched points.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return build_join(
        manifest_path=args.manifest,
        tracts_path=args.tracts,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        assign_nearest_fallback=not args.no_nearest_fallback,
    )


if __name__ == "__main__":
    raise SystemExit(main())
