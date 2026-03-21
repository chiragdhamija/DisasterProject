#!/usr/bin/env python3
"""Reproject and clean external geospatial layers into canonical EPSG:3310."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from spatial_standards import TARGET_CRS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXT_ROOT = REPO_ROOT / "Ext_Datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "NextDayWildFireSpr" / "data" / "interim" / "geospatial_3310"


def _clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        # Fallback for environments without make_valid support.
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def _write_layer(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    layer_name: str,
) -> None:
    if output_path.exists():
        output_path.unlink()
    gdf.to_file(output_path, layer=layer_name, driver="GPKG", index=False)


def preprocess(
    ext_root: Path,
    output_dir: Path,
) -> int:
    tracts_path = ext_root / "TIGER2020_CaliforniaTractsShapefile" / "tl_2020_06_tract.shp"
    roads_path = (
        ext_root
        / "CaliforniaRoads_InfraShapefile-CRS_-_Functional_Classification"
        / "CRS_-_Functional_Classification.shp"
    )
    fire_path = (
        ext_root
        / "California_Historic_Fire_Perimeters_-6273763535668926275"
        / "California_Fire_Perimeters_(all).shp"
    )

    missing = [
        str(p)
        for p in (tracts_path, roads_path, fire_path)
        if not p.exists()
    ]
    if missing:
        print(f"[ERROR] Missing input layers: {missing}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tracts
    tracts = gpd.read_file(tracts_path)
    tracts = _clean_geometries(tracts)
    tracts = tracts[["GEOID", "NAMELSAD", "ALAND", "AWATER", "geometry"]].copy()
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    tracts = tracts.to_crs(TARGET_CRS)

    # Roads
    roads = gpd.read_file(roads_path)
    roads = _clean_geometries(roads)
    keep_road_cols = [c for c in ["OBJECTID", "RouteID", "F_System", "Shape_Leng", "geometry"] if c in roads.columns]
    roads = roads[keep_road_cols].copy()
    roads = roads.to_crs(TARGET_CRS)

    # Fire perimeters
    fire = gpd.read_file(fire_path)
    fire = _clean_geometries(fire)
    keep_fire_cols = [
        c
        for c in [
            "YEAR_",
            "STATE",
            "AGENCY",
            "UNIT_ID",
            "FIRE_NAME",
            "INC_NUM",
            "ALARM_DATE",
            "CONT_DATE",
            "CAUSE",
            "GIS_ACRES",
            "geometry",
        ]
        if c in fire.columns
    ]
    fire = fire[keep_fire_cols].copy()
    if "ALARM_DATE" in fire.columns:
        fire["ALARM_DATE"] = pd.to_datetime(fire["ALARM_DATE"], errors="coerce")
    if "CONT_DATE" in fire.columns:
        fire["CONT_DATE"] = pd.to_datetime(fire["CONT_DATE"], errors="coerce")
    if "GIS_ACRES" in fire.columns:
        fire["GIS_ACRES"] = pd.to_numeric(fire["GIS_ACRES"], errors="coerce")
    fire = fire.to_crs(TARGET_CRS)

    tracts_out = output_dir / "tracts_3310.gpkg"
    roads_out = output_dir / "roads_3310.gpkg"
    fire_out = output_dir / "fire_perimeters_3310.gpkg"

    _write_layer(tracts, tracts_out, "tracts")
    _write_layer(roads, roads_out, "roads")
    _write_layer(fire, fire_out, "fire_perimeters")

    summary = {
        "target_crs": TARGET_CRS,
        "input": {
            "tracts": str(tracts_path),
            "roads": str(roads_path),
            "fire_perimeters": str(fire_path),
        },
        "output": {
            "tracts": str(tracts_out),
            "roads": str(roads_out),
            "fire_perimeters": str(fire_out),
        },
        "counts": {
            "tracts": int(len(tracts)),
            "roads": int(len(roads)),
            "fire_perimeters": int(len(fire)),
        },
        "crs": {
            "tracts": str(tracts.crs),
            "roads": str(roads.crs),
            "fire_perimeters": str(fire.crs),
        },
    }
    summary_path = output_dir / "geospatial_preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Tracts saved: {tracts_out} ({len(tracts)} rows)")
    print(f"[DONE] Roads saved: {roads_out} ({len(roads)} rows)")
    print(f"[DONE] Fire perimeters saved: {fire_out} ({len(fire)} rows)")
    print(f"[DONE] Summary saved: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext_root", type=Path, default=DEFAULT_EXT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return preprocess(ext_root=args.ext_root, output_dir=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
