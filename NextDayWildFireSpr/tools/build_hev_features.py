#!/usr/bin/env python3
"""Build per-sample Hazard/Exposure/Vulnerability feature table for CA samples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from spatial_standards import TARGET_CRS


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"
EXT_ROOT = REPO_ROOT / "Ext_Datasets"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"

DEFAULT_SAMPLE_TRACT = INTERIM_DIR / "sample_tract_join.csv"
DEFAULT_ROADS = INTERIM_DIR / "geospatial_3310" / "roads_3310.gpkg"
DEFAULT_FIRE = INTERIM_DIR / "geospatial_3310" / "fire_perimeters_3310.gpkg"
DEFAULT_ACS_JSON = EXT_ROOT / "acs_2020_exposure.json"
DEFAULT_SVI_CSV = EXT_ROOT / "SVI_2020_CaliforniaTract.csv"
DEFAULT_OUTPUT_CSV = INTERIM_DIR / "sample_features_hev.csv"
DEFAULT_SUMMARY_JSON = INTERIM_DIR / "sample_features_hev_summary.json"


def _normalize_geoid(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    text = re.sub(r"\.0$", "", text)
    if not text.isdigit():
        return pd.NA
    return text.zfill(11)


def _read_geolayer(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".gpkg":
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    return gpd.read_file(path)


def _read_acs(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("ACS JSON must be a list with header + rows")

    header = payload[0]
    rows = payload[1:]
    data = pd.DataFrame(rows, columns=header)

    for col in ["state", "county", "tract", "B01003_001E", "B25001_001E", "B25077_001E"]:
        if col not in data.columns:
            raise ValueError(f"ACS JSON missing column: {col}")

    data["GEOID"] = (
        data["state"].astype(str).str.zfill(2)
        + data["county"].astype(str).str.zfill(3)
        + data["tract"].astype(str).str.zfill(6)
    )
    data["acs_population"] = pd.to_numeric(data["B01003_001E"], errors="coerce")
    data["acs_housing_units"] = pd.to_numeric(data["B25001_001E"], errors="coerce")
    data["acs_median_home_value"] = pd.to_numeric(data["B25077_001E"], errors="coerce")
    return data[["GEOID", "acs_population", "acs_housing_units", "acs_median_home_value"]]


def _read_svi(path: Path) -> pd.DataFrame:
    svi = pd.read_csv(path)
    if "FIPS" not in svi.columns:
        raise ValueError("SVI CSV missing FIPS column")

    base_cols = [
        c
        for c in ["FIPS", "RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
        if c in svi.columns
    ]
    svi = svi[base_cols].copy()
    svi["GEOID"] = svi["FIPS"].map(_normalize_geoid)

    keep = [
        c
        for c in [
            "GEOID",
            "RPL_THEMES",
            "RPL_THEME1",
            "RPL_THEME2",
            "RPL_THEME3",
            "RPL_THEME4",
        ]
        if c in svi.columns
    ]
    out = svi[keep].copy()
    for c in keep:
        if c != "GEOID":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    rename = {
        "RPL_THEMES": "svi_rpl_themes",
        "RPL_THEME1": "svi_rpl_theme1",
        "RPL_THEME2": "svi_rpl_theme2",
        "RPL_THEME3": "svi_rpl_theme3",
        "RPL_THEME4": "svi_rpl_theme4",
    }
    out = out.rename(columns=rename)
    return out


def _compute_road_features(
    sample_df: pd.DataFrame,
    roads_path: Path,
) -> pd.DataFrame:
    roads = _read_geolayer(roads_path, layer="roads")
    if str(roads.crs) != TARGET_CRS:
        roads = roads.to_crs(TARGET_CRS)

    keep = [c for c in ["RouteID", "F_System", "geometry"] if c in roads.columns]
    roads = roads[keep].copy()

    points = gpd.GeoDataFrame(
        sample_df[["sample_id"]].copy(),
        geometry=gpd.points_from_xy(sample_df["sample_lon"], sample_df["sample_lat"]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    nearest = gpd.sjoin_nearest(
        points,
        roads,
        how="left",
        distance_col="road_nearest_dist_m",
    )
    nearest = nearest.sort_values(["sample_id", "road_nearest_dist_m"], kind="mergesort")
    nearest = nearest.drop_duplicates(subset=["sample_id"], keep="first")

    cols = [c for c in ["sample_id", "road_nearest_dist_m", "RouteID", "F_System"] if c in nearest.columns]
    out = nearest[cols].copy()
    out = out.rename(columns={"RouteID": "road_nearest_route_id", "F_System": "road_nearest_f_system"})
    return pd.DataFrame(out)


def _compute_fire_history_features(
    sample_df: pd.DataFrame,
    fire_path: Path,
    lookback_years: int,
    buffer_m: float,
) -> pd.DataFrame:
    fire = _read_geolayer(fire_path, layer="fire_perimeters")
    if str(fire.crs) != TARGET_CRS:
        fire = fire.to_crs(TARGET_CRS)

    if "ALARM_DATE" not in fire.columns:
        raise ValueError("Fire layer missing ALARM_DATE column")

    fire = fire[[c for c in ["ALARM_DATE", "GIS_ACRES", "geometry"] if c in fire.columns]].copy()
    fire["ALARM_DATE"] = pd.to_datetime(fire["ALARM_DATE"], errors="coerce")
    fire["GIS_ACRES"] = pd.to_numeric(fire.get("GIS_ACRES", 0.0), errors="coerce").fillna(0.0)
    fire = fire.dropna(subset=["ALARM_DATE", "geometry"]).copy()
    fire = fire[~fire.geometry.is_empty].copy()

    points = gpd.GeoDataFrame(
        sample_df[["sample_id", "sample_date"]].copy(),
        geometry=gpd.points_from_xy(sample_df["sample_lon"], sample_df["sample_lat"]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    points["sample_date"] = pd.to_datetime(points["sample_date"], errors="coerce")

    out = pd.DataFrame(
        {
            "sample_id": sample_df["sample_id"].values,
            f"past_fire_count_{lookback_years}y_{int(buffer_m/1000)}km": np.zeros(len(sample_df), dtype=np.int32),
            f"past_fire_acres_{lookback_years}y_{int(buffer_m/1000)}km": np.zeros(len(sample_df), dtype=np.float64),
            f"days_since_fire_min_{lookback_years}y_{int(buffer_m/1000)}km": np.full(len(sample_df), np.nan, dtype=np.float64),
        }
    )

    sample_id_to_idx = {sid: i for i, sid in enumerate(out["sample_id"].tolist())}

    count_col = f"past_fire_count_{lookback_years}y_{int(buffer_m/1000)}km"
    acres_col = f"past_fire_acres_{lookback_years}y_{int(buffer_m/1000)}km"
    days_col = f"days_since_fire_min_{lookback_years}y_{int(buffer_m/1000)}km"

    unique_dates = sorted(d for d in points["sample_date"].dropna().unique())
    for current_date in unique_dates:
        subset = points[points["sample_date"] == current_date].copy()
        if subset.empty:
            continue

        lower_date = pd.Timestamp(current_date) - pd.Timedelta(days=365 * lookback_years)
        fire_window = fire[(fire["ALARM_DATE"] <= current_date) & (fire["ALARM_DATE"] >= lower_date)].copy()
        if fire_window.empty:
            continue

        buffers = subset[["sample_id", "geometry"]].copy()
        buffers["geometry"] = buffers.geometry.buffer(buffer_m)
        buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=TARGET_CRS)

        joined = gpd.sjoin(
            buffers,
            fire_window[["ALARM_DATE", "GIS_ACRES", "geometry"]],
            how="left",
            predicate="intersects",
        )
        joined = joined[joined["ALARM_DATE"].notna()].copy()
        if joined.empty:
            continue

        joined["days_since_fire"] = (pd.Timestamp(current_date) - pd.to_datetime(joined["ALARM_DATE"])) / pd.Timedelta(days=1)
        grouped = joined.groupby("sample_id", as_index=False).agg(
            fire_count=("ALARM_DATE", "count"),
            fire_acres=("GIS_ACRES", "sum"),
            min_days_since_fire=("days_since_fire", "min"),
        )

        for row in grouped.itertuples(index=False):
            idx = sample_id_to_idx.get(row.sample_id)
            if idx is None:
                continue
            out.at[idx, count_col] = int(row.fire_count)
            out.at[idx, acres_col] = float(row.fire_acres)
            out.at[idx, days_col] = float(row.min_days_since_fire)

    return out


def build_features(
    sample_tract_csv: Path,
    roads_path: Path,
    fire_path: Path,
    acs_json: Path,
    svi_csv: Path,
    output_csv: Path,
    summary_json: Path,
    fire_lookback_years: int,
    fire_buffer_m: float,
) -> int:
    for p in [sample_tract_csv, roads_path, fire_path, acs_json, svi_csv]:
        if not p.exists():
            print(f"[ERROR] Required input not found: {p}")
            return 2

    sample = pd.read_csv(sample_tract_csv)
    required = {"sample_id", "sample_lon", "sample_lat", "sample_date", "split", "GEOID", "ALAND"}
    missing = sorted(required - set(sample.columns))
    if missing:
        print(f"[ERROR] sample_tract CSV missing columns: {missing}")
        return 2

    sample["sample_lon"] = pd.to_numeric(sample["sample_lon"], errors="coerce")
    sample["sample_lat"] = pd.to_numeric(sample["sample_lat"], errors="coerce")
    sample["ALAND"] = pd.to_numeric(sample["ALAND"], errors="coerce")
    sample["GEOID"] = sample["GEOID"].map(_normalize_geoid)
    sample["sample_date"] = pd.to_datetime(sample["sample_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    acs = _read_acs(acs_json)
    svi = _read_svi(svi_csv)

    merged = sample.merge(acs, on="GEOID", how="left")
    merged = merged.merge(svi, on="GEOID", how="left")

    merged["tract_area_km2"] = merged["ALAND"] / 1_000_000.0
    merged["exposure_pop_density_km2"] = np.where(
        merged["tract_area_km2"] > 0,
        merged["acs_population"] / merged["tract_area_km2"],
        np.nan,
    )
    merged["exposure_housing_density_km2"] = np.where(
        merged["tract_area_km2"] > 0,
        merged["acs_housing_units"] / merged["tract_area_km2"],
        np.nan,
    )

    road_features = _compute_road_features(merged, roads_path=roads_path)
    merged = merged.merge(road_features, on="sample_id", how="left")

    fire_features = _compute_fire_history_features(
        merged,
        fire_path=fire_path,
        lookback_years=fire_lookback_years,
        buffer_m=fire_buffer_m,
    )
    merged = merged.merge(fire_features, on="sample_id", how="left")

    merged["vulnerability_index"] = merged.get("svi_rpl_themes")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    summary = {
        "input": {
            "sample_tract_csv": str(sample_tract_csv),
            "roads": str(roads_path),
            "fire": str(fire_path),
            "acs_json": str(acs_json),
            "svi_csv": str(svi_csv),
        },
        "output_csv": str(output_csv),
        "rows": int(len(merged)),
        "split_counts": merged["split"].value_counts(dropna=False).to_dict(),
        "null_rates": {
            "geoid_missing": float(merged["GEOID"].isna().mean()),
            "acs_population_missing": float(merged["acs_population"].isna().mean()),
            "svi_rpl_themes_missing": float(merged.get("svi_rpl_themes", pd.Series([], dtype=float)).isna().mean() if "svi_rpl_themes" in merged.columns else 1.0),
            "road_dist_missing": float(merged["road_nearest_dist_m"].isna().mean()),
        },
        "fire_history": {
            "lookback_years": fire_lookback_years,
            "buffer_m": fire_buffer_m,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote HEV feature table: {output_csv}")
    print(f"[DONE] Wrote summary: {summary_json}")
    print(f"[INFO] Rows: {len(merged)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_tract_csv", type=Path, default=DEFAULT_SAMPLE_TRACT)
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROADS)
    parser.add_argument("--fire", type=Path, default=DEFAULT_FIRE)
    parser.add_argument("--acs_json", type=Path, default=DEFAULT_ACS_JSON)
    parser.add_argument("--svi_csv", type=Path, default=DEFAULT_SVI_CSV)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--fire_lookback_years", type=int, default=5)
    parser.add_argument("--fire_buffer_m", type=float, default=10000.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return build_features(
        sample_tract_csv=args.sample_tract_csv,
        roads_path=args.roads,
        fire_path=args.fire,
        acs_json=args.acs_json,
        svi_csv=args.svi_csv,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        fire_lookback_years=args.fire_lookback_years,
        fire_buffer_m=args.fire_buffer_m,
    )


if __name__ == "__main__":
    raise SystemExit(main())
