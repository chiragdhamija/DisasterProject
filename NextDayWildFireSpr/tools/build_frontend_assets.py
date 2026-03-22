#!/usr/bin/env python3
"""Build frontend-ready geospatial assets for spread, trajectory, and risk maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
FRONTEND_DIR = PROJECT_DIR / "frontend" / "data"

DEFAULT_SAMPLE_RISK = INTERIM_DIR / "sample_risk_scores.csv"
DEFAULT_DATE_SUMMARY = INTERIM_DIR / "date_risk_summary.csv"
DEFAULT_TRACT_RISK = INTERIM_DIR / "tract_risk_summary.csv"
DEFAULT_TRACTS_GPKG = INTERIM_DIR / "geospatial_3310" / "tracts_3310.gpkg"
DEFAULT_OUTPUT_DIR = FRONTEND_DIR
DEFAULT_SUMMARY_JSON = FRONTEND_DIR / "frontend_assets_summary.json"
CALIFORNIA_BBOX = [-124.48, 32.4, -114.13, 42.1]  # [min_lon, min_lat, max_lon, max_lat]


def _safe_float(value: object) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _to_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        num = float(value)
        return num if np.isfinite(num) else None
    if value is None:
        return None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: object, *, indent: int | None = None) -> None:
    safe = _to_json_safe(payload)
    path.write_text(json.dumps(safe, indent=indent, allow_nan=False), encoding="utf-8")


def _round_or_none(value: object, digits: int) -> float | None:
    num = _safe_float(value)
    if num is None:
        return None
    return round(num, digits)


def _to_feature(row: pd.Series) -> dict[str, object]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(row["sample_lon"]), float(row["sample_lat"])],
        },
        "properties": {
            "sample_id": row["sample_id"],
            "sample_date": row["sample_date"],
            "split": row.get("split"),
            "hazard_index": _safe_float(row.get("hazard_index")),
            "risk_score": _safe_float(row.get("risk_score")),
            "risk_score_weighted": _safe_float(row.get("risk_score_weighted")),
            "risk_tier": row.get("risk_tier"),
            "risk_eal_usd": _safe_float(row.get("risk_eal_usd")),
            "hazard_prob_mean": _safe_float(row.get("hazard_prob_mean")),
            "hazard_prob_max": _safe_float(row.get("hazard_prob_max")),
            "hazard_pred_fire_frac": _safe_float(row.get("hazard_pred_fire_frac")),
            "vulnerability_index": _safe_float(row.get("vulnerability_index")),
            "exposure_index": _safe_float(row.get("exposure_index")),
            "GEOID": str(row.get("GEOID")) if pd.notna(row.get("GEOID")) else None,
        },
    }


def _weighted_centroids(sample_df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    rows = []
    for date_val, group in sample_df.groupby("sample_date", sort=True):
        w = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        lon = pd.to_numeric(group["sample_lon"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(group["sample_lat"], errors="coerce").to_numpy(dtype=float)
        if np.sum(w) <= 0:
            lon_c = float(np.nanmean(lon))
            lat_c = float(np.nanmean(lat))
        else:
            lon_c = float(np.average(lon, weights=w))
            lat_c = float(np.average(lat, weights=w))
        rows.append({"sample_date": date_val, "centroid_lon": lon_c, "centroid_lat": lat_c})
    return pd.DataFrame(rows)


def build_assets(
    sample_risk_csv: Path,
    date_summary_csv: Path,
    tract_risk_csv: Path,
    tracts_gpkg: Path,
    output_dir: Path,
    summary_json: Path,
    trajectory_weight_col: str,
    keep_null_tracts: bool,
    simplify_tolerance: float,
    round_decimals: int,
) -> int:
    required_paths = [sample_risk_csv, date_summary_csv, tract_risk_csv, tracts_gpkg]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(f"[ERROR] Missing required input(s): {missing}")
        return 2

    sample = pd.read_csv(sample_risk_csv, dtype={"sample_id": str, "GEOID": str})
    date_summary = pd.read_csv(date_summary_csv)
    tract_risk = pd.read_csv(tract_risk_csv, dtype={"GEOID": str})

    needed_cols = {"sample_id", "sample_date", "sample_lon", "sample_lat", "hazard_index", "risk_score"}
    missing_cols = sorted(needed_cols - set(sample.columns))
    if missing_cols:
        print(f"[ERROR] sample_risk CSV missing columns: {missing_cols}")
        return 2

    if trajectory_weight_col not in sample.columns:
        print(f"[WARN] trajectory_weight_col '{trajectory_weight_col}' missing; falling back to risk_score")
        trajectory_weight_col = "risk_score"

    sample["sample_date"] = pd.to_datetime(sample["sample_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sample = sample.dropna(subset=["sample_date", "sample_lon", "sample_lat"]).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    # 1) Daily spread points (GeoJSON debug + compact by-date payload for frontend).
    sample = sample.sort_values(["sample_date", "sample_id"], kind="mergesort").reset_index(drop=True)
    points_fc = {"type": "FeatureCollection", "features": [_to_feature(r) for _, r in sample.iterrows()]}
    points_path = output_dir / "spread_points.geojson"
    _write_json(points_path, points_fc)

    compact_points_by_date: dict[str, list[list[object]]] = {}
    for date_val, group in sample.groupby("sample_date", sort=True):
        rows: list[list[object]] = []
        for r in group.itertuples(index=False):
            rows.append(
                [
                    _round_or_none(getattr(r, "sample_lon", None), round_decimals),
                    _round_or_none(getattr(r, "sample_lat", None), round_decimals),
                    _round_or_none(getattr(r, "hazard_index", None), round_decimals),
                    _round_or_none(getattr(r, "risk_score", None), round_decimals),
                    _round_or_none(getattr(r, "risk_eal_usd", None), 2),
                ]
            )
        compact_points_by_date[str(date_val)] = rows

    compact_points_payload = {
        "dates": sorted(compact_points_by_date.keys()),
        "schema": ["lon", "lat", "hazard_index", "risk_score", "risk_eal_usd"],
        "points_by_date": compact_points_by_date,
    }
    compact_points_path = output_dir / "spread_daily_compact.json"
    _write_json(compact_points_path, compact_points_payload)

    # 2) Trajectory from weighted centroids by day.
    centroids = _weighted_centroids(sample, weight_col=trajectory_weight_col)
    centroids = centroids.sort_values("sample_date", kind="mergesort").reset_index(drop=True)
    line_coords = centroids[["centroid_lon", "centroid_lat"]].to_numpy(dtype=float).tolist()
    traj_features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": line_coords},
            "properties": {"name": "daily_weighted_centroid_path", "weight_col": trajectory_weight_col},
        }
    ]
    for _, r in centroids.iterrows():
        traj_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r["centroid_lon"]), float(r["centroid_lat"])]},
                "properties": {"sample_date": r["sample_date"]},
            }
        )
    traj_fc = {"type": "FeatureCollection", "features": traj_features}
    traj_path = output_dir / "spread_trajectory.geojson"
    _write_json(traj_path, traj_fc)

    compact_centroids = []
    for _, r in centroids.iterrows():
        compact_centroids.append(
            {
                "sample_date": r["sample_date"],
                "lon": _round_or_none(r["centroid_lon"], round_decimals),
                "lat": _round_or_none(r["centroid_lat"], round_decimals),
            }
        )
    compact_traj_payload = {
        "weight_col": trajectory_weight_col,
        "centroids": compact_centroids,
        "line": [
            [
                _round_or_none(row["lon"], round_decimals),
                _round_or_none(row["lat"], round_decimals),
            ]
            for row in compact_centroids
        ],
    }
    compact_traj_path = output_dir / "spread_trajectory_compact.json"
    _write_json(compact_traj_path, compact_traj_payload)

    # 3) Daily summary json for timeline charts.
    if "sample_date" in date_summary.columns:
        date_summary["sample_date"] = pd.to_datetime(date_summary["sample_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily_rows = date_summary.sort_values("sample_date", kind="mergesort").to_dict(orient="records")
    daily_path = output_dir / "daily_risk_summary.json"
    _write_json(daily_path, daily_rows)

    # 4) Tract risk choropleth layer (EPSG:3310 -> EPSG:4326).
    tracts = gpd.read_file(tracts_gpkg, layer="tracts")
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    tract_risk["GEOID"] = tract_risk["GEOID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    merged = tracts.merge(tract_risk, on="GEOID", how="left")
    merged = merged.to_crs("EPSG:4326")
    if not keep_null_tracts and "risk_score_mean" in merged.columns:
        merged = merged[merged["risk_score_mean"].notna()].copy()

    if simplify_tolerance > 0:
        merged["geometry"] = merged.geometry.simplify(
            tolerance=simplify_tolerance,
            preserve_topology=True,
        )

    tract_cols = [
        "GEOID",
        "samples",
        "risk_score_mean",
        "risk_score_p90",
        "hazard_index_mean",
        "exposure_index_mean",
        "vulnerability_index_mean",
        "risk_eal_usd_sum",
        "expected_property_loss_usd_hev_sum",
        "expected_population_affected_sum",
        "expected_housing_units_affected_sum",
        "geometry",
    ]
    keep_cols = [c for c in tract_cols if c in merged.columns]
    merged = merged[keep_cols].copy()
    tract_geojson = json.loads(merged.to_json())
    tract_path = output_dir / "tract_risk.geojson"
    _write_json(tract_path, tract_geojson)

    summary = {
        "inputs": {
            "sample_risk_csv": str(sample_risk_csv),
            "date_summary_csv": str(date_summary_csv),
            "tract_risk_csv": str(tract_risk_csv),
            "tracts_gpkg": str(tracts_gpkg),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "spread_points_geojson": str(points_path),
            "spread_daily_compact_json": str(compact_points_path),
            "spread_trajectory_geojson": str(traj_path),
            "spread_trajectory_compact_json": str(compact_traj_path),
            "daily_risk_summary_json": str(daily_path),
            "tract_risk_geojson": str(tract_path),
        },
        "counts": {
            "spread_points": int(len(sample)),
            "trajectory_dates": int(len(centroids)),
            "daily_summary_rows": int(len(date_summary)),
            "tract_features": int(len(merged)),
        },
        "trajectory_weight_col": trajectory_weight_col,
        "keep_null_tracts": keep_null_tracts,
        "simplify_tolerance": simplify_tolerance,
        "round_decimals": round_decimals,
        "map_extent": {
            "default_bbox": CALIFORNIA_BBOX,
            "derived_sample_bbox": {
                "min_lon": _safe_float(sample["sample_lon"].min()),
                "min_lat": _safe_float(sample["sample_lat"].min()),
                "max_lon": _safe_float(sample["sample_lon"].max()),
                "max_lat": _safe_float(sample["sample_lat"].max()),
            },
        },
    }
    _write_json(summary_json, summary, indent=2)

    print(f"[DONE] spread points: {points_path}")
    print(f"[DONE] spread compact: {compact_points_path}")
    print(f"[DONE] trajectory: {traj_path}")
    print(f"[DONE] trajectory compact: {compact_traj_path}")
    print(f"[DONE] daily summary: {daily_path}")
    print(f"[DONE] tract risk layer: {tract_path}")
    print(f"[DONE] summary: {summary_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample_risk_csv", type=Path, default=DEFAULT_SAMPLE_RISK)
    p.add_argument("--date_summary_csv", type=Path, default=DEFAULT_DATE_SUMMARY)
    p.add_argument("--tract_risk_csv", type=Path, default=DEFAULT_TRACT_RISK)
    p.add_argument("--tracts_gpkg", type=Path, default=DEFAULT_TRACTS_GPKG)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    p.add_argument(
        "--trajectory_weight_col",
        default="risk_score",
        help="Column used to weight daily centroid trajectory",
    )
    p.add_argument(
        "--keep_null_tracts",
        action="store_true",
        help="Keep tracts without risk summary values in tract_risk.geojson",
    )
    p.add_argument(
        "--simplify_tolerance",
        type=float,
        default=0.001,
        help="Geometry simplify tolerance in EPSG:4326 degrees (0 disables simplify)",
    )
    p.add_argument(
        "--round_decimals",
        type=int,
        default=5,
        help="Decimal places for compact spread/trajectory coordinates and indices",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    return build_assets(
        sample_risk_csv=args.sample_risk_csv,
        date_summary_csv=args.date_summary_csv,
        tract_risk_csv=args.tract_risk_csv,
        tracts_gpkg=args.tracts_gpkg,
        output_dir=args.output_dir,
        summary_json=args.summary_json,
        trajectory_weight_col=args.trajectory_weight_col,
        keep_null_tracts=args.keep_null_tracts,
        simplify_tolerance=args.simplify_tolerance,
        round_decimals=args.round_decimals,
    )


if __name__ == "__main__":
    raise SystemExit(main())
