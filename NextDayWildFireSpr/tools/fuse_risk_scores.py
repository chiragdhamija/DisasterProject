#!/usr/bin/env python3
"""Fuse hazard predictions with exposure/vulnerability features into risk scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"

DEFAULT_HAZARD_CSV = INTERIM_DIR / "hazard_predictions.csv"
DEFAULT_HEV_CSV = INTERIM_DIR / "sample_features_hev.csv"
DEFAULT_OUTPUT_SAMPLE_CSV = INTERIM_DIR / "sample_risk_scores.csv"
DEFAULT_OUTPUT_TRACT_CSV = INTERIM_DIR / "tract_risk_summary.csv"
DEFAULT_OUTPUT_DATE_CSV = INTERIM_DIR / "date_risk_summary.csv"
DEFAULT_SUMMARY_JSON = INTERIM_DIR / "risk_fusion_summary.json"


def _scale_minmax(series: pd.Series, log1p: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if log1p:
        s = np.log1p(np.clip(s, a_min=0.0, a_max=None))
    min_v = s.min(skipna=True)
    max_v = s.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v) or max_v <= min_v:
        return pd.Series(np.full(len(s), np.nan), index=s.index, dtype=float)
    return (s - min_v) / (max_v - min_v)


def _weighted_row_mean(df: pd.DataFrame, cols: list[str], weights: list[float]) -> pd.Series:
    arr = df[cols].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(arr)
    weighted = np.where(valid, arr * w, 0.0)
    w_eff = np.where(valid, w, 0.0)
    den = w_eff.sum(axis=1)
    num = weighted.sum(axis=1)
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
    return pd.Series(out, index=df.index, dtype=float)


def _nanquantile_or_nan(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def fuse_risk(
    hazard_csv: Path,
    hev_csv: Path,
    output_sample_csv: Path,
    output_tract_csv: Path,
    output_date_csv: Path,
    summary_json: Path,
    hazard_weight: float,
    exposure_weight: float,
    vulnerability_weight: float,
) -> int:
    if not hazard_csv.exists():
        print(f"[ERROR] hazard_csv not found: {hazard_csv}")
        return 2
    if not hev_csv.exists():
        print(f"[ERROR] HEV feature CSV not found: {hev_csv}")
        return 2

    hazard = pd.read_csv(hazard_csv, dtype={"sample_id": str})
    hev = pd.read_csv(hev_csv, dtype={"sample_id": str, "GEOID": str})

    if "sample_id" not in hazard.columns or "sample_id" not in hev.columns:
        print("[ERROR] Both input files must contain sample_id")
        return 2

    keep_hazard = [
        c
        for c in [
            "sample_id",
            "output_split",
            "sample_date",
            "sample_lon",
            "sample_lat",
            "hazard_prob_mean",
            "hazard_prob_p95",
            "hazard_prob_max",
            "hazard_pred_fire_frac",
        ]
        if c in hazard.columns
    ]
    hazard = hazard[keep_hazard].copy()

    if "GEOID" in hev.columns:
        hev["GEOID"] = (
            hev["GEOID"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(11)
        )

    merged = hev.merge(hazard, on="sample_id", how="inner", suffixes=("_hev", "_haz"))
    if merged.empty:
        print("[ERROR] No overlapping sample_id rows between HEV table and hazard predictions")
        return 2

    # Hazard component: keep in [0, 1].
    merged["hazard_index"] = np.clip(pd.to_numeric(merged["hazard_prob_mean"], errors="coerce"), 0.0, 1.0)

    # Exposure components (scaled to [0,1]).
    merged["exp_pop_density_idx"] = _scale_minmax(merged.get("exposure_pop_density_km2"), log1p=True)
    merged["exp_housing_density_idx"] = _scale_minmax(merged.get("exposure_housing_density_km2"), log1p=True)
    merged["exp_home_value_idx"] = _scale_minmax(merged.get("acs_median_home_value"), log1p=True)
    road_dist = pd.to_numeric(merged.get("road_nearest_dist_m"), errors="coerce")
    merged["exp_road_proximity_idx"] = _scale_minmax(1.0 / (1.0 + np.clip(road_dist, a_min=0.0, a_max=None)))

    exposure_cols = [
        "exp_pop_density_idx",
        "exp_housing_density_idx",
        "exp_home_value_idx",
        "exp_road_proximity_idx",
    ]
    exposure_weights = [0.35, 0.25, 0.25, 0.15]
    merged["exposure_index"] = _weighted_row_mean(merged, exposure_cols, exposure_weights)

    # Vulnerability component (SVI percentile preferred).
    if "svi_rpl_themes" in merged.columns:
        merged["vulnerability_index"] = pd.to_numeric(merged["svi_rpl_themes"], errors="coerce")
    else:
        vul_cols = [c for c in ["svi_rpl_theme1", "svi_rpl_theme2", "svi_rpl_theme3", "svi_rpl_theme4"] if c in merged.columns]
        if vul_cols:
            merged["vulnerability_index"] = merged[vul_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            merged["vulnerability_index"] = np.nan
    merged["vulnerability_index"] = np.clip(merged["vulnerability_index"], 0.0, 1.0)

    # Weighted risk fusion (kept as a secondary comparison metric).
    total_w = hazard_weight + exposure_weight + vulnerability_weight
    if total_w <= 0:
        print("[ERROR] Sum of weights must be positive")
        return 2
    hazard_weight /= total_w
    exposure_weight /= total_w
    vulnerability_weight /= total_w

    merged["risk_score_weighted"] = _weighted_row_mean(
        merged,
        ["hazard_index", "exposure_index", "vulnerability_index"],
        [hazard_weight, exposure_weight, vulnerability_weight],
    )
    merged["risk_score_weighted"] = merged["risk_score_weighted"].clip(0.0, 1.0)

    # Primary risk per slides: R = H x E x V (index form for mapping).
    vul_default = float(pd.to_numeric(merged["vulnerability_index"], errors="coerce").median(skipna=True))
    if not np.isfinite(vul_default):
        vul_default = 0.5
    merged["vulnerability_for_risk"] = pd.to_numeric(
        merged["vulnerability_index"], errors="coerce"
    ).fillna(vul_default)
    merged["risk_score"] = (
        merged["hazard_index"] * merged["exposure_index"] * merged["vulnerability_for_risk"]
    ).clip(0.0, 1.0)

    # Additional composite index aligned to multiplicative HEV framing.
    merged["risk_hev_product"] = merged["risk_score"]

    # Monetary and impact proxies (tract-level socioeconomic values broadcast to samples).
    merged["acs_population"] = pd.to_numeric(merged.get("acs_population"), errors="coerce")
    merged["acs_housing_units"] = pd.to_numeric(merged.get("acs_housing_units"), errors="coerce")
    merged["acs_median_home_value"] = pd.to_numeric(merged.get("acs_median_home_value"), errors="coerce")

    # Handle known ACS sentinel/invalid values (for example negative placeholders).
    merged.loc[merged["acs_population"] <= 0, "acs_population"] = np.nan
    merged.loc[merged["acs_housing_units"] <= 0, "acs_housing_units"] = np.nan
    merged.loc[merged["acs_median_home_value"] <= 0, "acs_median_home_value"] = np.nan

    merged["asset_value_usd"] = merged["acs_housing_units"] * merged["acs_median_home_value"]
    merged["expected_property_loss_usd_weighted"] = merged["risk_score_weighted"] * merged["asset_value_usd"]
    merged["expected_property_loss_usd_hev"] = merged["risk_score"] * merged["asset_value_usd"]
    merged["expected_population_affected"] = merged["risk_score"] * merged["acs_population"]
    merged["expected_housing_units_affected"] = merged["risk_score"] * merged["acs_housing_units"]

    # EAL-style explicit aliases.
    merged["hazard_annual_prob"] = merged["hazard_index"]
    merged["exposure_asset_value_usd"] = merged["asset_value_usd"]
    merged["vulnerability_damage_fraction"] = merged["vulnerability_for_risk"]
    merged["risk_eal_usd"] = merged["expected_property_loss_usd_hev"]

    # Risk tiers by quantiles.
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    try:
        merged["risk_tier"] = pd.qcut(
            merged["risk_score"],
            q=5,
            labels=labels,
            duplicates="drop",
        )
    except Exception:
        merged["risk_tier"] = "moderate"

    if "sample_date_haz" in merged.columns:
        merged["sample_date"] = merged["sample_date_haz"]
    elif "sample_date_hev" in merged.columns:
        merged["sample_date"] = merged["sample_date_hev"]
    if "sample_lon_haz" in merged.columns:
        merged["sample_lon"] = pd.to_numeric(merged["sample_lon_haz"], errors="coerce")
    elif "sample_lon_hev" in merged.columns:
        merged["sample_lon"] = pd.to_numeric(merged["sample_lon_hev"], errors="coerce")
    if "sample_lat_haz" in merged.columns:
        merged["sample_lat"] = pd.to_numeric(merged["sample_lat_haz"], errors="coerce")
    elif "sample_lat_hev" in merged.columns:
        merged["sample_lat"] = pd.to_numeric(merged["sample_lat_hev"], errors="coerce")

    sample_cols = [
        c
        for c in [
            "sample_id",
            "split",
            "output_split",
            "sample_date",
            "sample_lon",
            "sample_lat",
            "GEOID",
            "hazard_index",
            "exposure_index",
            "vulnerability_index",
            "risk_score",
            "risk_score_weighted",
            "risk_hev_product",
            "risk_tier",
            "hazard_prob_mean",
            "hazard_prob_p95",
            "hazard_prob_max",
            "hazard_pred_fire_frac",
            "acs_population",
            "acs_housing_units",
            "asset_value_usd",
            "risk_eal_usd",
            "expected_property_loss_usd_weighted",
            "expected_property_loss_usd_hev",
            "expected_population_affected",
            "expected_housing_units_affected",
            "exposure_pop_density_km2",
            "exposure_housing_density_km2",
            "acs_median_home_value",
            "road_nearest_dist_m",
            "svi_rpl_themes",
        ]
        if c in merged.columns
    ]
    sample_out = merged[sample_cols].copy()
    if "GEOID" in sample_out.columns:
        sample_out["GEOID"] = (
            sample_out["GEOID"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(11)
        )

    tract_out = (
        sample_out.groupby("GEOID", dropna=False)
        .agg(
            samples=("sample_id", "count"),
            risk_score_mean=("risk_score", "mean"),
            risk_score_weighted_mean=("risk_score_weighted", "mean"),
            risk_hev_product_mean=("risk_hev_product", "mean"),
            risk_score_p90=("risk_score", lambda s: _nanquantile_or_nan(s, 0.9)),
            hazard_index_mean=("hazard_index", "mean"),
            exposure_index_mean=("exposure_index", "mean"),
            vulnerability_index_mean=("vulnerability_index", "mean"),
            asset_value_usd_mean=("asset_value_usd", "mean"),
            risk_eal_usd_sum=("risk_eal_usd", "sum"),
            expected_property_loss_usd_weighted_sum=("expected_property_loss_usd_weighted", "sum"),
            expected_property_loss_usd_hev_sum=("expected_property_loss_usd_hev", "sum"),
            expected_population_affected_sum=("expected_population_affected", "sum"),
            expected_housing_units_affected_sum=("expected_housing_units_affected", "sum"),
        )
        .reset_index()
        .sort_values("risk_score_mean", ascending=False, kind="mergesort")
    )

    if "sample_date" in sample_out.columns:
        date_out = (
            sample_out.groupby("sample_date", dropna=False)
            .agg(
                samples=("sample_id", "count"),
                risk_score_mean=("risk_score", "mean"),
                risk_score_weighted_mean=("risk_score_weighted", "mean"),
                risk_hev_product_mean=("risk_hev_product", "mean"),
                hazard_index_mean=("hazard_index", "mean"),
                exposure_index_mean=("exposure_index", "mean"),
                vulnerability_index_mean=("vulnerability_index", "mean"),
                risk_eal_usd_sum=("risk_eal_usd", "sum"),
                expected_property_loss_usd_weighted_sum=("expected_property_loss_usd_weighted", "sum"),
                expected_property_loss_usd_hev_sum=("expected_property_loss_usd_hev", "sum"),
                expected_population_affected_sum=("expected_population_affected", "sum"),
                expected_housing_units_affected_sum=("expected_housing_units_affected", "sum"),
            )
            .reset_index()
            .sort_values("sample_date", kind="mergesort")
        )
    else:
        date_out = pd.DataFrame(
            columns=[
                "sample_date",
                "samples",
                "risk_score_mean",
                "risk_score_weighted_mean",
                "risk_hev_product_mean",
                "hazard_index_mean",
                "exposure_index_mean",
                "vulnerability_index_mean",
                "risk_eal_usd_sum",
                "expected_property_loss_usd_weighted_sum",
                "expected_property_loss_usd_hev_sum",
                "expected_population_affected_sum",
                "expected_housing_units_affected_sum",
            ]
        )

    for p in [output_sample_csv, output_tract_csv, output_date_csv, summary_json]:
        p.parent.mkdir(parents=True, exist_ok=True)
    sample_out.to_csv(output_sample_csv, index=False)
    tract_out.to_csv(output_tract_csv, index=False)
    date_out.to_csv(output_date_csv, index=False)

    summary = {
        "inputs": {
            "hazard_csv": str(hazard_csv),
            "hev_csv": str(hev_csv),
        },
        "formula": {
            "primary_risk": "risk_score = hazard_index * exposure_index * vulnerability_for_risk",
            "eal_usd": "risk_eal_usd = hazard_index * asset_value_usd * vulnerability_for_risk",
            "vulnerability_imputation": {
                "strategy": "median_fill_if_missing",
                "value": vul_default,
                "rows_imputed": int(pd.to_numeric(merged["vulnerability_index"], errors="coerce").isna().sum()),
            },
        },
        "fusion_weights_normalized": {
            "hazard_weight": hazard_weight,
            "exposure_weight": exposure_weight,
            "vulnerability_weight": vulnerability_weight,
        },
        "outputs": {
            "sample_risk_csv": str(output_sample_csv),
            "tract_risk_csv": str(output_tract_csv),
            "date_risk_csv": str(output_date_csv),
            "rows_sample": int(len(sample_out)),
            "rows_tract": int(len(tract_out)),
            "rows_date": int(len(date_out)),
        },
        "distributions": {
            "risk_score_mean": float(sample_out["risk_score"].mean()),
            "risk_score_weighted_mean": float(sample_out["risk_score_weighted"].mean()),
            "risk_score_p95": float(sample_out["risk_score"].quantile(0.95)),
            "risk_hev_product_mean": float(sample_out["risk_hev_product"].mean()),
            "risk_eal_usd_total": float(
                pd.to_numeric(sample_out["risk_eal_usd"], errors="coerce").sum(skipna=True)
            ),
            "expected_property_loss_usd_weighted_total": float(
                pd.to_numeric(sample_out["expected_property_loss_usd_weighted"], errors="coerce").sum(skipna=True)
            ),
            "expected_property_loss_usd_hev_total": float(
                pd.to_numeric(sample_out["expected_property_loss_usd_hev"], errors="coerce").sum(skipna=True)
            ),
            "expected_population_affected_total": float(
                pd.to_numeric(sample_out["expected_population_affected"], errors="coerce").sum(skipna=True)
            ),
            "expected_housing_units_affected_total": float(
                pd.to_numeric(sample_out["expected_housing_units_affected"], errors="coerce").sum(skipna=True)
            ),
            "risk_tier_counts": (
                sample_out["risk_tier"]
                .astype("string")
                .fillna("missing")
                .value_counts(dropna=False)
                .to_dict()
            ),
            "null_rates": {
                "hazard_index": float(sample_out["hazard_index"].isna().mean()),
                "exposure_index": float(sample_out["exposure_index"].isna().mean()),
                "vulnerability_index": float(sample_out["vulnerability_index"].isna().mean()),
            },
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote sample risk scores: {output_sample_csv}")
    print(f"[DONE] Wrote tract risk summary: {output_tract_csv}")
    print(f"[DONE] Wrote date risk summary: {output_date_csv}")
    print(f"[DONE] Wrote risk summary: {summary_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard_csv", type=Path, default=DEFAULT_HAZARD_CSV)
    parser.add_argument("--hev_csv", type=Path, default=DEFAULT_HEV_CSV)
    parser.add_argument("--output_sample_csv", type=Path, default=DEFAULT_OUTPUT_SAMPLE_CSV)
    parser.add_argument("--output_tract_csv", type=Path, default=DEFAULT_OUTPUT_TRACT_CSV)
    parser.add_argument("--output_date_csv", type=Path, default=DEFAULT_OUTPUT_DATE_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--hazard_weight", type=float, default=0.5)
    parser.add_argument("--exposure_weight", type=float, default=0.3)
    parser.add_argument("--vulnerability_weight", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return fuse_risk(
        hazard_csv=args.hazard_csv,
        hev_csv=args.hev_csv,
        output_sample_csv=args.output_sample_csv,
        output_tract_csv=args.output_tract_csv,
        output_date_csv=args.output_date_csv,
        summary_json=args.summary_json,
        hazard_weight=args.hazard_weight,
        exposure_weight=args.exposure_weight,
        vulnerability_weight=args.vulnerability_weight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
