#!/usr/bin/env python3
"""Serve frontend static files with local API endpoints for date-windowed wildfire views."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"
FRONTEND_DIR = PROJECT_DIR / "frontend"
FRONTEND_DATA_DIR = FRONTEND_DIR / "data"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _normalize_geoid(value: object) -> str | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    if txt.endswith(".0"):
        txt = txt[:-2]
    digits = "".join(ch for ch in txt if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(11)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    base = int(pos)
    rest = pos - base
    if base + 1 < len(sorted_values):
        return sorted_values[base] + rest * (sorted_values[base + 1] - sorted_values[base])
    return sorted_values[base]


def _compute_breaks(values: list[float]) -> list[float]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return [0.2, 0.4, 0.6, 0.8]
    return [_quantile(clean, q) for q in (0.2, 0.4, 0.6, 0.8)]


class DataStore:
    def __init__(
        self,
        frontend_data_dir: Path,
        sample_risk_csv: Path,
        horizon_default: int,
    ) -> None:
        self.frontend_data_dir = frontend_data_dir
        self.sample_risk_csv = sample_risk_csv
        self.horizon_default = max(0, horizon_default)

        self.points_by_date: dict[str, list[list[float]]] = {}
        self.centroids_by_date: dict[str, dict[str, float | str]] = {}
        self.trajectories: list[dict[str, object]] = []
        self.daily_by_date: dict[str, dict[str, object]] = {}
        self.dates: list[str] = []
        self.date_to_index: dict[str, int] = {}
        self.min_date: str | None = None
        self.max_date: str | None = None
        self.point_breaks: list[float] = [0.2, 0.4, 0.6, 0.8]

        self.tract_geometry_by_geoid: dict[str, dict[str, object]] = {}
        self.tract_features_by_date: dict[str, dict[str, object]] = {}
        self.tract_breaks_by_date: dict[str, list[float]] = {}

        self._load_points()
        self._load_trajectory()
        self._load_daily_summary()
        self._build_date_index()
        self._load_tract_geometries()
        self._build_daily_tract_risk()

    def _read_json(self, path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_points(self) -> None:
        payload = self._read_json(self.frontend_data_dir / "spread_daily_compact.json")
        by_date = payload.get("points_by_date", {})
        for date_key, rows in by_date.items():
            cleaned_rows: list[list[float]] = []
            for row in rows or []:
                if len(row) < 5:
                    continue
                lon = _safe_float(row[0], default=float("nan"))
                lat = _safe_float(row[1], default=float("nan"))
                if not math.isfinite(lon) or not math.isfinite(lat):
                    continue
                cleaned_rows.append(
                    [
                        lon,
                        lat,
                        _safe_float(row[2]),
                        _safe_float(row[3]),
                        _safe_float(row[4]),
                    ]
                )
            self.points_by_date[str(date_key)] = cleaned_rows

        risk_values = []
        for rows in self.points_by_date.values():
            for row in rows:
                risk_values.append(row[3])
        self.point_breaks = _compute_breaks(risk_values)

    def _load_trajectory(self) -> None:
        payload = self._read_json(self.frontend_data_dir / "spread_trajectory_compact.json")
        for row in payload.get("centroids", []):
            date_key = str(row.get("sample_date", "")).strip()
            if not date_key:
                continue
            lon = _safe_float(row.get("lon"), default=float("nan"))
            lat = _safe_float(row.get("lat"), default=float("nan"))
            if not math.isfinite(lon) or not math.isfinite(lat):
                continue
            self.centroids_by_date[date_key] = {"sample_date": date_key, "lon": lon, "lat": lat}

        # New multi-trajectory payload (cluster-linked trajectories).
        raw_trajs = payload.get("trajectories", [])
        for idx, traj in enumerate(raw_trajs):
            raw_points = traj.get("points", [])
            points = []
            for p in raw_points:
                date_key = str(p.get("sample_date", "")).strip()
                if not date_key:
                    continue
                lon = _safe_float(p.get("lon"), default=float("nan"))
                lat = _safe_float(p.get("lat"), default=float("nan"))
                if not math.isfinite(lon) or not math.isfinite(lat):
                    continue
                points.append(
                    {
                        "sample_date": date_key,
                        "lon": lon,
                        "lat": lat,
                        "weight_sum": _safe_float(p.get("weight_sum")),
                        "samples": int(_safe_float(p.get("samples"), default=0.0)),
                    }
                )
            if not points:
                continue
            tid_raw = traj.get("trajectory_id")
            try:
                tid = int(tid_raw)
            except (TypeError, ValueError):
                tid = idx + 1
            self.trajectories.append({"trajectory_id": tid, "points": points})

    def _load_daily_summary(self) -> None:
        rows = self._read_json(self.frontend_data_dir / "daily_risk_summary.json")
        for row in rows:
            date_key = str(row.get("sample_date", "")).strip()
            if not date_key:
                continue
            self.daily_by_date[date_key] = row

    def _build_date_index(self) -> None:
        traj_dates: set[str] = set()
        for traj in self.trajectories:
            for p in traj.get("points", []):
                date_val = str(p.get("sample_date", "")).strip()
                if date_val:
                    traj_dates.add(date_val)
        all_dates = (
            set(self.points_by_date.keys())
            | set(self.daily_by_date.keys())
            | set(self.centroids_by_date.keys())
            | traj_dates
        )
        self.dates = sorted(all_dates)
        self.date_to_index = {d: i for i, d in enumerate(self.dates)}
        if self.dates:
            self.min_date = self.dates[0]
            self.max_date = self.dates[-1]

    def _load_tract_geometries(self) -> None:
        geojson = self._read_json(self.frontend_data_dir / "tract_risk.geojson")
        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            geoid = _normalize_geoid(props.get("GEOID"))
            geom = feature.get("geometry")
            if geoid and geom:
                self.tract_geometry_by_geoid[geoid] = geom

    def _build_daily_tract_risk(self) -> None:
        date_geoid_acc: dict[str, dict[str, dict[str, float | int]]] = {}
        with self.sample_risk_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_key = str(row.get("sample_date", "")).strip()
                geoid = _normalize_geoid(row.get("GEOID"))
                if not date_key or not geoid:
                    continue
                bucket = date_geoid_acc.setdefault(date_key, {}).setdefault(
                    geoid,
                    {"samples": 0, "risk_sum": 0.0, "hazard_sum": 0.0, "eal_sum": 0.0},
                )
                bucket["samples"] = int(bucket["samples"]) + 1
                bucket["risk_sum"] = float(bucket["risk_sum"]) + _safe_float(row.get("risk_score"))
                bucket["hazard_sum"] = float(bucket["hazard_sum"]) + _safe_float(row.get("hazard_index"))
                bucket["eal_sum"] = float(bucket["eal_sum"]) + _safe_float(row.get("risk_eal_usd"))

        for date_key, geoid_rows in date_geoid_acc.items():
            features = []
            risk_values = []
            for geoid, acc in geoid_rows.items():
                geom = self.tract_geometry_by_geoid.get(geoid)
                if geom is None:
                    continue
                samples = int(acc["samples"])
                if samples <= 0:
                    continue
                risk_mean = float(acc["risk_sum"]) / samples
                hazard_mean = float(acc["hazard_sum"]) / samples
                eal_sum = float(acc["eal_sum"])
                risk_values.append(risk_mean)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "GEOID": geoid,
                            "samples": samples,
                            "risk_score_mean": risk_mean,
                            "hazard_index_mean": hazard_mean,
                            "risk_eal_usd_sum": eal_sum,
                        },
                    }
                )
            self.tract_features_by_date[date_key] = {"type": "FeatureCollection", "features": features}
            self.tract_breaks_by_date[date_key] = _compute_breaks(risk_values)

    def _resolve_date(self, raw_date: str | None) -> str:
        if not self.dates:
            return ""
        if raw_date and raw_date in self.date_to_index:
            return raw_date
        if raw_date:
            idx = bisect_left(self.dates, raw_date)
            if idx >= len(self.dates):
                return self.dates[-1]
            return self.dates[idx]
        return self.dates[0]

    def get_meta(self) -> dict[str, object]:
        if not self.dates:
            return {"dates": [], "horizon_default": self.horizon_default}
        return {
            "dates": self.dates,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "default_date": self.min_date,
            "horizon_default": self.horizon_default,
            "point_breaks": self.point_breaks,
            "california_bounds": [[32.35, -124.55], [42.1, -114.05]],
        }

    def get_window(self, raw_date: str | None, raw_horizon: str | None) -> dict[str, object]:
        if not self.dates:
            return {"window_dates": [], "points_by_date": {}, "centroids": [], "trajectories": [], "daily": []}

        date_key = self._resolve_date(raw_date)
        horizon = self.horizon_default
        if raw_horizon is not None:
            try:
                horizon = max(0, min(7, int(raw_horizon)))
            except ValueError:
                horizon = self.horizon_default

        # Option B behavior: window follows next available sampled dates.
        if date_key in self.date_to_index:
            idx = self.date_to_index[date_key]
        else:
            idx = 0
        window_dates = self.dates[idx : idx + horizon + 1]
        points_by_date = {d: self.points_by_date.get(d, []) for d in window_dates}
        centroids = [self.centroids_by_date[d] for d in window_dates if d in self.centroids_by_date]
        window_set = set(window_dates)
        trajectories = []
        for traj in self.trajectories:
            pts = [p for p in traj.get("points", []) if p.get("sample_date") in window_set]
            if not pts:
                continue
            trajectories.append(
                {
                    "trajectory_id": int(traj.get("trajectory_id", 0)),
                    "points": pts,
                }
            )
        daily_rows = [self.daily_by_date.get(d, {"sample_date": d}) for d in window_dates]

        return {
            "base_date": date_key,
            "window_dates": window_dates,
            "horizon": horizon,
            "points_by_date": points_by_date,
            "centroids": centroids,
            "trajectories": trajectories,
            "daily": daily_rows,
            "point_breaks": self.point_breaks,
        }

    def get_tract_risk(self, raw_date: str | None) -> dict[str, object]:
        if not self.dates:
            return {"sample_date": "", "risk_breaks": [0.2, 0.4, 0.6, 0.8], "feature_collection": {"type": "FeatureCollection", "features": []}}
        date_key = self._resolve_date(raw_date)
        return {
            "sample_date": date_key,
            "risk_breaks": self.tract_breaks_by_date.get(date_key, [0.2, 0.4, 0.6, 0.8]),
            "feature_collection": self.tract_features_by_date.get(date_key, {"type": "FeatureCollection", "features": []}),
        }


class APIHandler(SimpleHTTPRequestHandler):
    data_store: DataStore | None = None

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            super().do_GET()
            return

        if self.data_store is None:
            self._send_json({"error": "Data store unavailable"}, status=500)
            return

        q = parse_qs(parsed.query)
        if parsed.path == "/api/meta":
            self._send_json(self.data_store.get_meta())
            return
        if parsed.path == "/api/window":
            date_val = q.get("date", [None])[0]
            horizon_val = q.get("horizon", [None])[0]
            self._send_json(self.data_store.get_window(date_val, horizon_val))
            return
        if parsed.path == "/api/tract-risk":
            date_val = q.get("date", [None])[0]
            self._send_json(self.data_store.get_tract_risk(date_val))
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return

        self._send_json({"error": f"Unknown endpoint: {parsed.path}"}, status=404)


def build_handler(directory: Path, data_store: DataStore):
    class Handler(APIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    Handler.data_store = data_store
    return Handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--frontend_dir", type=Path, default=FRONTEND_DIR)
    p.add_argument("--frontend_data_dir", type=Path, default=FRONTEND_DATA_DIR)
    p.add_argument("--sample_risk_csv", type=Path, default=INTERIM_DIR / "sample_risk_scores.csv")
    p.add_argument("--horizon_default", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    missing = [str(p) for p in [args.frontend_dir, args.frontend_data_dir, args.sample_risk_csv] if not p.exists()]
    if missing:
        print(f"[ERROR] Missing required path(s): {missing}")
        return 2

    store = DataStore(
        frontend_data_dir=args.frontend_data_dir,
        sample_risk_csv=args.sample_risk_csv,
        horizon_default=args.horizon_default,
    )
    handler = build_handler(args.frontend_dir, store)

    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"[SERVE] http://{args.host}:{args.port}")
        print(
            "[SERVE] Endpoints: /api/meta, /api/window?date=YYYY-MM-DD&horizon=2 "
            "(next AVAILABLE dates), /api/tract-risk?date=YYYY-MM-DD"
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
