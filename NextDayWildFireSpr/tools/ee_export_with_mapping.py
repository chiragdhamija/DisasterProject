#!/usr/bin/env python3
"""Export NDWS-style TFRecords while preserving sample geospatial/time mapping.

This script mirrors the public Google Research next-day wildfire export logic,
with one critical addition: each sampled tile stores metadata fields:
`sample_lon`, `sample_lat`, `sample_date`, `start_day`, and `split`.

Requirements:
1. Google Earth Engine Python API (`earthengine-api`)
2. Authenticated EE session (`earthengine authenticate --project <PROJECT_ID>`)
3. Earth Engine API enabled on that GCP project
4. Export target: Google Drive folder or GCS bucket
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Optional

try:
    import ee
except ImportError:
    ee = None


DATA_SOURCES = {
    "ELEVATION_SRTM": "USGS/SRTMGL1_003",
    "VEGETATION_VIIRS": "NOAA/VIIRS/001/VNP13A1",
    "DROUGHT_GRIDMET": "GRIDMET/DROUGHT",
    "WEATHER_GRIDMET": "IDAHO_EPSCOR/GRIDMET",
    "FIRE_MODIS": "MODIS/006/MOD14A1",
    "POPULATION": "CIESIN/GPWv411/GPW_Population_Density",
}

DATA_BANDS = {
    "ELEVATION_SRTM": ["elevation"],
    "VEGETATION_VIIRS": ["NDVI"],
    "DROUGHT_GRIDMET": ["pdsi"],
    "WEATHER_GRIDMET": ["pr", "sph", "th", "tmmn", "tmmx", "vs", "erc"],
    "FIRE_MODIS": ["FireMask"],
    "POPULATION": ["population_density"],
}

DATA_TIME_SAMPLING_DAYS = {
    "VEGETATION_VIIRS": 8,
    "DROUGHT_GRIDMET": 5,
    "WEATHER_GRIDMET": 2,
    "FIRE_MODIS": 1,
}

FEATURE_BANDS = [
    "elevation",
    "population",
    "pdsi",
    "NDVI",
    "pr",
    "sph",
    "th",
    "tmmn",
    "tmmx",
    "vs",
    "erc",
    "PrevFireMask",
]
LABEL_BANDS = ["FireMask"]
METADATA_FIELDS = ["sample_lon", "sample_lat", "sample_date", "start_day", "split"]

US_BBOX = [-124, 24, -73, 49]
RESAMPLING_SCALE = 20000


def get_sampling_geometry(region: str) -> ee.Geometry:
    """Returns export geometry for the requested region."""
    region = region.lower()
    if region == "us":
        return ee.Geometry.Rectangle(US_BBOX)
    if region == "ca":
        states = ee.FeatureCollection("TIGER/2018/States")
        return states.filter(ee.Filter.eq("STATEFP", "06")).geometry()
    raise ValueError(f"Unsupported region: {region}")


def get_image(key: str) -> ee.Image:
    return ee.Image(DATA_SOURCES[key]).select(DATA_BANDS[key])


def get_image_collection(key: str) -> ee.ImageCollection:
    return ee.ImageCollection(DATA_SOURCES[key]).select(DATA_BANDS[key])


def remove_mask(image: ee.Image) -> ee.Image:
    return image.updateMask(ee.Image(1))


def zero_image(bands: list[str]) -> ee.Image:
    """Returns a float zero-valued image with the given band names."""
    return ee.Image.constant([0] * len(bands)).rename(bands).toFloat()


def window_reduce(
    source_key: str,
    start: ee.Date,
    end: ee.Date,
    reducer: str = "median",
    apply_remove_mask: bool = False,
) -> ee.Image:
    """Reduces a date-filtered collection with a safe zero-image fallback."""
    bands = DATA_BANDS[source_key]
    fallback = zero_image(bands)
    ic = get_image_collection(source_key).filterDate(start, end)
    if apply_remove_mask:
        ic = ic.map(remove_mask)

    if reducer == "median":
        reduced = ee.Image(ee.Algorithms.If(ic.size().gt(0), ic.median(), fallback))
    elif reducer == "max":
        reduced = ee.Image(ee.Algorithms.If(ic.size().gt(0), ic.max(), fallback))
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")

    return reduced.rename(bands).toFloat().unmask(0)


def convert_features_to_arrays(image_list: list[ee.Image], kernel_size: int) -> ee.Image:
    """Converts a list of images into kernel_size x kernel_size neighborhood arrays."""
    feature_stack = ee.Image.cat(image_list).float()
    kernel_list = ee.List.repeat(1, kernel_size)
    kernel_lists = ee.List.repeat(kernel_list, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_lists)
    return feature_stack.neighborhoodToArray(kernel)


def split_days_into_train_eval_test(
    start_date: ee.Date,
    end_date: ee.Date,
    split_ratio: float,
    window_length_days: int = 1,
    seed: int = 123,
) -> dict[str, list[int]]:
    """Splits day indices into train/eval/test lists."""
    if window_length_days < 1:
        raise ValueError("window_length_days must be >= 1")
    num_days = int(ee.Date(end_date).difference(ee.Date(start_date), "day").getInfo())
    days = list(range(0, num_days, window_length_days))
    rng = random.Random(seed)
    rng.shuffle(days)

    num_eval = int(len(days) * split_ratio)
    train_days = days[:-2 * num_eval] if num_eval > 0 else days
    eval_days = days[-2 * num_eval:-num_eval] if num_eval > 0 else []
    test_days = days[-num_eval:] if num_eval > 0 else []
    return {
        "train": sorted(train_days),
        "eval": sorted(eval_days),
        "test": sorted(test_days),
    }


def get_detection_count(
    detection_image: ee.Image, geometry: ee.Geometry, sampling_scale: int
) -> Optional[int]:
    detection_stats = detection_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=sampling_scale,
        bestEffort=True,
        maxPixels=1e13,
    )
    try:
        value = detection_stats.get("detection").getInfo()
        if value is None:
            return 0
        return int(value)
    except ee.EEException:
        return None


def add_metadata(
    feature: ee.Feature,
    split_name: str,
    start_day: int,
    sample_date: ee.String,
) -> ee.Feature:
    coords = ee.List(feature.geometry().coordinates())
    return feature.set(
        {
            "sample_lon": coords.get(0),
            "sample_lat": coords.get(1),
            "sample_date": sample_date,
            "start_day": start_day,
            "split": split_name,
        }
    )


def extract_samples_with_metadata(
    image: ee.Image,
    detection_count: Optional[int],
    geometry: ee.Geometry,
    split_name: str,
    start_day: int,
    sample_date: ee.String,
    sampling_ratio: int = 0,
    sampling_limit_per_call: int = 60,
    no_fire_samples_per_day: int = 60,
    resolution: int = 1000,
    seed: int = 123,
) -> ee.FeatureCollection:
    """Samples positives/negatives and preserves point geometry + metadata."""
    feature_collection = ee.FeatureCollection([])
    if detection_count is None:
        # Detection stats can fail intermittently in EE reduce calls.
        # Fall back to class-0 sampling so the date is not dropped.
        pos_per_call = 0
        neg_per_call = max(1, no_fire_samples_per_day)
        num_calls = 1
    elif detection_count <= 0:
        pos_per_call = 0
        neg_per_call = max(1, no_fire_samples_per_day)
        num_calls = 1
    else:
        pos_per_call = max(1, sampling_limit_per_call // (sampling_ratio + 1))
        neg_per_call = pos_per_call * sampling_ratio
        num_calls = max(1, math.ceil(detection_count / pos_per_call))

    class_values: list[int] = []
    class_points: list[int] = []
    if neg_per_call > 0:
        class_values.append(0)
        class_points.append(int(neg_per_call))
    if pos_per_call > 0:
        class_values.append(1)
        class_points.append(int(pos_per_call))

    if not class_values:
        return feature_collection

    for _ in range(num_calls):
        samples = image.stratifiedSample(
            region=geometry,
            numPoints=0,
            classBand="detection",
            scale=resolution,
            seed=seed,
            classValues=class_values,
            classPoints=class_points,
            dropNulls=True,
            geometries=True,
        )
        samples = samples.map(
            lambda f: add_metadata(f, split_name, start_day, sample_date)
        )
        feature_collection = feature_collection.merge(samples)

    return feature_collection


def get_time_slices(
    window_start: ee.Date,
    projection: ee.Projection,
    lag: int = 0,
) -> tuple[ee.Image, ee.Image, ee.Image, ee.Image, ee.Image, ee.Image]:
    """Extracts temporal slices aligned to a 1-day prediction window."""
    window_end = window_start.advance(1, "day")

    drought = (
        window_reduce(
            source_key="DROUGHT_GRIDMET",
            start=window_start.advance(
                -lag - DATA_TIME_SAMPLING_DAYS["DROUGHT_GRIDMET"], "day"
            ),
            end=window_start.advance(-lag, "day"),
            reducer="median",
        )
        .reproject(projection)
        .resample("bicubic")
        .unmask(0)
    )
    vegetation = (
        window_reduce(
            source_key="VEGETATION_VIIRS",
            start=window_start.advance(
                -lag - DATA_TIME_SAMPLING_DAYS["VEGETATION_VIIRS"], "day"
            ),
            end=window_start.advance(-lag, "day"),
            reducer="median",
        )
        .reproject(projection)
        .resample("bicubic")
        .unmask(0)
    )
    weather = (
        window_reduce(
            source_key="WEATHER_GRIDMET",
            start=window_start.advance(
                -lag - DATA_TIME_SAMPLING_DAYS["WEATHER_GRIDMET"], "day"
            ),
            end=window_start.advance(-lag, "day"),
            reducer="median",
        )
        .reproject(projection.atScale(RESAMPLING_SCALE))
        .resample("bicubic")
        .unmask(0)
    )
    prev_fire = (
        window_reduce(
            source_key="FIRE_MODIS",
            start=window_start.advance(
                -lag - DATA_TIME_SAMPLING_DAYS["FIRE_MODIS"], "day"
            ),
            end=window_start.advance(-lag, "day"),
            reducer="max",
            apply_remove_mask=True,
        )
        .rename("PrevFireMask")
        .unmask(0)
    )
    fire = (
        window_reduce(
            source_key="FIRE_MODIS",
            start=window_start,
            end=window_end,
            reducer="max",
            apply_remove_mask=True,
        )
        .rename("FireMask")
        .unmask(0)
    )
    detection = (
        fire.clamp(6, 7)
        .subtract(6)
        .rename("detection")
        .unmask(0)
        .toInt8()
    )
    return drought, vegetation, weather, prev_fire, fire, detection


def export_feature_collection(
    feature_collection: ee.FeatureCollection,
    export_dest: str,
    bucket: str | None,
    folder: str,
    description: str,
    selectors: list[str],
) -> None:
    if export_dest == "gcs":
        task = ee.batch.Export.table.toCloudStorage(
            collection=feature_collection,
            description=description,
            bucket=bucket,
            fileNamePrefix=f"{folder.rstrip('/')}/{description}",
            fileFormat="TFRecord",
            selectors=selectors,
        )
    elif export_dest == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            fileFormat="TFRecord",
            selectors=selectors,
        )
    else:
        raise ValueError(f"Unsupported export_dest: {export_dest}")
    task.start()
    print(f"[EXPORT STARTED] dest={export_dest} job={description}")


def export_split(
    split_name: str,
    start_days: list[int],
    start_date: ee.Date,
    sampling_geometry: ee.Geometry,
    export_dest: str,
    bucket: str | None,
    folder: str,
    prefix: str,
    kernel_size: int,
    sampling_scale: int,
    sampling_ratio: int,
    sampling_limit_per_call: int,
    no_fire_samples_per_day: int,
    num_samples_per_file: int,
    seed: int,
    dry_run: bool,
) -> None:
    if not start_days:
        print(f"[SKIP] split={split_name}: no start days")
        return

    selectors = FEATURE_BANDS + LABEL_BANDS + METADATA_FIELDS
    if dry_run:
        print(
            f"[DRY RUN] split={split_name}, days={len(start_days)}, "
            f"selectors={selectors}"
        )
        return

    elevation = get_image("ELEVATION_SRTM").rename("elevation").toFloat().unmask(0)
    population = (
        ee.Image(
            ee.Algorithms.If(
                get_image_collection("POPULATION").size().gt(0),
                get_image_collection("POPULATION").median(),
                zero_image(DATA_BANDS["POPULATION"]),
            )
        )
        .rename("population")
        .toFloat()
        .unmask(0)
    )
    projection = get_image_collection("WEATHER_GRIDMET").first().select("pr").projection()
    feature_collection = ee.FeatureCollection([])
    file_count = 0

    for start_day in start_days:
        window_start = start_date.advance(start_day, "day")
        sample_date = window_start.format("YYYY-MM-dd")

        drought, vegetation, weather, prev_fire, fire, detection = get_time_slices(
            window_start, projection
        )
        image_list = [elevation, population, drought, vegetation, weather, prev_fire, fire]
        arrays = convert_features_to_arrays(image_list, kernel_size)
        to_sample = detection.addBands(arrays)

        fire_count = get_detection_count(
            detection_image=detection,
            geometry=sampling_geometry,
            sampling_scale=10 * sampling_scale,
        )
        if fire_count is None:
            print(
                f"[WARN] split={split_name} day_offset={start_day}: "
                "detection count unavailable; using balanced fallback sampling."
            )

        samples = extract_samples_with_metadata(
            image=to_sample,
            detection_count=fire_count,
            geometry=sampling_geometry,
            split_name=split_name,
            start_day=start_day,
            sample_date=sample_date,
            sampling_ratio=sampling_ratio,
            sampling_limit_per_call=sampling_limit_per_call,
            no_fire_samples_per_day=no_fire_samples_per_day,
            resolution=sampling_scale,
            seed=seed + start_day,
        )
        feature_collection = feature_collection.merge(samples)

        try:
            current_size = int(feature_collection.size().getInfo())
        except ee.EEException as exc:
            print(
                f"[WARN] split={split_name} day_offset={start_day}: "
                f"failed to evaluate collection size ({exc}); flushing accumulator."
            )
            current_size = 0
            feature_collection = ee.FeatureCollection([])

        if current_size > num_samples_per_file:
            description = f"{split_name}_{prefix}_{file_count:03d}"
            export_feature_collection(
                feature_collection=feature_collection,
                export_dest=export_dest,
                bucket=bucket,
                folder=folder,
                description=description,
                selectors=selectors,
            )
            file_count += 1
            feature_collection = ee.FeatureCollection([])

    try:
        remaining = int(feature_collection.size().getInfo())
    except ee.EEException:
        remaining = 0
    if remaining > 0:
        description = f"{split_name}_{prefix}_{file_count:03d}"
        export_feature_collection(
            feature_collection=feature_collection,
            export_dest=export_dest,
            bucket=bucket,
            folder=folder,
            description=description,
            selectors=selectors,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export NDWS-style TFRecords with sample geo/time metadata."
    )
    parser.add_argument(
        "--export_dest",
        choices=("drive", "gcs"),
        default="drive",
        help="Earth Engine export destination (default: drive)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket name (required only when --export_dest gcs)",
    )
    parser.add_argument(
        "--folder",
        default="wildfire_ndws",
        help="Destination folder (Drive folder name or GCS subfolder)",
    )
    parser.add_argument("--start_date", default="2020-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end_date", default="2021-01-01", help="YYYY-MM-DD")
    parser.add_argument("--prefix", default="ndws64_meta", help="Output filename prefix")
    parser.add_argument(
        "--region",
        choices=("ca", "us"),
        default="ca",
        help="Sampling region (default: ca)",
    )
    parser.add_argument("--kernel_size", type=int, default=64, help="Tile size in pixels")
    parser.add_argument(
        "--sampling_scale", type=int, default=1000, help="Sampling resolution in meters"
    )
    parser.add_argument(
        "--sampling_ratio",
        type=int,
        default=0,
        help="Negative-to-positive sampling ratio for fire days (default: 0)",
    )
    parser.add_argument(
        "--sampling_limit_per_call",
        type=int,
        default=60,
        help="Approximate samples requested per stratifiedSample call (default: 60)",
    )
    parser.add_argument(
        "--no_fire_samples_per_day",
        type=int,
        default=60,
        help="Class-0 samples requested when no positive fire pixel exists (default: 60)",
    )
    parser.add_argument(
        "--split_window_days",
        type=int,
        default=1,
        help="Day stride before split assignment (default: 1 for daily coverage)",
    )
    parser.add_argument(
        "--eval_split_ratio", type=float, default=0.1, help="Eval ratio (test uses same ratio)"
    )
    parser.add_argument(
        "--num_samples_per_file",
        type=int,
        default=2000,
        help="Approximate samples per exported TFRecord",
    )
    parser.add_argument("--seed", type=int, default=123, help="Split seed")
    parser.add_argument(
        "--ee_project",
        default=None,
        help="Google Cloud project id to use for Earth Engine initialization",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Compute split metadata only; no EE export"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if ee is None:
        print("[ERROR] Missing dependency: earthengine-api (`ee`).")
        print("Install with: pip install earthengine-api")
        return 2

    try:
        if args.ee_project:
            ee.Initialize(project=args.ee_project)
        else:
            ee.Initialize()
    except Exception as exc:
        print(f"[ERROR] Failed to initialize Earth Engine: {exc}")
        print(
            "Run `earthengine authenticate --project <YOUR_PROJECT_ID>` and "
            "enable Earth Engine API in that project, then retry."
        )
        if args.ee_project:
            print(f"[INFO] Attempted project: {args.ee_project}")
        else:
            print("[INFO] No --ee_project was provided; default EE project was used.")
        return 2

    if not args.dry_run and args.export_dest == "gcs" and not args.bucket:
        print("[ERROR] --bucket is required when --export_dest gcs.")
        return 2

    start_date = ee.Date(args.start_date)
    end_date = ee.Date(args.end_date)
    split_days = split_days_into_train_eval_test(
        start_date=start_date,
        end_date=end_date,
        split_ratio=args.eval_split_ratio,
        window_length_days=args.split_window_days,
        seed=args.seed,
    )
    sampling_geometry = get_sampling_geometry(args.region)

    print(f"[INFO] Sampling region: {args.region}")
    print(f"[INFO] Split day counts (window={args.split_window_days}):")
    for split_name in ("train", "eval", "test"):
        print(f"  - {split_name}: {len(split_days[split_name])} start days")

    for split_name in ("train", "eval", "test"):
        export_split(
            split_name=split_name,
            start_days=split_days[split_name],
            start_date=start_date,
            sampling_geometry=sampling_geometry,
            export_dest=args.export_dest,
            bucket=args.bucket,
            folder=args.folder,
            prefix=args.prefix,
            kernel_size=args.kernel_size,
            sampling_scale=args.sampling_scale,
            sampling_ratio=args.sampling_ratio,
            sampling_limit_per_call=args.sampling_limit_per_call,
            no_fire_samples_per_day=args.no_fire_samples_per_day,
            num_samples_per_file=args.num_samples_per_file,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[DONE] Dry run complete.")
    else:
        print("[DONE] Export tasks submitted to Earth Engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
