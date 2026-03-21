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


def get_image(key: str) -> ee.Image:
    return ee.Image(DATA_SOURCES[key]).select(DATA_BANDS[key])


def get_image_collection(key: str) -> ee.ImageCollection:
    return ee.ImageCollection(DATA_SOURCES[key]).select(DATA_BANDS[key])


def remove_mask(image: ee.Image) -> ee.Image:
    return image.updateMask(ee.Image(1))


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
    window_length_days: int = 8,
    seed: int = 123,
) -> dict[str, list[int]]:
    """Splits day indices into train/eval/test chunks."""
    num_days = int(ee.Date(end_date).difference(ee.Date(start_date), "day").getInfo())
    days = list(range(0, num_days, window_length_days))
    rng = random.Random(seed)
    rng.shuffle(days)

    num_eval = int(len(days) * split_ratio)
    return {
        "train": days[:-2 * num_eval] if num_eval > 0 else days,
        "eval": days[-2 * num_eval:-num_eval] if num_eval > 0 else [],
        "test": days[-num_eval:] if num_eval > 0 else [],
    }


def get_detection_count(
    detection_image: ee.Image, geometry: ee.Geometry, sampling_scale: int
) -> int:
    detection_stats = detection_image.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=sampling_scale
    )
    try:
        return int(detection_stats.get("detection").getInfo())
    except ee.EEException:
        return -1


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
    detection_count: int,
    geometry: ee.Geometry,
    split_name: str,
    start_day: int,
    sample_date: ee.String,
    sampling_ratio: int = 0,
    sampling_limit_per_call: int = 60,
    resolution: int = 1000,
    seed: int = 123,
) -> ee.FeatureCollection:
    """Samples positives/negatives and preserves point geometry + metadata."""
    feature_collection = ee.FeatureCollection([])
    num_per_call = sampling_limit_per_call // (sampling_ratio + 1)

    for _ in range(math.ceil(detection_count / num_per_call)):
        samples = image.stratifiedSample(
            region=geometry,
            numPoints=0,
            classBand="detection",
            scale=resolution,
            seed=seed,
            classValues=[0, 1],
            classPoints=[num_per_call * sampling_ratio, num_per_call],
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

    drought_ic = get_image_collection("DROUGHT_GRIDMET")
    vegetation_ic = get_image_collection("VEGETATION_VIIRS")
    weather_ic = get_image_collection("WEATHER_GRIDMET")
    fire_ic = get_image_collection("FIRE_MODIS")

    drought = (
        drought_ic.filterDate(
            window_start.advance(-lag - DATA_TIME_SAMPLING_DAYS["DROUGHT_GRIDMET"], "day"),
            window_start.advance(-lag, "day"),
        )
        .median()
        .reproject(projection)
        .resample("bicubic")
    )
    vegetation = (
        vegetation_ic.filterDate(
            window_start.advance(-lag - DATA_TIME_SAMPLING_DAYS["VEGETATION_VIIRS"], "day"),
            window_start.advance(-lag, "day"),
        )
        .median()
        .reproject(projection)
        .resample("bicubic")
    )
    weather = (
        weather_ic.filterDate(
            window_start.advance(-lag - DATA_TIME_SAMPLING_DAYS["WEATHER_GRIDMET"], "day"),
            window_start.advance(-lag, "day"),
        )
        .median()
        .reproject(projection.atScale(RESAMPLING_SCALE))
        .resample("bicubic")
    )
    prev_fire = (
        fire_ic.filterDate(
            window_start.advance(-lag - DATA_TIME_SAMPLING_DAYS["FIRE_MODIS"], "day"),
            window_start.advance(-lag, "day"),
        )
        .map(remove_mask)
        .max()
        .rename("PrevFireMask")
    )
    fire = fire_ic.filterDate(window_start, window_end).map(remove_mask).max()
    detection = fire.clamp(6, 7).subtract(6).rename("detection")
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
    export_dest: str,
    bucket: str | None,
    folder: str,
    prefix: str,
    kernel_size: int,
    sampling_scale: int,
    num_samples_per_file: int,
    dry_run: bool,
) -> None:
    if not start_days:
        print(f"[SKIP] split={split_name}: no start days")
        return

    selectors = FEATURE_BANDS + LABEL_BANDS + METADATA_FIELDS
    if dry_run:
        print(f"[DRY RUN] split={split_name}, selectors={selectors}")
        return

    elevation = get_image("ELEVATION_SRTM")
    population = (
        get_image_collection("POPULATION")
        .filterDate(start_date, start_date.advance(max(start_days), "day"))
        .median()
        .rename("population")
    )
    projection = get_image_collection("WEATHER_GRIDMET").first().select("pr").projection()
    geometry = ee.Geometry.Rectangle(US_BBOX)

    all_days = [day + i for day in start_days for i in range(7)]
    feature_collection = ee.FeatureCollection([])
    file_count = 0

    for start_day in all_days:
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
            geometry=geometry,
            sampling_scale=10 * sampling_scale,
        )
        if fire_count <= 0:
            continue

        samples = extract_samples_with_metadata(
            image=to_sample,
            detection_count=fire_count,
            geometry=geometry,
            split_name=split_name,
            start_day=start_day,
            sample_date=sample_date,
            sampling_ratio=0,
            sampling_limit_per_call=60,
            resolution=sampling_scale,
            seed=123,
        )
        feature_collection = feature_collection.merge(samples)

        try:
            current_size = int(feature_collection.size().getInfo())
        except ee.EEException:
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
    parser.add_argument("--kernel_size", type=int, default=64, help="Tile size in pixels")
    parser.add_argument(
        "--sampling_scale", type=int, default=1000, help="Sampling resolution in meters"
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
        window_length_days=8,
        seed=args.seed,
    )

    print("[INFO] Split chunk counts:")
    for split_name in ("train", "eval", "test"):
        print(f"  - {split_name}: {len(split_days[split_name])} start chunks")

    for split_name in ("train", "eval", "test"):
        export_split(
            split_name=split_name,
            start_days=split_days[split_name],
            start_date=start_date,
            export_dest=args.export_dest,
            bucket=args.bucket,
            folder=args.folder,
            prefix=args.prefix,
            kernel_size=args.kernel_size,
            sampling_scale=args.sampling_scale,
            num_samples_per_file=args.num_samples_per_file,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[DONE] Dry run complete.")
    else:
        print("[DONE] Export tasks submitted to Earth Engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
