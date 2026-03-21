"""Spatial standards for geospatial preprocessing and feature engineering."""

# Canonical projection for all California-wide vector/raster joins.
TARGET_CRS_EPSG = 3310
TARGET_CRS = "EPSG:3310"
TARGET_CRS_NAME = "NAD83 / California Albers"

# Units implied by EPSG:3310.
DISTANCE_UNIT = "meter"
AREA_UNIT = "square_meter"

