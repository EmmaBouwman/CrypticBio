from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
from dotenv import load_dotenv
from geopy.distance import distance
from huggingface_hub import snapshot_download
from PIL import Image
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from sentinelhub.exceptions import DownloadFailedException

load_dotenv()

base_folder = Path(os.getenv("DATA_FOLDER"))
cb_image_path = base_folder / os.getenv("CB_IMAGE_PATH", "").strip("/")
sh_image_path = base_folder / os.getenv("SENTINEL_IMAGE_PATH", "").strip("/")
parquets_before_download = base_folder / "parquets"
parquets = base_folder / os.getenv("PARQUETS_PATH", "").strip("/")
db_path = base_folder / os.getenv("DATABASE")


def check_exists_dir(path: Path):
    if not path.exists():
        print("Created the directory {path}")
        path.mkdir(parents=True, exist_ok=True)


def download_parquests(parquet_path):
    snapshot_download(
        repo_id="gmanolache/CrypticBio",
        repo_type="dataset",
        local_dir=parquet_path,
        # This only grabs the parquet files in the 'CrypticBio' subdirectory
        allow_patterns="CrypticBio/*.parquet",
        local_dir_use_symlinks=False,
        resume_download=True,  # Allows you to restart if your internet cuts out
    )
    print(f"Finished downloading parquet files and are stored at {parquet_path}")


def create_database(db_path, parquet_path):
    if os.path.exists(db_path):
        print(
            "Database already exists, "
            "make sure you check the database and delete if needed"
        )
    else:
        con = duckdb.connect(db_path)

        # Create a table directly from a glob pattern of parquet files
        con.execute(f"""
            CREATE OR REPLACE TABLE crypticbio AS 
            SELECT * FROM read_parquet('{parquet_path}/CrypticBio/*.parquet')
        """)
        print(f"Finished creating the database at {db_path}")


def save_image(path, image):
    print(path)
    img = Image.fromarray(image.astype(np.uint8))
    img.save(path)


def get_bounding_box(lat, lon, size_meters=10):
    d = distance(meters=size_meters / 2)

    max_lat = d.destination((lat, lon), bearing=0).latitude
    min_lat = d.destination((lat, lon), bearing=180).latitude
    max_lon = d.destination((lat, lon), bearing=90).longitude
    min_lon = d.destination((lat, lon), bearing=270).longitude

    return [min_lon, min_lat, max_lon, max_lat]


def get_date_range(date_str, days_buffer=15):
    center_date = datetime.strptime(date_str, "%Y-%m-%d")
    delta = timedelta(days=days_buffer)

    start_date = center_date - delta
    end_date = center_date + delta

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_image(lat, lon, date, config, save_path, width=2500, resolution=10, attempts=3):
    bbox = BBox(bbox=get_bounding_box(lat, lon, width), crs=CRS.WGS84)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    start_date, end_date = get_date_range(date)

    evalscript_true_color = """
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )

    for i in range(attempts):
        try:
            true_color_imgs = request_true_color.get_data()
            continue
        except DownloadFailedException as e:
            response = getattr(e.request_exception, "response", None)
            status_code = getattr(response, "status_code", None)

            if status_code == 429:
                print("Rate limit hit...")
                time.sleep((2**i) * 10)
                continue
            raise e
    if true_color_imgs:
        save_image(save_path, true_color_imgs[0])
    else:
        raise RuntimeError(
            "Failed to gather image data from API. Probably due to rate limit"
        )
