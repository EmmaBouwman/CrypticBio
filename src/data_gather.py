from __future__ import annotations

import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import requests
from geopy.distance import distance
from huggingface_hub import snapshot_download
from PIL import Image
from requests.exceptions import HTTPError
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from sentinelhub.exceptions import DownloadFailedException


class DuckDBManager:
    """
    A database manager for handling DuckDB database connections and table initialization

    Mostly be used with "with DuckDBManager(<path>, readOnly=True) as db:"

    Attributes:
        db_path (Path): File system path to the DuckDB database file.
        table_name (str): The name of the table to interact with or create.
        con (duckdb.DuckDBPyConnection): The active connection object, initialized 
                                         in __enter__.
        readOnly (bool): If True, opens the database in read-only mode.
    """

    def __init__(
        self, db_path: Path, table_name: str = "crypticbio", readOnly: bool = True
    ):
        """
        Initializes the manager with the database path and configuration.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.con = None
        self.readOnly = readOnly

    def __enter__(self):
        """
        Establishes the connection to DuckDB and returns the manager instance.
        """
        self.con = duckdb.connect(self.db_path, read_only=self.readOnly)
        return self

    def __exit__(self, _type, _value, _traceback):
        """
        Closes the connection and resets the connection attribute to None.
        """
        if self.con:
            self.con.close()
            self.con = None

    def create_db(self, parquet_path: Path):
        """
        Creates a table from Parquet files if it does not already exist.
        """
        try:
            # create table from parquets
            self.con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} AS 
                SELECT * FROM read_parquet('{parquet_path}/*.parquet')
            """)
        except duckdb.IOException as e:
            # Specific handling for missing files or permission issues
            raise RuntimeError(
                f"File Error: Could not read Parquet files at {parquet_path}."
            ) from e
        except Exception as e:
            # General catch-all for SQL syntax errors or schema mismatches
            raise RuntimeError(
                f"An unexpected error occurred while creating {self.table_name}"
            ) from e

    def extend_db(
        self, crypticbio_img_folder: Path, sentinel_img_folder: Path, show_sample=True
    ):

        # adding columns
        self.con.execute(
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS id INTEGER"
        )
        self.con.execute(
            f"ALTER TABLE {self.table_name} "
            + "ADD COLUMN IF NOT EXISTS crypticbio_image VARCHAR"
        )
        self.con.execute(
            f"ALTER TABLE {self.table_name} "
            + "ADD COLUMN IF NOT EXISTS sentinel_image VARCHAR"
        )

        # row ID
        self.con.execute(f"UPDATE {self.table_name} SET id = rowid")

        # image paths
        self.con.execute(f"""
        UPDATE {self.table_name}
        SET crypticbio_image = '{crypticbio_img_folder}/' || id || '.png',
            sentinel_image = '{sentinel_img_folder}/' || id || '.png'
        """)

        print(f"Database {self.table_name} extended with id and image columns")

        # check:
        if show_sample:
            print("First 5 rows after extending:")
            result = self.con.execute(f"""
                SELECT id, crypticbio_image, sentinel_image 
                FROM {self.table_name} 
                LIMIT 5
            """).fetchall()
            for row in result:
                print(row)

    def delete_db(self):
        try:
            self.con.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        except duckdb.TransactionException:
            raise RuntimeError(
                f"Could not drop {self.table_name}, "
                "because it is currently being used by another process."
            )
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while dropping {self.table_name}"
            ) from e


class SentinelHubManager:
    """
    Manages the acquisition and processing of satellite imagery via Sentinel Hub.

    Attributes:
        config (SHConfig): Configuration object for Sentinel Hub authentication.
        width (int): The width/height of the square area of interest in meters.
        resolution (int): The spatial resolution of the image in meters per pixel.
        attempts (int): Maximum number of retry attempts for API requests.
        evalscript (str): JavaScript-based Evalscript for band selection and processing.
    """

    def __init__(self, config, width=2500, resolution=10, attempts=3):
        """
        Initializes the manager with API configurations and default image parameters.
        """
        self.config = config
        self.width = width
        self.resolution = resolution
        self.attempts = attempts
        self.evalscript = """
            function setup() {
                return {
                    input: [{ bands: ["B02", "B03", "B04", "CLP"] }],
                    output: [
                        { id: "default", bands: 3 }, 
                        { id: "metadata", bands: 1 }
                    ]
                };
            }
            function evaluatePixel(sample) {
                return {
                    default: [sample.B04, sample.B03, sample.B02], 
                    metadata: [sample.CLP] 
                };
            }
        """

    def _get_bounding_box(self, lat, lon):
        """
        Calculates a bounding box centered on a lat/lon coordinate.
        """
        d = distance(meters=self.width / 2)

        max_lat = d.destination((lat, lon), bearing=0).latitude
        min_lat = d.destination((lat, lon), bearing=180).latitude
        max_lon = d.destination((lat, lon), bearing=90).longitude
        min_lon = d.destination((lat, lon), bearing=270).longitude

        return [min_lon, min_lat, max_lon, max_lat]

    def _get_date_range(self, date_str, days_buffer=15):
        """
        Generates a start and end date string centered around a specific date.
        """
        center_date = datetime.strptime(date_str, "%Y-%m-%d")
        delta = timedelta(days=days_buffer)

        start = (center_date - delta).strftime("%Y-%m-%d")
        end = (center_date + delta).strftime("%Y-%m-%d")

        return start, end

    def get_and_save_image(self, lat, lon, date, save_path, db_path):
        """
        Coordinates the full workflow: request, cloud check, retry, and DB update.
        Note: If cloud cover is too high, it recursively attempts previous years.
        """
        bbox_coords = self._get_bounding_box(lat, lon)
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        bbox_size = bbox_to_dimensions(bbox, resolution=self.resolution)
        start_date, end_date = self._get_date_range(date)

        for i in range(3):
            # Perform the request
            request = SentinelHubRequest(
                evalscript=self.evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(start_date, end_date),
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.PNG),
                    SentinelHubRequest.output_response("metadata", MimeType.TIFF),
                ],
                bbox=bbox,
                size=bbox_size,
                config=self.config,
            )

            images = self._request_with_retry(request)

            if images:
                # get data
                data = images[0]["default.png"]
                cloud = images[0]["metadata.tif"]
                # Calculate cloud coverage percentage from the CLP band
                cloud_score = int((np.mean(cloud) / 255) * 100)

                if cloud_score < 75:  # Logic check: Retries previous year if too cloudy
                    date = datetime.strptime(date, "%Y-%m-%d")
                    date = date.replace(year=date.year - 1)
                    date = date.strftime("%Y-%m-%d")
                    start_date, end_date = self._get_date_range(date)
                    continue

                base_name, extension = os.path.splitext(save_path)
                final_save_path = f"{base_name}_{cloud_score}{extension}"

                self._save_to_location(final_save_path, data)
                # Ensure record is updated in the database for the new image path
                with DuckDBManager(db_path, readOnly=False) as db:
                    db.con.execute(
                        """
                        UPDATE crypticbio 
                        SET sentinel_image = ?
                        WHERE id = ?
                    """,
                        [final_save_path, int(base_name.split("/")[-1])],
                    )

                print(f"Success: Image found for year {date}")

                return True, final_save_path
            else:
                print(f"Too cloudy or no data for {date}. Trying previous year...")
                date = datetime.strptime(date, "%Y-%m-%d")
                date = date.replace(year=date.year - 1)
                date = date.strftime("%Y-%m-%d")
                start_date, end_date = self._get_date_range(date)

        return False, ""

    def _request_with_retry(self, request):
        """
        Performs network requests with exponential backoff for rate limiting.
        """
        for i in range(self.attempts):
            try:
                return request.get_data()
            except (DownloadFailedException, HTTPError) as e:
                if isinstance(e, DownloadFailedException):
                    response = getattr(e.request_exception, "response", None)
                    status_code = getattr(response, "status_code", None)

                    if status_code == 429:  # Too many requests
                        print(f"Rate limit hit. Retrying in {(2**i) * 10}s...")
                        time.sleep((2**i) * 10)
                        continue
                    raise e
                else:
                    raise e
        warnings.warn(
            f"Failed to download image after {self.attempts} attempts.", RuntimeWarning
        )

    def _save_to_location(self, path, image_array):
        """
        Applies brightness corrections and saves the image to disk.
        """
        try:
            # Boost brightness for visual clarity before saving
            brigher_image = np.clip(image_array * 2.5, 0, 255)
            img = Image.fromarray(brigher_image.astype(np.uint8))
            img.save(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Directory does not exist for path: {path}. ")
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot write to {path}. ")
        except Exception as e:
            raise RuntimeError(
                "An unexpected error occurred while saving the image."
            ) from e


class CrypticImageManager:
    """
    Handles the downloading of ground-level images from external URLs.

    Attributes:
        id (str): Unique identifier for the image record.
        url (str): The external source URL of the image.
        attempts (int): Maximum number of retry attempts for the download.
        save_path (str): Local filesystem path where the image will be stored.
    """

    def __init__(self, row, attempts=3):
        """
        Initializes the manager using a row from the database metadata.
        """
        self.id = str(row["id"])
        self.url = row["url"]
        self.attempts = attempts
        self.save_path = row["crypticbio_image"]

    def get_and_save_image(self):
        """
        Executes the download and triggers storage to the local disk.
        """
        content = self._request_with_retry()
        if content:
            self._save_to_location(content)
            return True
        return False

    def _save_to_location(self, content):
        """
        Writes binary image content to the specified file path.
        """
        try:
            with open(self.save_path, "wb") as f:
                f.write(content)
        except OSError as e:
            raise OSError(f"Error saving file {self.id}") from e

    def _request_with_retry(self):
        """
        Attempts to download image content with exponential backoff on failure.
        """
        for i in range(self.attempts):
            try:
                response = requests.get(self.url, timeout=10)
                response.raise_for_status()
                return response.content
            except requests.RequestException:
                time.sleep(10 * (2**i))

        warnings.warn(
            f"Failed to download image after {self.attempts} attempts.", RuntimeWarning
        )


def check_exists_dir(path):
    """
    Ensures that a directory exists at the specified path
    or creates it.

    Args:
        path (str or Path): The filesystem path to check or create.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def download_parquests(parquet_path):
    """
    Downloads the CrypticBio dataset metadata from Hugging Face.

    Args:
        parquet_path (str or Path): The local directory where the
            parquet files will be stored.
    """
    snapshot_download(
        repo_id="gmanolache/CrypticBio",
        repo_type="dataset",
        local_dir=parquet_path,
        # This only grabs the parquet files in the 'CrypticBio' subdirectory
        allow_patterns="CrypticBio/*.parquet",
        local_dir_use_symlinks=False,
        resume_download=True,  # Allows you to restart if your internet cuts out
    )
