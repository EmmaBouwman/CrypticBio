from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
from pathlib import Path
import duckdb
import time
import numpy as np
import pandas as pd

load_dotenv()

base_folder = Path(os.getenv('DATA_FOLDER'))
cb_image_path = base_folder / os.getenv('CB_IMAGE_PATH', '').strip('/')
sentinel_image_path = base_folder / os.getenv('SENTINEL_IMAGE_PATH', '').strip('/')
parquets = base_folder / "parquets"
db_path = base_folder / os.getenv('DATABASE')

def download_parquests(parquet_path):
    snapshot_download(
        repo_id="gmanolache/CrypticBio", 
        repo_type="dataset",
        local_dir=parquet_path,
        # This only grabs the parquet files in the 'CrypticBio' subdirectory
        allow_patterns="CrypticBio/*.parquet", 
        local_dir_use_symlinks=False,
        resume_download=True  # Allows you to restart if your internet cuts out
    )
    print(f"Finished downloading parquet files and are stored at {parquet_path}")

def create_database(db_path, parquet_path):
    if os.path.exists(db_path):
        print("Database already exists, make sure you check the database and delete if needed")
    else:
        con = duckdb.connect(db_path)

        # Create a table directly from a glob pattern of parquet files
        con.execute(f"""
            CREATE OR REPLACE TABLE crypticbio AS 
            SELECT * FROM read_parquet('{parquet_path}/CrypticBio/*.parquet')
        """)
        print(f"Finished creating the database at {db_path}")