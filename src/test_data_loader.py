import os
import duckdb
from pathlib import Path
import pandas as pd
from src.data_gather import DuckDBManager
from dotenv import load_dotenv
load_dotenv(".env")

base_folder  = Path(os.getenv("DATA_FOLDER"))
db_path = base_folder / os.getenv("DATABASE")


with DuckDBManager(db_path, read_only=True) as db:
    for row in db.con.execute("""
        SELECT id, crypticbio_image, sentinel_image
        FROM crypticbio
        LIMIT 100
    """).fetchall():
        

