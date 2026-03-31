import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_gather import DuckDBManager

load_dotenv(".env")

base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")
cb_folder = base_path / os.getenv("CB_IMAGE_PATH")
sh_folder = base_path / os.getenv("SENTINEL_IMAGE_PATH")

with DuckDBManager(db_path) as db:
    db.extend_db(cb_folder, sh_folder, False)
