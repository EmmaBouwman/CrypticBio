import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_gather import DuckDBManager, download_parquests, check_exists_dir

load_dotenv(".env")

base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")
cb_folder = base_path / os.getenv("CB_IMAGE_PATH")
sh_folder = base_path / os.getenv("SH_IMAGE_PATH")

parquet_path = base_path / "parquets"

check_exists_dir(base_path)
check_exists_dir(cb_folder)
check_exists_dir(sh_folder)
check_exists_dir(parquet_path)

download_parquests(parquet_path)
downloaded_parquets_path = base_path / os.getenv("PARQUETS_PATH")

with DuckDBManager(db_path, readOnly=False) as db:
    db.create_db(downloaded_parquets_path)
    db.extend_db(cb_folder, sh_folder, False)