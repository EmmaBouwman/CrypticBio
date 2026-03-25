import os
from pathlib import Path
from dotenv import load_dotenv
from sentinelhub import SHConfig
from pathlib import Path
from src.data_gather import DuckDBManager



load_dotenv(".env")

base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")

with DuckDBManager(db_path) as db:
    db.extend_db()