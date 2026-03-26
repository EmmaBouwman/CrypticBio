# uv run scripts/image_download.py "<path_to_csv>"" "<cluster_name>"
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sentinelhub import SHConfig

from src.data_gather import (
    CrypticImageManager,
    DuckDBManager,
    SentinelHubManager,
    check_exists_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch crypticbio rows based on cluster CSV."
    )
    parser.add_argument("csv_file", type=str, help="Path to the cluster CSV file")
    parser.add_argument(
        "cluster_name", type=str, help="The column name to extract (e.g., 'cluster 1')"
    )
    args = parser.parse_args()

    csv_path = Path.cwd() / args.csv_file
    if not csv_path.exists():
        raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")

    try:
        csv = pd.read_csv(csv_path)
        row_ids = csv[args.cluster_name].dropna().astype(int).tolist()
    except KeyError:
        raise KeyError(
            f"""Error: Cluster name does not exist: {args.cluster_name} 
            from columns: {list(csv.columns)}"""
        )

    with DuckDBManager(db_path) as db:
        df = db.con.execute(
            "SELECT * FROM crypticbio WHERE id = ANY(?)", [row_ids]
        ).df()

    sh = SentinelHubManager(config)
    for idx, row in df.iterrows():
        print(f"--- Processing {idx} out of {len(row_ids) - 1} items ---")
        target_date = f"{int(row['year'])}-{int(row['month'])}-{int(row['day'])}"
        sh.get_and_save_image(
            row["decimalLatitude"],
            row["decimalLongitude"],
            target_date,
            row["sentinel_image"],
        )
        cb = CrypticImageManager(row)
        cb.get_and_save_image()


if __name__ == "__main__":
    load_dotenv(".env")
    load_dotenv(".env_sentinel")

    DATA_FOLDER = Path(os.getenv("DATA_FOLDER"))
    CB_IMAGE_PATH = os.getenv("CB_IMAGE_PATH")
    SH_IMAGE_PATH = os.getenv("SH_IMAGE_PATH")
    DB_NAME = os.getenv("DATABASE")

    cb_save_path = DATA_FOLDER / CB_IMAGE_PATH
    sh_save_path = DATA_FOLDER / SH_IMAGE_PATH
    db_path = DATA_FOLDER / DB_NAME

    check_exists_dir(cb_save_path)
    check_exists_dir(sh_save_path)

    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    main()
