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
    
    bad_ids = set()
    if os.path.exists("bad_rows.txt"):
        with open("bad_rows.txt", "r") as f:
            bad_ids = {line.split()[0] for line in f if line.strip()}
    
    for idx, row in df.iterrows():
        if str(row['id']) in bad_ids:
            continue
        if os.path.exists(row["crypticbio_image"]):
            continue
        print(f"--- Processing {row['id']}, {idx} out of {len(row_ids) - 1} items ---")
        print(f"--- {row['scientificName']} ---")
        try:
            target_date = f"{int(row['year'])}-{int(row['month'])}-{int(row['day'])}"
            flag, image_path = sh.get_and_save_image(
                row["decimalLatitude"],
                row["decimalLongitude"],
                target_date,
                row["sentinel_image"],
            )
            if flag is False:
                with open("bad_rows.txt", "a") as f:
                    f.write(f"{row['id']}\n")
                bad_ids.add(str(row['id']))
                continue
            
            cb = CrypticImageManager(row)
            flag_2 = cb.get_and_save_image()
            if flag_2 is False:
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                with open("bad_rows.txt", "a") as f:
                    f.write(f"{row['id']}\n")
                bad_ids.add(str(row['id'])) 
                continue
        except Exception as e:
            print(f"Error at ID {row['id']}: {e}")
            with open("bad_rows.txt", "a") as f:
                f.write(f"{row['id']} - Error: {str(e)}\n")
            bad_ids.add(str(row['id']))

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
