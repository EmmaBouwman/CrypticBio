import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.data_gather import DuckDBManager

load_dotenv()

# 1. Setup Paths
base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")
sh_folder = base_path / os.getenv("SH_IMAGE_PATH")

# 2. Extract IDs and Paths into a List of Dictionaries
# This replicates your logic: taking the first part of the filename before "_"
data_for_df = []
for f in os.listdir(sh_folder):
    if f.endswith('.png'):
        try:
            # Extract ID: e.g., "123_mask.png" -> 123
            row_id = int(f.split("_")[0])
            full_path = str(sh_folder / f)
            data_for_df.append({"id": row_id, "new_sentinel_path": full_path})
        except ValueError:
            print(f"Skipping file with invalid ID format: {f}")

# 3. Create DataFrame for Bulk Update
df_updates = pd.DataFrame(data_for_df)

# 4. Perform Bulk Update with DuckDB
with DuckDBManager(db_path, readOnly=False) as db:
    if df_updates.empty:
        print("No valid files found to update.")
    else:
        # Register the DataFrame as a temporary virtual table
        db.con.register('updates_tmp', df_updates)
        
        print(f"Syncing {len(df_updates)} paths to database using bulk update...")

        # Perform the Atomic Update: Join main table with temp table on 'id'
        db.con.execute("""
            UPDATE crypticbio 
            SET sentinel_image = updates_tmp.new_sentinel_path 
            FROM updates_tmp 
            WHERE crypticbio.id = updates_tmp.id
        """)

        db.con.unregister('updates_tmp')
        
        print("Database paths synchronized successfully.")

    # 5. Quick Verification
    first_id = int(df_updates.iloc[0]['id'])
    last_id = int(df_updates.iloc[-1]['id'])
    verification_df = db.con.execute("""
        SELECT id, sentinel_image 
        FROM crypticbio 
        WHERE id = ? OR id = ?
        ORDER BY id ASC
    """, [first_id, last_id]).df()

    if len(verification_df) > 0:
        print(verification_df.iloc[0]["sentinel_image"].split("/")[-1])
        print(verification_df.iloc[1]["sentinel_image"].split("/")[-1])