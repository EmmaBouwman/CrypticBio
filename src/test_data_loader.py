import os
import duckdb
from PIL import Image
from pathlib import Path
import pandas as pd
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv


load_dotenv(".env")


base_folder  = Path(os.getenv("DATA_FOLDER"))
db_path = base_folder / os.getenv("DATABASE")

class CrypticBioDataset(Dataset):
    def __init__(self, db_path):
        self.manager = DuckDBManager(db_path, read_only=True)

        with self.manager as db:
            self.data = db.con.execute("""
                SELECT id, crypticbio_image, sentinel_image
                FROM crypticbio
                ORDER BY RANDOM()
            """).fetchall()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _id, cb_path, sh_path = self.data[idx]

        return {
            "id": _id,
            "cb": cb_path,
            "sh": sh_path
        }
    
      
if __name__ == "__main__":
    dataset = CrypticBioDataset(db_path)

    print("dataset size:", len(dataset))
    print("sample:", dataset[0])


    
