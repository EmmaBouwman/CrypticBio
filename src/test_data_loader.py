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
        self.manager = DuckDBManager(db_path)

        with self.manager as db:
            self.data = db.con.execute("""
                SELECT id, crypticbio_image, sentinel_image, scientific_name
                FROM crypticbio
                ORDER BY RANDOM()
            """).fetchall()
    
    def __len__(self):
        return len(self.data)
    
    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        return torch.tensor(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        _id, cb_path, sh_path = self.data[idx]

        cb_img = self._load_image(cb_path)
        sh_img = self._load_image(sh_path)

        return {
            "id": _id,
            "cb": cb_img,
            "sh": sh_img
        }
    
      
if __name__ == "__main__":
    dataset = CrypticBioDataset(db_path)

    print("dataset size:", len(dataset))
    print("sample:", dataset[0])


    
