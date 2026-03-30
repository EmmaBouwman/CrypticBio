from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import duckdb
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

class CrypticBioDataset(Dataset):
    def __init__(self, db_path: Path, table_name: str = "crypticbio", transform=None):
        self.db_path = db_path
        self.table_name = table_name
        self.transform = transform

        # Connect to DuckDB en alle rows ophalen
        self.con = duckdb.connect(self.db_path)
        self.data = self.con.execute(f"""
            SELECT id, crypticbio_image, sentinel_image
            FROM {self.table_name}
        """).fetchall()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        _, cb_path, sh_path = row

       
        cb_img = Image.open(cb_path).convert("RGB")
        sh_img = Image.open(sh_path).convert("RGB")

        
        if self.transform:
            cb_img = self.transform(cb_img)
            sh_img = self.transform(sh_img)
        else:
          
            cb_img = torch.tensor(np.array(cb_img)).permute(2, 0, 1).float() / 255.0
            sh_img = torch.tensor(np.array(sh_img)).permute(2, 0, 1).float() / 255.0

        return {"crypticbio": cb_img, "sentinel": sh_img}

def get_dataloader(db_path: Path, batch_size=4, shuffle=True, num_workers=2, transform=None):
    dataset = CrypticBioDataset(db_path=db_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


load_dotenv(".env")

base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")

dataloader = get_dataloader(db_path, batch_size=2)

batch = next(iter(dataloader))
print("CrypticBio batch shape:", batch["crypticbio"].shape)
print("Sentinel batch shape:", batch["sentinel"].shape)