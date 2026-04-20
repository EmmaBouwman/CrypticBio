import os
import glob
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv


load_dotenv(".env")


base_folder = Path(os.getenv("DATA_FOLDER"))
db_path = base_folder / os.getenv("DATABASE")
cb_folder = base_folder / os.getenv("CB_IMAGE_PATH")


class CrypticBioDataset(Dataset):
    def __init__(self, ids, name_to_id, cb_folder, sh_folder, limit=None):
        self.ids = ids[:limit] if limit else ids
        self.name_to_id = name_to_id

        self.cb_folder = Path(cb_folder)
        self.sh_folder = Path(sh_folder)


        with DuckDBManager(db_path, read_only=True) as db:
            df = db.con.execute("""
                SELECT id, scientificName
                FROM crypticbio
            """).df()

        # ensure string keys
        self.id_to_label = dict(zip(df["id"].astype(str), df["scientificName"]))


        # sentinel lookup
        self.sh_lookup = {}

        for f in self.sh_folder.glob("*.png"):
            file_id = f.name.split("_")[0]
            self.sh_lookup[file_id] = f


    def __len__(self):
        return len(self.ids)
    

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        return torch.tensor(arr).permute(2, 0, 1).float() / 255.0
    

    def __getitem__(self, idx):
        _id = str(self.ids[idx])

        cb_path = self.cb_folder / f"{_id}.png"

        if not cb_path.exists():
            raise FileNotFoundError(f"Missing CB image: {cb_path}")

      
        # Sentinel image lookup in dict
        sh_path = self.sh_lookup.get(_id)

        if sh_path is None:
            raise FileNotFoundError(f"Missing SH image for id {_id}")

        # load images
        cb_img = self._load_image(cb_path)
        sh_img = self._load_image(sh_path)

      
        # label
        label_name = self.id_to_label.get(_id)

        if label_name is None:
            raise ValueError(f"no label for id {_id}")
        
        label = self.name_to_id[label_name]


        return cb_img, sh_img, torch.tensor(label)
    

