import os
import torch
from PIL import Image
from pathlib import Path
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset
from dotenv import load_dotenv


class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        cb, sh, label = self.subset[index]
        if self.transform:
            cb = self.transform(cb)
            sh = self.transform(sh)
        return cb, sh, label


class CrypticBioDataset(Dataset):
    def __init__(self, ids, name_to_id, cb_folder, sh_folder, db_path, limit=None):
        self.ids = ids[:limit] if limit else ids
        self.name_to_id = name_to_id
        self.cb_folder = Path(cb_folder)
        self.sh_folder = Path(sh_folder)

        with DuckDBManager(db_path) as db:
            df = db.con.execute("""
                SELECT id, scientificName
                FROM crypticbio
            """).df()

        self.id_to_label = dict(zip(df["id"].astype(str), df["scientificName"]))

        self.sh_lookup = {}

        for f in self.sh_folder.glob("*.png"):
            file_id = f.name.split("_")[0]
            self.sh_lookup[file_id] = f


    def __len__(self):
        return len(self.ids)
    

    def _load_image(self, path, size=(224, 224)):
        return Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    

    def __getitem__(self, idx):
        _id = str(self.ids[idx])

        cb_path = self.cb_folder / f"{_id}.png"
        if not cb_path.exists():
            raise FileNotFoundError(f"Missing CB image: {cb_path}")

        sh_path = self.sh_lookup.get(_id)
        if sh_path is None:
            raise FileNotFoundError(f"Missing SH image for id {_id}")

        cb_img = self._load_image(cb_path)
        sh_img = self._load_image(sh_path)

        label_name = self.id_to_label.get(_id)
        if label_name is None:
            raise ValueError(f"no label for id {_id}")
        label = self.name_to_id[label_name]

        return cb_img, sh_img, torch.tensor(label)
    

