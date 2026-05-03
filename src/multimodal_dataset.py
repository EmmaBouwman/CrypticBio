import torch
from PIL import Image
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset


class CrypticBioDataset(Dataset):
    def __init__(self, ids, name_to_id, db_path):
        self.name_to_id = name_to_id
        

        with DuckDBManager(db_path) as db:
            self.data = db.con.execute("""
                SELECT id, crypticbio_image, sentinel_image, scientificName
                FROM crypticbio WHERE id = ANY(?)""", 
                [ids]).df()


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cb_img = Image.open(row['crypticbio_image']).convert('RGB')
        sh_img = Image.open(row['sentinel_image']).convert('RGB')
        label  = self.name_to_id[row['scientificName']]
        return cb_img, sh_img, torch.tensor(label, dtype=torch.long)
    

