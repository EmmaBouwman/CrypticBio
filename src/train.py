import os
import torch
import torch.nn as nn
from pathlib import Path
from early_fusion import EarlyFusionModel
from src.data_gather import DuckDBManager
from multimodal_dataset import CrypticBioDataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv

print("start")
load_dotenv(".env")

#folders
base_folder = Path(os.getenv("DATA_FOLDER"))
db_path = base_folder / os.getenv("DATABASE")
cb_folder = base_folder / os.getenv("CB_IMAGE_PATH")
sh_folder = base_folder / os.getenv("SH_IMAGE_PATH")


#IDs from CB folder
ids= [
    os.path.splitext(f)[0]
    for f in os.listdir(cb_folder)
    if f.endswith(".png")
    ]

#define name_to_id
with DuckDBManager(db_path, read_only=True) as db:
    species = db.con.execute("""
        SELECT DISTINCT scientificName FROM crypticbio
    """).df()["scientificName"].tolist()

name_to_id = {name: i for i, name in enumerate(species)}

print("before dataset")
#dataloader
dataset = CrypticBioDataset(
    ids=ids,
    name_to_id=name_to_id,
    cb_folder=cb_folder,
    sh_folder=sh_folder
)
print("after dataset")

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)



#main training loop
if __name__=="__main__":

    model = EarlyFusionModel(num_classes=len(name_to_id))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("dataset size:", len(dataset))
    print("starting training loop")

    for cb, sh, labels in loader:
        print("batch loaded")
        out = model(cb, sh)

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss:", loss.item())
        break 

