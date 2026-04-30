import os
import torch
import torch.nn as nn
from pathlib import Path
from early_fusion import EarlyFusionModel
from src.data_gather import DuckDBManager
from multimodal_dataset import CrypticBioDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dotenv import load_dotenv


def main():
    load_dotenv(".env")

    
    base_folder = Path(os.getenv("DATA_FOLDER"))
    db_path = base_folder / os.getenv("DATABASE")
    cb_folder = base_folder / os.getenv("CB_IMAGE_PATH")
    sh_folder = base_folder / os.getenv("SH_IMAGE_PATH")

    ids = [int(os.path.splitext(f)[0]) for f in os.listdir(cb_folder) if f.endswith(".png")]

    with DuckDBManager(db_path) as db:
        filtered = db.con.execute("""
            WITH valid AS (
                SELECT scientificName FROM crypticbio
                WHERE id = ANY(?) 
                GROUP BY scientificName HAVING COUNT(*) >= 50
            )
            SELECT id, scientificName FROM crypticbio
            WHERE scientificName IN (SELECT scientificName FROM valid)
            AND id = ANY(?)
        """, [ids, ids]).df()

    valid_ids = filtered["id"].tolist()
    species = sorted(filtered["scientificName"].unique().tolist())
    name_to_id = {name: i for i, name in enumerate(species)}
    print(f"Dataset: {len(valid_ids)} samples, {len(species)} species")

    dataset = CrypticBioDataset(
        ids=valid_ids,
        name_to_id=name_to_id,
        cb_folder=cb_folder,
        sh_folder=sh_folder,
    )

    labels_array = [dataset.id_to_label[str(i)] for i in valid_ids]
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels_array, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels_array[i] for i in temp_idx], random_state=42)

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    test_ds  = torch.utils.data.Subset(dataset, test_idx)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = EarlyFusionModel(num_classes=len(name_to_id))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("\n--- quick check ---")
    cb_batch, sh_batch, label_batch = next(iter(loader))
    
    for step in range(10):
        out = model(cb_batch, sh_batch)
        loss = criterion(out, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  step {step+1:2d} | loss: {loss.item():.4f}")

    print(f"\nOutput shape:   {out.shape}")
    print(f"Num classes:    {len(name_to_id)}")
    print(f"Predictions:    {out.argmax(dim=1).tolist()}")
    print(f"Labels:         {label_batch.tolist()}")


if __name__ == "__main__":
    main()