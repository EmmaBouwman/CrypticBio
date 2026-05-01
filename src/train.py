import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from early_fusion import EarlyFusionModel
from model_transformer import get_transforms
from src.data_gather import DuckDBManager
from multimodal_dataset import CrypticBioDataset, TransformDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dotenv import load_dotenv


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for cb, sh, labels in tqdm(loader, desc="Training"):
        cb, sh, labels = cb.to(device), sh.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(cb, sh)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for cb, sh, labels in tqdm(loader, desc="Evaluating"):
            cb, sh, labels = cb.to(device), sh.to(device), labels.to(device)
            outputs = model(cb, sh)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


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
        db_path=db_path
    )

    labels_array = [dataset.id_to_label[str(i)] for i in valid_ids]
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels_array, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels_array[i] for i in temp_idx], random_state=42)

    data_transforms = get_transforms(transform_size=-1, normalize=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    train_ds = TransformDataset(torch.utils.data.Subset(dataset, train_idx), transform=data_transforms['train'])
    val_ds   = TransformDataset(torch.utils.data.Subset(dataset, val_idx),   transform=data_transforms['val'])
    test_ds  = TransformDataset(torch.utils.data.Subset(dataset, test_idx),  transform=data_transforms['test'])

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}") 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EarlyFusionModel(num_classes=len(name_to_id), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.channel_proj.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(),   'lr': 1e-4},
    ], weight_decay=0.01)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    early_stop = False

    for epoch in range(20): 
        if early_stop:
            print("Early stopping.")
            break

        print(f"\nEpoch {epoch+1}/20")

        if epoch == 5:
            print("Unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(),     'lr': 1e-6},
                {'params': model.channel_proj.parameters(), 'lr': 1e-4},
                {'params': model.classifier.parameters(),   'lr': 1e-4},
            ], weight_decay=0.01)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
            }, "best_early_fusion.pth")
            print(f"New best val acc: {best_val_acc:.2f}% — model saved.")

        if epochs_no_improve >= patience:
            early_stop = True

if __name__ == "__main__":
    main()