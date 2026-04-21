import os 
import torch

from dotenv import load_dotenv
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.model_transformer import BirdSatClassifier, BirdSateliteDataset, data_transforms, train_epoch, bs_check, Transform
from src.data_gather import DuckDBManager
from pathlib import Path

load_dotenv()

base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")
cb_folder = base_path / os.getenv("CB_IMAGE_PATH")

all_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(cb_folder) if f.endswith('.png')]

with DuckDBManager(db_path) as db:
    # 1. Get the species that meet the count requirement
    # 2. ALSO get the actual rowids for those species in one go
    filtered_data = db.con.execute("""
        WITH valid_species AS (
            SELECT scientificName
            FROM crypticbio 
            WHERE rowid = ANY(?) 
            GROUP BY scientificName
            HAVING COUNT(*) >= 50
        )
        SELECT rowid, scientificName 
        FROM crypticbio 
        WHERE scientificName IN (SELECT scientificName FROM valid_species)
        AND rowid = ANY(?)
    """, [all_ids, all_ids]).df()

# Update your lists based on the filtered dataframe
species_list = filtered_data['scientificName'].unique().tolist()
valid_ids = filtered_data['rowid'].tolist()

print(f"Found {len(species_list)} number of labels")
name_to_id = {}
id_to_name = {}
for idx, name in enumerate(species_list):
    name_to_id[name] = idx
    id_to_name[idx] = name
print("Finished label transformation to int")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(species_list)
model = BirdSatClassifier(num_classes=num_classes).to(device)
print("Created the model")

dataset = BirdSateliteDataset(valid_ids, name_to_id, db_path)
print(f"Created the dataset with following dimension {dataset.data.shape}")

labels = dataset.data['scientificName'].values
train_idx, temp_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

temp_labels = labels[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=temp_labels,   
    random_state=42
)
print("Created indexes")

train_sub = torch.utils.data.Subset(dataset, train_idx)
val_sub = torch.utils.data.Subset(dataset, val_idx)
test_sub = torch.utils.data.Subset(dataset, test_idx)
print("Created subsets")

train_ds = Transform(train_sub, transform=data_transforms['train'])
val_ds = Transform(val_sub, transform=data_transforms['val'])
test_ds = Transform(test_sub, transform=data_transforms['test'])
print("Transformed the data")

optimizer = torch.optim.AdamW([
    {'params': model.bird_backbone.parameters(), 'lr': 1e-5},
    {'params': model.sat_backbone.parameters(), 'lr': 1e-5},
    {'params': model.cross_attn.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
], weight_decay=0.01)

criterion = CrossEntropyLoss(label_smoothing=0.1)
print("Optimizer and Criterion ready")

batch_size = 32

train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

num_epochs = 20
best_val_acc = 0.0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = bs_check(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save the "Best" model weights
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'species_map': id_to_name # Save your ID map too!
        }, "best_bird_sat_model.pth")
        print("New model saved!")

checkpoint = torch.load("best_bird_sat_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

test_loader = DataLoader(
    test_ds, 
    batch_size=batch_size, 
    shuffle=False
)

test_loss, test_acc = bs_check(model, test_loader, criterion, device)
print(f"\n🚀 Final Test Accuracy: {test_acc:.2f}%")