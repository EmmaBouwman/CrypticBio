import os 
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from src.model_transformer import BirdSatClassifier, BirdSateliteDataset, data_transforms, train_epoch, bs_check

load_dotenv()

base_path = Path(os.getenv("DATA_FOLDER"))
db_path = base_path / os.getenv("DATABASE")
cb_folder = base_path / os.getenv("CB_IMAGE_PATH")

all_ids = [os.path.splitext(f)[0] for f in os.listdir(cb_folder) if f.endswith('.png')]

train_ids, temp_ids = train_test_split(all_ids, test_size=0.20, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

with DuckDBManager(db_path) as db:
    all_species = db.con.execute(
        "SELECT DISTINCT scientific_name FROM crypticbio ORDER BY scientific_name"
    ).df()
    species_list = all_species['scientific_name'].tolist()

self.name_to_id = {}
self.id_to_name = {}
for idx, name in enumerate(species_list):
    self.name_to_id[name] = idx
    self.id_to_name[idx] = name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(species_list)
model = BirdSatClassifier(num_classes=num_classes).to(device)

train_ds = BirdSateliteDataset(train_ids, name_to_id, transform=data_transforms['train'])
val_ds = BirdSateliteDataset(val_ids, name_to_id, transform=data_transforms['val'])
test_ds = BirdSateliteDataset(test_ids, name_to_id, transform=data_transforms['test'])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

optimizer = torch.optim.AdamW([
    {'params': model.bird_backbone.parameters(), 'lr': 1e-5},
    {'params': model.sat_backbone.parameters(), 'lr': 1e-5},
    {'params': model.cross_attn.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
], weight_decay=0.01)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


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

test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"\n🚀 Final Test Accuracy: {test_acc:.2f}%")