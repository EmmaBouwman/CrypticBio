import os 
import torch
import argparse
from pathlib import Path
from dotenv import load_dotenv
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.model_transformer import ( 
    AnimalSatClassifier, 
    AnimalSateliteDataset, 
    SingleModalityClassifier,
    ModelType,
    get_transforms, 
    train_epoch, 
    train_epoch_single_modality,
    bs_check, 
    bs_check_single_modality,
    Transform,
    TransformSingleModality
)
from src.data_gather import DuckDBManager

def parse_args():
    parser = argparse.ArgumentParser(description="Train Animal Sat Transformer Classifier")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr_backbone", type=float, default=1e-6, help="Learning rate for ViT backbones")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for attention and classifier")
    parser.add_argument("--min_samples", type=int, default=50, help="Minimum samples per species to include")
    parser.add_argument("--model_type", type=int, default=3, 
                        help="Model to be used, 1 = Only Animal image, 2 = Only Satelite image, 3 = Cross attention (both images)")
    
    # Model Setup
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="Timm model string")
    parser.add_argument("--save_name", type=str, default="best_animal_sat_model.pth", help="Filename for best checkpoint")
    parser.add_argument("--transform_size", type=int, default=-1, 
                        help="A number similar to the folder you created to resize an image with")

    parser.add_argument("--test_only", action="store_true", help="Whether to only test a saved model")
    
    # Hardware/Environment
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()

    # Setup Paths
    base_path = Path(os.getenv("DATA_FOLDER"))
    db_path = base_path / os.getenv("DATABASE")
    cb_folder = base_path / os.getenv("CB_IMAGE_PATH")

    # Load IDs from folder
    all_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(cb_folder) if f.endswith('.png')]

    print(f"Filtering species with at least {args.min_samples} samples...")
    with DuckDBManager(db_path) as db:
        filtered_data = db.con.execute("""
            WITH valid_species AS (
                SELECT scientificName
                FROM crypticbio 
                WHERE rowid = ANY(?) 
                GROUP BY scientificName
                HAVING COUNT(*) >= ?
            )
            SELECT rowid, scientificName 
            FROM crypticbio 
            WHERE scientificName IN (SELECT scientificName FROM valid_species)
            AND rowid = ANY(?)
        """, [all_ids, args.min_samples, all_ids]).df()

    species_list = sorted(filtered_data['scientificName'].unique().tolist())
    valid_ids = filtered_data['rowid'].tolist()

    name_to_id = {name: idx for idx, name in enumerate(species_list)}
    id_to_name = {idx: name for idx, name in enumerate(species_list)}
    num_classes = len(species_list)
    print(f"Dataset ready: {len(valid_ids)} samples across {num_classes} classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = ModelType(args.model_type)

    dataset = AnimalSateliteDataset(valid_ids, name_to_id, db_path, args.transform_size, model_type)
    labels = dataset.data['scientificName'].values
    
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)

    data_transforms = get_transforms(args.transform_size)

    if model_type == ModelType.Both:
        model = AnimalSatClassifier(num_classes=num_classes, model_name=args.model_name).to(device)
        if not args.test_only:
            train_ds = Transform(torch.utils.data.Subset(dataset, train_idx), transform=data_transforms['train'])
            val_ds = Transform(torch.utils.data.Subset(dataset, val_idx), transform=data_transforms['val'])
        test_ds = Transform(torch.utils.data.Subset(dataset, test_idx), transform=data_transforms['test'])

        optimizer = torch.optim.AdamW([
            {'params': model.cross_attn.parameters(), 'lr': args.lr_head},
            {'params': model.classifier.parameters(), 'lr': args.lr_head},
        ], weight_decay=0.01)

    else:
        model = SingleModalityClassifier(num_classes=num_classes, model_name=args.model_name).to(device)
        if not args.test_only:
            train_ds = TransformSingleModality(torch.utils.data.Subset(dataset, train_idx), transform=data_transforms['train'])
            val_ds = TransformSingleModality(torch.utils.data.Subset(dataset, val_idx), transform=data_transforms['val'])
        test_ds = TransformSingleModality(torch.utils.data.Subset(dataset, test_idx), transform=data_transforms['test'])

        optimizer = torch.optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': args.lr_head},
        ], weight_decay=0.01)
    
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    if not args.test_only:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        patience = 5
        epochs_no_improve = 0
        best_val_loss = float('inf')
        early_stop = False

        best_val_acc = 0.0
        for epoch in range(args.epochs):
            if early_stop:
                print("Stopping early due to lack of improvement.")
                break

            print(f"\nEpoch {epoch+1}/{args.epochs}")

            if epoch == 5:
                if model_type == ModelType.Both:
                    for param in model.animal_backbone.parameters():
                        param.requires_grad = True

                    for param in model.sat_backbone.parameters():
                        param.requires_grad = True

                    optimizer = torch.optim.AdamW([
                        {'params': model.animal_backbone.parameters(), 'lr': args.lr_backbone}, # Very low LR for fine-tuning
                        {'params': model.sat_backbone.parameters(), 'lr': args.lr_backbone},
                        {'params': model.cross_attn.parameters(), 'lr': args.lr_head},
                        {'params': model.classifier.parameters(), 'lr': args.lr_head},
                    ], weight_decay=0.01)
                else:
                    for param in model.backbone.parameters():
                        param.requires_grad = True

                    optimizer = torch.optim.AdamW([
                        {'params': model.backbone.parameters(), 'lr': args.lr_backbone}, # Very low LR for fine-tuning
                        {'params': model.classifier.parameters(), 'lr': args.lr_head}
                    ], weight_decay=0.01)

            if model_type == ModelType.Both:
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = bs_check(model, val_loader, criterion, device)
            else:
                train_loss = train_epoch_single_modality(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = bs_check_single_modality(model, val_loader, criterion, device)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement in Val Loss for {epochs_no_improve} epochs.")

            # 2. Save Best Model (based on Accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'species_map': id_to_name
                }, args.save_name)
                print(f"New best accuracy! Model saved to {args.save_name}")

            # 3. Trigger Early Stop
            if epochs_no_improve >= patience and epoch >= 5:
                early_stop = True

    try:
        checkpoint = torch.load(args.save_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"Error: Could not find {args.save_name}. Did you train the model already?")
        return

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if model_type == ModelType.Both:
        _, test_acc = bs_check(model, test_loader, criterion, device)
    else:
        _, test_acc = bs_check_single_modality(model, test_loader, criterion, device)
    print(f"\n🚀 Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()