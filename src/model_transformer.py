import torch
import torch.nn as nn
import pandas as pd
import timm

from tqdm import tqdm
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

class Transform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        bird_img, sat_img, label = self.subset[index]
        if self.transform:
            bird_img = self.transform(bird_img)
            sat_img = self.transform(sat_img)
        return bird_img, sat_img, label

class BirdSateliteDataset(Dataset):
    def __init__(self, row_ids, name_to_id, db_path):
        self.name_to_id = name_to_id

        with DuckDBManager(db_path) as db:
            self.data = db.con.execute(
                "SELECT * FROM crypticbio WHERE id = ANY(?)", [row_ids]
            ).df()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        bird_path = row['crypticbio_image']
        bird_img = Image.open(bird_path).convert('RGB')

        sat_path = row['sentinel_image']
        sat_img = Image.open(sat_path).convert('RGB')

        label = self.name_to_id[row['scientificName']]

        return bird_img, sat_img, torch.tensor(label, dtype=torch.long)

class BirdSatClassifier(nn.Module):
    def __init__(self, num_classes, model_name='vit_tiny_patch16_224', freeze_backbone=True):
        super().__init__()
        
        # 1. Backbones for feature extraction
        self.bird_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.sat_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.bird_backbone.parameters():
                param.requires_grad = False
            for param in self.sat_backbone.parameters():
                param.requires_grad = False

        embed_dim = self.bird_backbone.num_features

        # 2. Cross-Attention: Bird patches query the Satellite features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        # 3. Final Classification Head
        # Concatenates: [Bird tokens] + [Satelite with bird tokens]
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, bird_img, sat_img):
        # Feature Extraction
        b_features = self.bird_backbone.forward_features(bird_img)
        s_features = self.sat_backbone.forward_features(sat_img)

        bird_full = b_features[:, 0, :]
        bird_queries = b_features[:, 1:, :]

        # Cross-Attention: Using the bird queries to find relevant info in the satellite map
        attn_output, _ = self.cross_attn(
            query=bird_queries, 
            key=s_features, 
            value=s_features
        )

        # Global Average Pooling
        sat_adjusted = attn_output.mean(dim=1)    # [Batch, 768]

        # Combine bird and sat with bird queries
        combined = torch.cat([bird_full, sat_adjusted], dim=-1) # [Batch, 1536]

        # Final Scientific label Prediction
        return self.classifier(combined)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for bird, sat, labels in tqdm(loader, desc="Training"):
        bird, sat, labels = bird.to(device), sat.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(bird, sat)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def bs_check(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for bird, sat, labels in loader:
            bird, sat, labels = bird.to(device), sat.to(device), labels.to(device)
            outputs = model(bird, sat)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), 100. * correct / total