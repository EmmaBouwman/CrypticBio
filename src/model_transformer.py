import torch
import torch.nn as nn
import pandas as pd
import timm

from tqdm import tqdm
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from enum import Enum

def get_transforms(transform_size, mean, std):
    common_post_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    resize_transform = [transforms.Resize((224, 224))] if transform_size == -1 else []

    return {
        'train': transforms.Compose(
            resize_transform + 
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)] + 
            common_post_transforms
        ),
        'val': transforms.Compose(
            resize_transform + 
            common_post_transforms
        ),
        'test': transforms.Compose(
            resize_transform + 
            common_post_transforms
        )
    }

class ModelType(Enum):
    Animal = 1
    Satelite = 2
    Both = 3

class Transform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        animal_img, sat_img, label = self.subset[index]
        if self.transform:
            animal_img = self.transform(animal_img)
            sat_img = self.transform(sat_img)
        return animal_img, sat_img, label

class TransformSingleModality(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

class AnimalSateliteDataset(Dataset):
    def __init__(self, row_ids, name_to_id, db_path, transform_size: int = -1, model_type: ModelType = ModelType.Both):
        self.name_to_id = name_to_id
        self.transform_size = transform_size
        self.model_type = model_type

        with DuckDBManager(db_path) as db:
            self.data = db.con.execute(
                "SELECT * FROM crypticbio WHERE id = ANY(?)", [row_ids]
            ).df()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.transform_size != -1:
            animal_path = row['crypticbio_image'].replace("images_cb", f"images_cb_{self.transform_size}")
            sat_path = row['sentinel_image'].replace("images_sh", f"images_sh_{self.transform_size}")
        else:
            animal_path = row['crypticbio_image']
            sat_path = row['sentinel_image']
        
        label = self.name_to_id[row['scientificName']]

        if self.model_type == ModelType.Animal:
            animal_img = Image.open(animal_path).convert('RGB')
            return animal_img, torch.tensor(label, dtype=torch.long)
        elif self.model_type == ModelType.Satelite:
            sat_img = Image.open(sat_path).convert('RGB')
            return sat_img, torch.tensor(label, dtype=torch.long)
        else:
            animal_img = Image.open(animal_path).convert('RGB')
            sat_img = Image.open(sat_path).convert('RGB')
            return animal_img, sat_img, torch.tensor(label, dtype=torch.long)

class AnimalSatClassifier(nn.Module):
    def __init__(self, num_classes, model_name='vit_tiny_patch16_224', freeze_backbone=True):
        super().__init__()
        
        # 1. Backbones for feature extraction
        self.animal_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.sat_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.animal_backbone.parameters():
                param.requires_grad = False
            for param in self.sat_backbone.parameters():
                param.requires_grad = False

        embed_dim = self.animal_backbone.num_features

        # 2. Cross-Attention: animal patches query the Satellite features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        # Add a LayerNorm for the combined features (1536 dim)
        self.norm = nn.LayerNorm(embed_dim * 2)

        # 4. Final Classification Head
        # Concatenates: [animal tokens] + [Satelite with animal tokens]
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, animal_img, sat_img):
        # Feature Extraction
        b_features = self.animal_backbone.forward_features(animal_img)
        s_features = self.sat_backbone.forward_features(sat_img)

        animal_full = b_features[:, 0, :]
        animal_queries = b_features[:, 1:, :]

        # Cross-Attention: Using the animal queries to find relevant info in the satellite map
        attn_output, _ = self.cross_attn(
            query=animal_queries, 
            key=s_features, 
            value=s_features
        )

        # Global Average Pooling
        sat_adjusted = attn_output.mean(dim=1)    # [Batch, 768]

        # Combine animal and sat with animal queries
        combined = torch.cat([animal_full, sat_adjusted], dim=-1) # [Batch, 1536]

        # Apply Layernorm before classification
        combined = self.norm(combined)

        # Final Scientific label Prediction
        return self.classifier(combined)

class SingleModalityClassifier(nn.Module):
    def __init__(self, num_classes, model_name='vit_tiny_patch16_224', freeze_backbone=True):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img):
        features = self.backbone.forward_features(img)
        
        return self.classifier(features[:, 0, :])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for animal, sat, labels in tqdm(loader, desc="Training"):
        animal, sat, labels = animal.to(device), sat.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(animal, sat)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch_single_modality(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for image, labels in tqdm(loader, desc="Training"):
        image, labels = image.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(image)
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
        for animal, sat, labels in loader:
            animal, sat, labels = animal.to(device), sat.to(device), labels.to(device)
            outputs = model(animal, sat)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), 100. * correct / total

def bs_check_single_modality(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for image, labels in loader:
            image, labels = image.to(device), labels.to(device)
            outputs = model(image)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), 100. * correct / total