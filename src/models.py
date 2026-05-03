import torch
import torch.nn as nn
import pandas as pd
import timm

from tqdm import tqdm
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
from enum import Enum

class ModelType(Enum):
    Animal = 1
    Satelite = 2
    Both = 3
    Early = 4
    Late = 5

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

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', freeze_backbone=False):
        super().__init__()

        
        self.channel_proj = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.num_features 

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, cb_img, sh_img):
        x = torch.cat([cb_img, sh_img], dim=1)   
        x = self.channel_proj(x)                  
        features = self.backbone(x)               
        return self.classifier(features)

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()

        self.cb_encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.sh_encoder = models.resnet18(weights="IMAGENET1K_V1")

        self.cb_encoder.fc = nn.Identity()
        self.sh_encoder.fc = nn.Identity()

        embed_dim = 512
        fusion_dim = embed_dim * 2

        if freeze_backbone:
            for param in self.cb_encoder.parameters():
                param.requires_grad = False
            for param in self.sh_encoder.parameters():
                param.requires_grad = False
       	
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, cb, sh):
        feat_cb = self.cb_encoder(cb)
        feat_sh = self.sh_encoder(sh)

        features = torch.cat([feat_cb, feat_sh], dim=1)             
        
        return self.classifier(features)

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

def train_epoch_single_modality(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for image, labels in tqdm(loader, desc="Training"):
        image, labels = image.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(image)
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
        for animal, sat, labels in loader:
            animal, sat, labels = animal.to(device), sat.to(device), labels.to(device)
            outputs = model(animal, sat)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), 100. * correct / total

def evaluate_single_modality(model, loader, criterion, device):
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