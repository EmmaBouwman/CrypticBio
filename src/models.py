import torch
import torch.nn as nn
import pandas as pd
import timm

from tqdm import tqdm
from src.data_gather import DuckDBManager
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
from enum import Enum

class ModelType(Enum):
    """Enum representing the available model/fusion strategies."""
    Animal = 1
    Satelite = 2
    Both = 3
    Early = 4
    Late = 5
    Gated = 6

class AnimalSatClassifier(nn.Module):
    """
    Dual-backbone classifier that fuses animal and satellite imagery via cross-attention.

    The animal backbone's patch tokens act as queries against the satellite
    backbone's features, which allows the model to attend to geographically
    relevant regions before classification.

    Args:
        num_classes (int): Number of output classes.
        model_name (str): timm backbone identifier (default: vit_tiny_patch16_224).
        freeze_backbone (bool): If True, backbone weights are frozen at init.
    """
    def __init__(self, num_classes, model_name='vit_tiny_patch16_224', freeze_backbone=True):
        super().__init__()
        
        # Separate backbones so each modality learns its own representations
        self.animal_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.sat_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.animal_backbone.parameters():
                param.requires_grad = False
            for param in self.sat_backbone.parameters():
                param.requires_grad = False

        embed_dim = self.animal_backbone.num_features

        # Cross-attention: animal patch tokens query the satellite feature map
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim*2),                  # Stability for fused features
            nn.Linear(embed_dim*2, embed_dim),          # Bottleneck: 384 -> 192
            nn.GELU(),                                  # Transformer-standard activation
            nn.Dropout(0.25),                           # Light regularization
            nn.LayerNorm(embed_dim),                    # Norm the hidden layer
            nn.Dropout(0.5),                            # Heavier regularization before output
            nn.Linear(embed_dim, num_classes)           # Final prediction
        )

    def forward(self, animal_img, sat_img, attn_bool: bool = False):
        """
        Args:
            animal_img (Tensor): Batch of animal images [B, C, H, W].
            sat_img (Tensor): Batch of satellite images [B, C, H, W].
            attn_bool (bool): If True, also return the cross-attention weights.

        Returns:
            logits (Tensor): Class scores [B, num_classes].
            attn_weights (Tensor, optional): Attention weights, only when attn_bool=True.
        """
        b_features = self.animal_backbone.forward_features(animal_img)
        s_features = self.sat_backbone.forward_features(sat_img)
        
        # Split CLS token from patch tokens
        animal_full = b_features[:, 0, :] # CLS token [B, embed_dim]
        animal_queries = b_features[:, 1:, :] # Patch token [B, N, embed_dim]

        # Let animal patches attend over satellite features
        attn_output, attn_weights = self.cross_attn(
            query=animal_queries, 
            key=s_features, 
            value=s_features,
            need_weights=True
        )

        # Global average pool the attended satellite features -> [B, embed_dim]
        sat_adjusted = attn_output.mean(dim=1)    

        # Concatenate CLS token with attended satellite representation -> [B, embed_dim*2]
        combined = torch.cat([animal_full, sat_adjusted], dim=-1) 

        
        logits = self.classifier(combined)

        if attn_bool:
            return logits, attn_weights
        
        return logits

class SingleModalityClassifier(nn.Module):
    def __init__(self, num_classes, model_name='vit_tiny_patch16_224', freeze_backbone=True):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),                 # Standard for Transformers
            nn.Linear(embed_dim, embed_dim // 2),    # Bottleneck: 192 -> 96
            nn.GELU(),                               # Transformer-style activation
            nn.Dropout(0.25),                        # Initial regularization
            nn.LayerNorm(embed_dim // 2),            # Stability for the hidden layer
            nn.Dropout(0.5),                         # Heavier dropout before output
            nn.Linear(embed_dim // 2, num_classes)   # Final prediction
        )

    def forward(self, img):
        """
        Args:
            img (Tensor): Batch of images [B, C, H, W].

        Returns:
            Tensor: Class scores [B, num_classes].
        """
        features = self.backbone.forward_features(img)
        # Use only the CLS token for classification
        return self.classifier(features[:, 0, :])

class EarlyFusionModel(nn.Module):
    """
    Multimodal classifier using early fusion.

    Both images are channel-concatenated into a 6-channel input before
    being passed through a single shared backbone.

    Args:
        num_classes (int): Number of output classes.
        model_name (str): timm backbone identifier (resnet18 or resnet50).
        freeze_backbone (bool): If True, backbone weights are frozen at init.
    """
    def __init__(self, num_classes, model_name='resnet50', freeze_backbone=False):
        super().__init__()
        
        # in_chans=6 accepts the two RGB images stacked along the channel dimension
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, in_chans=6)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.num_features 

        self.classifier = nn.Sequential(           
            nn.Flatten(),                           # [B, embed_dim]
            nn.BatchNorm1d(embed_dim),              # Stability before Linear
            nn.Dropout(0.25),                       # Light Dropout
            nn.Linear(embed_dim, embed_dim // 2),   # Bottleneck
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim // 2),         # Stability after Activation
            nn.Dropout(0.5),                        # Heavier Dropout before output
            nn.Linear(embed_dim // 2, num_classes),
        )


    def forward(self, cb_img, sh_img):
        """
        Args:
            cb_img (Tensor): CrypticBio images [B, 3, H, W].
            sh_img (Tensor): Sentinel images [B, 3, H, W].

        Returns:
            Tensor: Class scores [B, num_classes].
        """
        # Stack both modalities into a single 6-channel tensor
        x = torch.cat([cb_img, sh_img], dim=1)                     
        features = self.backbone(x)               
        return self.classifier(features)

class LateFusionModel(nn.Module):
    """
    Multimodal classifier using late fusion or gated fusion.

    Each modality is encoded by its own ResNet backbone. The resulting
    embeddings are then either concatenated (Late) or combined via a
    learned gate (Gated) before classification.

    Args:
        num_classes (int): Number of output classes.
        model_name (str): Backbone architecture ('resnet18' or 'resnet50').
        freeze_backbone (bool): If True, backbone weights are frozen at init.
        model_type (ModelType): Fusion strategy — ModelType.Late or ModelType.Gated.
    """
    def __init__(self, num_classes, model_name, freeze_backbone=False, model_type=ModelType.Late):
        super().__init__()
        print(model_type)
        self.model_type = ModelType(model_type)

        if model_name == "resnet18":
            self.cb_encoder = models.resnet18(weights="IMAGENET1K_V1")
            self.sh_encoder = models.resnet18(weights="IMAGENET1K_V1")
            embed_dim = 512

        elif model_name == "resnet50":
            self.cb_encoder = models.resnet50(weights="IMAGENET1K_V1")
            self.sh_encoder = models.resnet50(weights="IMAGENET1K_V1")
            embed_dim = 2048

        # Remove the original classification heads; we only need the feature vectors
        self.cb_encoder.fc = nn.Identity()
        self.sh_encoder.fc = nn.Identity()

        fusion_dim = embed_dim * 2 # Dimensionality after concatenating both embeddings       

        if self.model_type == ModelType.Gated:
             # Gate projects the concatenated features back to embed_dim
            self.gate_layer = nn.Linear(fusion_dim, embed_dim)

        # Wrap encoders so freeze_backbone can iterate over both
        self.backbone = nn.ModuleList([self.cb_encoder, self.sh_encoder])

        if freeze_backbone:
            for module in self.backbone:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Classifier input size depends on the fusion strategy
        if model_type == ModelType.Late:
            in_dim = fusion_dim   # Concatenated embeddings
            hidden_dim = embed_dim
        elif model_type == ModelType.Gated:
            in_dim = embed_dim    # Gate reduces back to embed_dim
            hidden_dim = embed_dim
        else:
            raise ValueError(f"Unsupported model_type: {model_type}") 

        self.classifier = nn.Sequential(           # Result: (Batch, 4096, 1, 1)
            nn.Flatten(),                          # Result: (Batch, 4096)
            nn.BatchNorm1d(in_dim),         # Stability before Linear
            nn.Dropout(0.25),                      # Light Dropout
            nn.Linear(in_dim, hidden_dim), # 4096 -> 2048 (The Bottleneck)
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),      # Stability after Activation
            nn.Dropout(0.5),                       # Heavier Dropout before output
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cb, sh):
        """
        Args:
            cb (Tensor): CrypticBio images [B, 3, H, W].
            sh (Tensor): Sentinel images [B, 3, H, W].

        Returns:
            Tensor: Class scores [B, num_classes].
        """
        feat_cb = self.cb_encoder(cb)
        feat_sh = self.sh_encoder(sh)

        if self.model_type == ModelType.Late:
             # Concatenate both feature vectors
            features = torch.cat([feat_cb, feat_sh], dim=1)
            
        elif self.model_type == ModelType.Gated:
            # Learn a soft gate that combines the two embeddings
            gate_input = torch.cat([feat_cb, feat_sh], dim=1)
            gate = torch.sigmoid(self.gate_layer(gate_input))
            features = gate * feat_cb + (1 - gate) * feat_sh

        return self.classifier(features)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full training epoch over a dual-modality data loader.

    Args:
        model (nn.Module): Model that accepts (cb, sh, labels) inputs.
        loader (DataLoader): Yields (cb_img, sh_img, labels) batches.
        optimizer (Optimizer): Parameter optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Target device.

    Returns:
        avg_loss (float): Mean loss over all batches.
        accuracy (float): Top-1 accuracy in percent.
        f1 (float): Weighted F1 score.
        recall (float): Weighted recall.
        precision (float): Weighted precision.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

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

        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1, recall, precision

def train_epoch_single_modality(model, loader, optimizer, criterion, device):
    """
    Run one full training epoch over a single-modality data loader.

    Args:
        model (nn.Module): Model that accepts a single image tensor.
        loader (DataLoader): Yields (image, labels) batches.
        optimizer (Optimizer): Parameter optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Target device.

    Returns:
        avg_loss (float): Mean loss over all batches.
        accuracy (float): Top-1 accuracy in percent.
        f1 (float): Weighted F1 score.
        recall (float): Weighted recall.
        precision (float): Weighted precision.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

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

        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1, recall, precision

def evaluate(model, loader, criterion, device):
    """
    Evaluate a dual-modality model on a validation or test set.

    Args:
        model (nn.Module): Model that accepts (animal, sat) inputs.
        loader (DataLoader): Yields (animal_img, sat_img, labels) batches.
        criterion (Loss): Loss function.
        device (torch.device): Target device.

    Returns:
        avg_loss (float): Mean loss over all batches.
        accuracy (float): Top-1 accuracy in percent.
        f1 (float): Weighted F1 score.
        recall (float): Weighted recall.
        precision (float): Weighted precision.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for animal, sat, labels in loader:
            animal, sat, labels = animal.to(device), sat.to(device), labels.to(device)
            outputs = model(animal, sat)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1, recall, precision
            

def evaluate_single_modality(model, loader, criterion, device):
    """
    Evaluate a single-modality model on a validation or test set.

    Args:
        model (nn.Module): Model that accepts a single image tensor.
        loader (DataLoader): Yields (image, labels) batches.
        criterion (Loss): Loss function.
        device (torch.device): Target device.

    Returns:
        avg_loss (float): Mean loss over all batches.
        accuracy (float): Top-1 accuracy in percent.
        f1 (float): Weighted F1 score.
        recall (float): Weighted recall.
        precision (float): Weighted precision.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for image, labels in loader:
            image, labels = image.to(device), labels.to(device)
            outputs = model(image)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1, recall, precision