import torch
import torch.nn as nn
import timm

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', freeze_backbone=False):
        super().__init__()

        
        self.channel_proj = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.num_features 

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, cb_img, sh_img):
        x = torch.cat([cb_img, sh_img], dim=1)   
        x = self.channel_proj(x)                  

        features = self.backbone.forward_features(x)  
        cls_token = features[:, 0, :]                

        return self.classifier(cls_token)
