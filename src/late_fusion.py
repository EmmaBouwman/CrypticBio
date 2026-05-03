import torch
import torch.nn as nn
import torchvision.models as models

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

        #features = torch.cat([feat_cb, feat_sh], dim=1)             
        
        self.fusion = nn.Linear(1024, 512)
        features = self.fusion(torch.cat([feat_cb, feat_sh], dim=1))
        
        return self.classifier(features)

