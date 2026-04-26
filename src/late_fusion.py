import torch
import torch.nn as nn
import timm


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        num_classes,
        bird_model_name='convnext_base',
        sat_model_name='efficientnet_b4'
    ):
        super().__init__()

        # Backbones
        self.cb_backbone = timm.create_model(
            bird_model_name, pretrained=True, num_classes=0
        )

        self.sh_backbone = timm.create_model(
            sat_model_name, pretrained=True, num_classes=0
        )

        # Feature dims
        cb_dim = self.cb_backbone.num_features
        sh_dim = self.sh_backbone.num_features

        # Project to same size
        common_dim = 512
        self.cb_proj = nn.Linear(cb_dim, common_dim)
        self.sh_proj = nn.Linear(sh_dim, common_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(common_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, cb_images, sh_images):
        cb_features = self.cb_backbone(cb_images)
        sh_features = self.sh_backbone(sh_images)

        cb_features = self.cb_proj(cb_features)
        sh_features = self.sh_proj(sh_features)

        fused = torch.cat((cb_features, sh_features), dim=1)

        return self.classifier(fused)