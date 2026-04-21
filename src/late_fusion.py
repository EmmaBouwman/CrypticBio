import torch
import torch.nn as nn
import torchvision.models as models

class LateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionModel, self).__init__()

        # Backbone for cb_images
        self.cb_backbone = models.resnet18(pretrained=True)
        self.cb_backbone.fc = nn.Identity()  

        # Backbone for sh_images
        self.sh_backbone = models.resnet18(pretrained=True)
        self.sh_backbone.fc = nn.Identity()

        fusion_dim = 512 * 2

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, cb_images, sh_images):
        cb_features = self.cb_backbone(cb_images)
        sh_features = self.sh_backbone(sh_images)

        
        fused = torch.cat((cb_features, sh_features), dim=1)  # Concatenate features

        out = self.classifier(fused)
        return out