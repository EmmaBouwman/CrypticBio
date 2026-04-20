import torch
import torch.nn as nn
import torchvision.models as models

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cb_encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.sh_encoder = models.resnet18(weights="IMAGENET1K_V1")

        self.cb_encoder.fc = nn.Identity()
        self.sh_encoder.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, cb, sh):
        cb_feat = self.cb_encoder(cb)
        sh_feat = self.sh_encoder(sh)

        fused = torch.cat([cb_feat, sh_feat], dim=1)
        return self.classifier(fused)
    

model = EarlyFusionModel(num_classes=10)

cb = torch.randn(4, 3, 224, 224)
sh = torch.randn(4, 3, 224, 224)

out = model(cb, sh)

print(out.shape)