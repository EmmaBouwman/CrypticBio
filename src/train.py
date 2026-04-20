import torch
import torch.nn as nn
from early_fusion import EarlyFusionModel

model = EarlyFusionModel(num_classes=10)

cb = torch.randn(4, 3, 224, 224)
sh = torch.randn(4, 3, 224, 224)
labels = torch.randint(0, 10, (4,))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

out = model(cb, sh)
loss = criterion(out, labels)

loss.backward()
optimizer.step()

print("loss:", loss.item())