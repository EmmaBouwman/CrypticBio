import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_out_file(path):
    epochs, train_loss, val_loss = [], [], []
    train_acc, val_acc = [], []

    pattern = re.compile(
        r"Epoch (\d+)/\d+.*?"
        r"Train Loss:\s*([\d.]+) \| Train Acc:\s*([\d.]+)%.*?"
        r"Val Loss:\s*([\d.]+) \| Val Acc:\s*([\d.]+)%",
        re.DOTALL 
    )

    with open(path) as f:
        content = f.read()

    for m in pattern.finditer(content):
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        train_acc.append(float(m.group(3)))
        val_loss.append(float(m.group(4)))
        val_acc.append(float(m.group(5)))

    return epochs, train_loss, val_loss, train_acc, val_acc

epochs, train_loss, val_loss, train_acc, val_acc = parse_out_file("./logs/job_early_fusion_model_2076542.out")
print(epochs[:5])       
print(train_acc[:5])    

