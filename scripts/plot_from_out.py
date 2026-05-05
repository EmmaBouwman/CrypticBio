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

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

for out_file in sys.argv[1:]:
    epochs, train_loss, val_loss, train_acc, val_acc = parse_out_file(out_file)

    if not epochs:
        print(f"Skipping {out_file}: no data found")
        continue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train")
    ax1.plot(epochs, val_loss, label="Val")
    ax1.axvline(x=5, color="gray", linestyle="--", linewidth=1.0, label="backbone unfrozen")
    ax1.legend()
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")        
    ax1.set_ylabel("Loss")        


    ax2.plot(epochs, train_acc, label="Train")
    ax2.plot(epochs, val_acc, label="Val")
    best_val = max(val_acc)
    best_ep  = epochs[val_acc.index(best_val)]

    ax2.scatter([best_ep], [best_val], color="red", zorder=5)
    ax2.annotate(f"best: {best_val:.1f}%\n(epoch {best_ep})",
             xy=(best_ep, best_val),
             xytext=(best_ep + 2, best_val - 5),
             arrowprops=dict(arrowstyle="->", color="red"),
             color="red", fontsize=8)
    ax2.axvline(x=5, color="gray", linestyle="--", linewidth=1.0, label="backbone unfrozen")
    ax2.legend()
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")       
    ax2.set_ylabel("Accuracy (%)")

    name = Path(out_file).stem
    plt.savefig(output_dir / f"{name}.png")
    plt.close()
