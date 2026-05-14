import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Global plot style
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

def parse_out_file(path):
    """
    Parse a training log file and extract per-epoch metrics.

    Args
        path : Path to the .out log file.

    Returns
        epochs (list[int]): Epoch numbers.
        train_loss (list[float]): Training loss per epoch.
        val_loss (list[float]): Validation loss per epoch.
        train_acc (list[float]): Training accuracy (%) per epoch.
        val_acc (list[float]): Validation accuracy (%) per epoch.
    """
    epochs, train_loss, val_loss = [], [], []
    train_acc, val_acc = [], []

    # Matches one epoch block, even when newlines appear between fields 
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

# Ensure the output directory exists before saving any plots
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Process each log file passed as a command-line argument
for out_file in sys.argv[1:]:
    epochs, train_loss, val_loss, train_acc, val_acc = parse_out_file(out_file)

    if not epochs:
        print(f"Skipping {out_file}: no data found")
        continue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # -- Loss plot --
    ax1.plot(epochs, train_loss, label="Train")
    ax1.plot(epochs, val_loss, label="Val")
    ax1.axvline(x=5, color="gray", linestyle="--", linewidth=1.0, label="backbone unfrozen")
    ax1.legend()
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")        
    ax1.set_ylabel("Loss")        

    # -- Accuracy plot --
    ax2.plot(epochs, train_acc, label="Train")
    ax2.plot(epochs, val_acc, label="Val")

    # Highlight the best validation point with a red dot and annotation
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

    # Use the filename as the figure title and output filename
    name = Path(out_file).stem
    fig.suptitle(name, fontsize=11, fontweight="bold")
    plt.savefig(output_dir / f"{name}.png")
    plt.close()
