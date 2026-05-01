import os
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from torchvision import transforms

def transform_img(img, size):
    preprocess_transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
    ])
    return preprocess_transform(img)

def resize_image(task, img_size):
    src, dst = task
    try:
        # Create destination folder if it doesn't exist
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        with Image.open(src) as img:
            # Convert to RGB to strip alpha channels/CMYK before resizing
            img = img.convert('RGB')
            resized_img = transform_img(img, img_size)
            resized_img.save(dst, "PNG")
    except Exception as e:
        print(f"Error processing {src}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Resize downloaded images in the folder to a specific size")
    
    # Training Hyperparameters
    parser.add_argument("--size", type=int, default=16, help="Size of the image (Rectangle only, so 1 side)")
    parser.add_argument("--source", type=str, default=20, help="Path where the images are stored")
    
    return parser.parse_args()

def main():
    args = parse_args()
    tasks_config = [
        {"src": args.source, "dst": f"{args.source}_{args.size}"}
    ]
    
    all_tasks = []

    # 3. Build the task list
    print("Scanning folders for images...")
    for config in tasks_config:
        src_root = Path(config["src"])
        dst_root = Path(config["dst"])
        
        if not src_root.exists():
            print(f"Warning: Source folder {src_root} not found. Skipping.")
            continue

        for f in os.listdir(src_root):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                src_path = src_root / f
                dst_path = dst_root / f
                all_tasks.append((src_path, dst_path))

    print(f"Total images to process: {len(all_tasks)}")

    # 4. Run in Parallel
    # Using 16 workers as requested for your Slurm job
    if all_tasks:
        print(f"Starting parallel resize (16 workers)...")
        with ProcessPoolExecutor(max_workers=16) as executor:
            # list() forces the generator to evaluate, which triggers the tqdm bar
            list(tqdm(executor.map(resize_image, all_tasks), total=len(all_tasks)))
        print("\nPreprocessing complete! Check your '_224' folders.")
    else:
        print("No images found to process.")

if __name__ == "__main__":
    main()