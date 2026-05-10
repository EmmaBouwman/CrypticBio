import os
import torch
import numpy as np
import cv2
import argparse
import warnings
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.dataset import AnimalSateliteDataset, get_transforms
from src.models import AnimalSatClassifier, ModelType, SingleModalityClassifier
from src.data_gather import DuckDBManager

# Global variable to store attention weights captured by the forward hook
extracted_attn_weights = None

def get_attention_hook(module, input, output):
    """
    PyTorch forward hook to capture internal attention weights during inference.
    
    Args:
        module: The layer being hooked.
        input: The input tensor to the layer.
        output: The output tensor from the layer.
    """
    global extracted_attn_weights
    # Capturing the attention dropout input
    extracted_attn_weights = input[0].detach()

def process_and_save_attention(attn_map, raw_img, output_path):
    """
    Processes raw attention weights into a visually interpretable heatmap overlay.

    Args:
        attn_map (np.ndarray): The 2D attention map (e.g., 14x14).
        raw_img (np.ndarray): The original image as a float array [0, 1].
        output_path (str): File path to save the resulting visualization.
    """
    # Resize map from patch-grid (14x14) to full image size (224x224)
    attn_map_resized = cv2.resize(attn_map, (224, 224))
    
    # Use 98th percentile to prevent outliers from washing out the map
    v_max = np.percentile(attn_map_resized, 98)  
    attn_map_resized = np.clip(attn_map_resized, None, v_max)
    
    # Normalize
    attn_map_min, attn_map_max = attn_map_resized.min(), attn_map_resized.max()
    attn_map_resized = (attn_map_resized - attn_map_min) / (attn_map_max - attn_map_min + 1e-8)
    
    # Generate Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0 # Match RGB float format
    
    # Combine 60% heatmap, 40% original image
    visualization = 0.6 * heatmap + 0.4 * raw_img
    visualization = (visualization / visualization.max()) * 255.0
    
    cv2.imwrite(output_path, cv2.cvtColor(np.uint8(visualization), cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser(description="Generate Raw Attention visualizations for Animal/Satellite models.")
    parser.add_argument("--model_path", type=str, default="best_animal_sat_base_resized.pth")
    parser.add_argument("--model_type", type=int, default=3, 
                        help="Model to be used, 1 = Only Animal image, 2 = Only Satelite image, 3 = Cross attention (both images)")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = ModelType(args.model_type)

    # Initialize model architecture based on checkpoint metadata
    checkpoint = torch.load(args.model_path, map_location=device)
    species_map = checkpoint['species_map']
    num_classes = len(species_map)

    if model_type == ModelType.Both:
        model = AnimalSatClassifier(num_classes=num_classes, model_name=args.model_name)
    elif model_type == ModelType.Animal or model_type == ModelType.Satelite:
        model = SingleModalityClassifier(num_classes=num_classes, model_name=args.model_name)
    else:
        raise ValueError(f"Model type {model_type} is not configured for attention mapping.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load file IDs and filter data via DuckDB
    base_path = Path(os.getenv("DATA_FOLDER"))
    db_path = base_path / os.getenv("DATABASE")
    cb_folder = base_path / os.getenv("CB_IMAGE_PATH")
    all_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(cb_folder) if f.endswith('.png')]

    with DuckDBManager(db_path) as db:
        # SQL filters for species with at least 50 samples to ensure statistical relevance
        filtered_data = db.con.execute("""
            WITH valid_species AS (
                SELECT scientificName
                FROM crypticbio 
                WHERE rowid = ANY(?) 
                GROUP BY scientificName
                HAVING COUNT(*) >= ?
            )
            SELECT rowid, scientificName 
            FROM crypticbio 
            WHERE scientificName IN (SELECT scientificName FROM valid_species)
            AND rowid = ANY(?)
        """, [all_ids, 50, all_ids]).df()

    species_list = sorted(filtered_data['scientificName'].unique().tolist())
    valid_ids = filtered_data['rowid'].tolist()
    name_to_id = {name: idx for idx, name in enumerate(species_list)}
    
    dataset = AnimalSateliteDataset(valid_ids, name_to_id, db_path, transform_size=224, model_type=model_type)
    labels = dataset.data['scientificName'].values

    # Stratified split to maintain class balance in visualizations
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)

    transforms_dict = get_transforms(224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    first_test_idx = test_idx[0]

    if model_type == ModelType.Both:
        animal_raw, sat_raw, label_id = dataset[first_test_idx]
        animal_tensor = transforms_dict['test'](animal_raw).unsqueeze(0).to(device)
        sat_tensor = transforms_dict['test'](sat_raw).unsqueeze(0).to(device)
        
        # Target block -3: Deep enough for semantics, avoids 'attention sink' bias in final layers
        attn_module = model.animal_backbone.blocks[-3].attn
        
        # Disable fused_attn to ensure the hook can access the intermediate dropout input
        if hasattr(attn_module, 'fused_attn'):
            attn_module.fused_attn = False
            
        hook_handle = attn_module.attn_drop.register_forward_hook(get_attention_hook)

        with torch.no_grad():
            logits, cross_attn_weights = model(animal_tensor, sat_tensor, attn_bool=True)

        raw_animal_img = np.array(animal_raw.resize((224, 224))) / 255.0
        raw_sat_img = np.array(sat_raw.resize((224, 224))) / 255.0

        # Parse Self-Attention
        animal_head_attn = extracted_attn_weights[0, :, 0, 1:]  
        avg_animal_attn = animal_head_attn.mean(dim=0) # Average over all attention heads               
        animal_attn_map = avg_animal_attn.reshape(14, 14).cpu().numpy()
        
        animal_out_path = os.path.join(args.output_dir, f"attention_animal_{first_test_idx}.jpg")
        process_and_save_attention(animal_attn_map, raw_animal_img, animal_out_path)

        # Parse Cross-Attention
        avg_sat_attn = cross_attn_weights[0].mean(dim=0)             
        sat_spatial_attn = avg_sat_attn[1:]                                          
        sat_attn_map = sat_spatial_attn.reshape(14, 14).cpu().numpy()
        
        sat_out_path = os.path.join(args.output_dir, f"attention_sat_{first_test_idx}.jpg")
        process_and_save_attention(sat_attn_map, raw_sat_img, sat_out_path)
        
        hook_handle.remove()

    elif model_type == ModelType.Animal or model_type == ModelType.Satelite:
        image_raw, label_id = dataset[first_test_idx]
        image_tensor = transforms_dict['test'](image_raw).unsqueeze(0).to(device)
        
        # Target the generic backbone block -3 for single-modality ViT
        attn_module = model.backbone.blocks[-3].attn
        if hasattr(attn_module, 'fused_attn'):
            attn_module.fused_attn = False
            
        hook_handle = attn_module.attn_drop.register_forward_hook(get_attention_hook)

        with torch.no_grad():
            logits = model(image_tensor)

        raw_img = np.array(image_raw.resize((224, 224))) / 255.0

        # Self-Attention for single image 
        image_head_attn = extracted_attn_weights[0, :, 0, 1:]          
        avg_image_attn = image_head_attn.mean(dim=0)               
        image_attn_map = avg_image_attn.reshape(14, 14).cpu().numpy()

        image_out_path = os.path.join(args.output_dir, f"attention_{model_type.name}_{first_test_idx}.jpg")
        process_and_save_attention(image_attn_map, raw_img, image_out_path)
        
        hook_handle.remove()

    print(f"Analysis saved for sample {first_test_idx}.")
    print(f"True Label: {species_map[label_id.item()]}")

if __name__ == "__main__":
    main()