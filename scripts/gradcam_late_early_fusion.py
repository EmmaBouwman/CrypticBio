import os
import torch
import argparse

from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.dataset import AnimalSateliteDataset, get_transforms
from src.models import (
    AnimalSatClassifier,
    SingleModalityClassifier,
    ModelType
)
from src.data_gather import DuckDBManager

from src.visualization.visualize_gradcam import run_visualization


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth"
    )

    parser.add_argument(
        "--model_type",
        type=int,
        default=3,
        help=(
            "1 = Animal only, "
            "2 = Satellite only, "
            "3 = Both / Late Fusion"
        )
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50"
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index inside test set"
    )

    args = parser.parse_args()
    load_dotenv()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_type = ModelType(args.model_type)

    checkpoint = torch.load(
        args.model_path,
        map_location=device
    )

    species_map = checkpoint["species_map"]
    num_classes = len(species_map)

    if model_type == ModelType.Both:

        model = AnimalSatClassifier(
            num_classes=num_classes,
            model_name=args.model_name
        )

    elif (
        model_type == ModelType.Animal
        or model_type == ModelType.Satelite
    ):

        model = SingleModalityClassifier(
            num_classes=num_classes,
            model_name=args.model_name
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    base_path = Path(os.getenv("DATA_FOLDER"))

    db_path = base_path / os.getenv("DATABASE")

    cb_folder = base_path / os.getenv("CB_IMAGE_PATH")

    all_ids = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(cb_folder)
        if f.endswith(".png")
    ]

    with DuckDBManager(db_path) as db:

        filtered_data = db.con.execute(
            """
            WITH valid_species AS (
                SELECT scientificName
                FROM crypticbio
                WHERE rowid = ANY(?)
                GROUP BY scientificName
                HAVING COUNT(*) >= ?
            )

            SELECT rowid, scientificName
            FROM crypticbio
            WHERE scientificName IN (
                SELECT scientificName
                FROM valid_species
            )
            AND rowid = ANY(?)
            """,
            [all_ids, 50, all_ids]
        ).df()

    species_list = sorted(
        filtered_data["scientificName"]
        .unique()
        .tolist()
    )

    valid_ids = filtered_data["rowid"].tolist()

    name_to_id = {
        name: idx
        for idx, name in enumerate(species_list)
    }

    dataset = AnimalSateliteDataset(
        valid_ids,
        name_to_id,
        db_path,
        transform_size=224,
        model_type=model_type
    )

    labels = dataset.data["scientificName"].values


    train_idx, temp_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=labels[temp_idx],
        random_state=42
    )

    selected_idx = test_idx[args.sample_index]

    transforms_dict = get_transforms(
        224,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )

    if model_type == ModelType.Both:

        cb_raw, sh_raw, label_id = dataset[selected_idx]

        cb_tensor = transforms_dict["test"](
            cb_raw
        ).unsqueeze(0).to(device)

        sh_tensor = transforms_dict["test"](
            sh_raw
        ).unsqueeze(0).to(device)

        run_visualization(
            model=model,
            cb=cb_tensor,
            sh=sh_tensor,
            device=device,
            model_type=model_type
        )

    elif (
        model_type == ModelType.Animal
        or model_type == ModelType.Satelite
    ):

        image_raw, label_id = dataset[selected_idx]

        image_tensor = transforms_dict["test"](
            image_raw
        ).unsqueeze(0).to(device)


        run_visualization(
            model=model,
            cb=image_tensor,
            sh=image_tensor,
            device=device,
            model_type=model_type
        )

    print(f"\nGrad-CAM generated for sample: {selected_idx}")
    print(f"True label: {species_map[label_id.item()]}")


if __name__ == "__main__":
    main()