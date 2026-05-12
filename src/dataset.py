import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data_gather import DuckDBManager
from src.models import ModelType


def get_transforms(transform_size: int, mean: list, std: list):
    """
    Perform pytorch transformations on a Pytorch Dataset

    Args:
        transform_size (int): The target size for resizing. If set to -1,
            the images are automatically resized to (224, 224). If any other
            value, no initial resizing is applied within this logic.
        mean (list): The sequence of means for each channel (R, G, B)
            used for normalization.
        std (list): The sequence of standard deviations for each
            channel (R, G, B) used for normalization.

    Returns:
        dict: A dictionary with keys "train", "val", and "test", where each
            value is a `transforms.Compose` object.
    """
    common_post_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    # Force the resize to 224 if not specialized
    resize_transform = [transforms.Resize((224, 224))] if transform_size == -1 else []

    return {
        "train": transforms.Compose(
            resize_transform
            + [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
            + common_post_transforms
        ),
        "val": transforms.Compose(resize_transform + common_post_transforms),
        "test": transforms.Compose(resize_transform + common_post_transforms),
    }


class Transform(Dataset):
    """
    A Dataset wrapper for multi-modal data (animal and satellite) that is used
    to easily transform a dataset class such as those below it

    Attributes:
        subset (Dataset): The underlying dataset containing raw (untransformed) samples.
        transform (callable, optional): A function/transform that takes in a
            PIL image and returns a transformed version.
    """

    def __init__(self, subset, transform=None):
        """
        Initializes the Transform wrapper.
        """
        self.subset = subset
        self.transform = transform

    def __len__(self):
        """
        Calculates the total number of samples in the wrapped subset.
        """
        return len(self.subset)

    def __getitem__(self, index):
        """
        Retrieves a sample from the subset at a specific index
        and applies transformations.
        """
        animal_img, sat_img, label = self.subset[index]

        if self.transform:
            animal_img = self.transform(animal_img)
            sat_img = self.transform(sat_img)

        return animal_img, sat_img, label


class TransformSingleModality(Dataset):
    """
    A Dataset wrapper for single-modal data that is used
    to easily transform a dataset class such as those below it

    Attributes:
        subset (Dataset): The underlying dataset containing raw samples.
        transform (callable, optional): A function/transform to apply to the image.
    """

    def __init__(self, subset, transform=None):
        """
        Initializes the TransformSingleModality wrapper.
        """
        self.subset = subset
        self.transform = transform

    def __len__(self):
        """
        Calculates the total number of samples in the wrapped subset.
        """
        return len(self.subset)

    def __getitem__(self, index):
        """
        Retrieves a sample from the subset at a specific index
        and applies transformations.
        """
        img, label = self.subset[index]

        if self.transform:
            img = self.transform(img)

        return img, label


class AnimalSateliteDataset(Dataset):
    """
    A specialized Dataset for fetching paired animal and satellite imagery.

    This class interfaces with a DuckDB database to retrieve image paths and
    labels based on specific row IDs. It supports multi-modal or single-modality
    data retrieval based on the specified model type and handles dynamic
    path resolution for different image resolutions.

    Attributes:
        name_to_id (dict): Mapping of scientific names (strings) to class IDs (ints).
        transform_size (int): The resolution suffix used to locate pre-resized
            images. If -1, original images are used.
        model_type (ModelType): Enum determining whether to return animal imagery,
            satellite imagery, or both.
        data (pd.DataFrame): The loaded metadata from the database.
    """

    def __init__(
        self,
        row_ids: list,
        name_to_id: dict,
        db_path,
        transform_size: int = -1,
        model_type: ModelType = ModelType.Both,
    ):
        """
        Initializes the dataset by querying the database for the provided IDs.
        """
        self.name_to_id = name_to_id
        self.transform_size = transform_size
        self.model_type = model_type

        # Use DuckDB to fetch metadata for the specific subset of IDs
        with DuckDBManager(db_path) as db:
            self.data = db.con.execute(
                "SELECT * FROM crypticbio WHERE id = ANY(?)", [row_ids]
            ).df()

    def __len__(self):
        """
        Returns the number of samples available in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves images and labels, handling path logic and modality selection.
        """
        row = self.data.iloc[idx]

        # Resolve paths: use pre-resized directories if transform_size is specified
        if self.transform_size != -1:
            animal_path = row["crypticbio_image"].replace(
                "images_cb", f"images_cb_{self.transform_size}"
            )
            sat_path = row["sentinel_image"].replace(
                "images_sh", f"images_sh_{self.transform_size}"
            )
        else:
            animal_path = row["crypticbio_image"]
            sat_path = row["sentinel_image"]

        label = self.name_to_id[row["scientificName"]]

        # Return data based on the requested modality
        if self.model_type == ModelType.Animal:
            animal_img = Image.open(animal_path).convert("RGB")
            return animal_img, torch.tensor(label, dtype=torch.long)

        elif self.model_type == ModelType.Satelite:
            sat_img = Image.open(sat_path).convert("RGB")
            return sat_img, torch.tensor(label, dtype=torch.long)

        else:
            # Default case: Both modalities
            animal_img = Image.open(animal_path).convert("RGB")
            sat_img = Image.open(sat_path).convert("RGB")
            return animal_img, sat_img, torch.tensor(label, dtype=torch.long)
