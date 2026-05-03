import torch
from PIL import Image
from src.data_gather import DuckDBManager
from torch.utils.data import Dataset

def get_transforms(transform_size, mean, std):
    common_post_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    resize_transform = [transforms.Resize((224, 224))] if transform_size == -1 else []

    return {
        'train': transforms.Compose(
            resize_transform + 
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)] + 
            common_post_transforms
        ),
        'val': transforms.Compose(
            resize_transform + 
            common_post_transforms
        ),
        'test': transforms.Compose(
            resize_transform + 
            common_post_transforms
        )
    }

class Transform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        animal_img, sat_img, label = self.subset[index]
        if self.transform:
            animal_img = self.transform(animal_img)
            sat_img = self.transform(sat_img)
        return animal_img, sat_img, label

class TransformSingleModality(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

class AnimalSateliteDataset(Dataset):
    def __init__(self, row_ids, name_to_id, db_path, transform_size: int = -1, model_type: ModelType = ModelType.Both):
        self.name_to_id = name_to_id
        self.transform_size = transform_size
        self.model_type = model_type

        with DuckDBManager(db_path) as db:
            self.data = db.con.execute(
                "SELECT * FROM crypticbio WHERE id = ANY(?)", [row_ids]
            ).df()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.transform_size != -1:
            animal_path = row['crypticbio_image'].replace("images_cb", f"images_cb_{self.transform_size}")
            sat_path = row['sentinel_image'].replace("images_sh", f"images_sh_{self.transform_size}")
        else:
            animal_path = row['crypticbio_image']
            sat_path = row['sentinel_image']
        
        label = self.name_to_id[row['scientificName']]

        if self.model_type == ModelType.Animal:
            animal_img = Image.open(animal_path).convert('RGB')
            return animal_img, torch.tensor(label, dtype=torch.long)
        elif self.model_type == ModelType.Satelite:
            sat_img = Image.open(sat_path).convert('RGB')
            return sat_img, torch.tensor(label, dtype=torch.long)
        else:
            animal_img = Image.open(animal_path).convert('RGB')
            sat_img = Image.open(sat_path).convert('RGB')
            return animal_img, sat_img, torch.tensor(label, dtype=torch.long)
    

