from typing import Any
import torch
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CatsDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform = None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.transform = transform


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(os.path.join(self.data_dir, self.files[index]))
        if self.transform:
            image = self.transform(image)

        return image
        

def collate_fn(dataset_items: list):
    return {
        "images": torch.stack([item for item in dataset_items])
    }

def get_dataloaders(data_dir: str, 
                    transform = None,
                    num_workers: int = 4, 
                    batch_size: int = 64, ):
    dataset = CatsDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return {
        "train": dataloader
    }