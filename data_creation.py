import torch
from torch.utils.data import Dataset
from utils import get_class_names
from PIL import Image
from pathlib import Path
from typing import Optional,Callable



class BrainTumorDataset(Dataset):

    def __init__(self, path: Path, transform: Optional[Callable] = None):
        self.paths = list(path.glob("*/*.jpg"))
        self.classes, self.class_to_idx, self.idx_to_class = get_class_names(path)
        self.transform = transform

    def load_image(self, index: int):
        image_path = self.paths[index]
        img = Image.open(image_path)
        return img.convert("RGB")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_to_idx = self.class_to_idx[class_name]

        if self.transform is None:
            return img, class_to_idx

        transformed_img = self.transform(img)
        return transformed_img, class_to_idx


