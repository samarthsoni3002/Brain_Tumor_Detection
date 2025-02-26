import torch
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from data_creation import BrainTumorDataset
from pathlib import Path
from typing import Tuple

def prepare_data(path:Path, train_str:str,test_str:str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(path)
    train_path = data_path / "Training"
    test_path = data_path / "Testing"

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((200, 200)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])


    train_dataset = BrainTumorDataset(train_path,train_transforms)
    test_dataset = BrainTumorDataset(test_path,test_transforms)

    val_size = int(0.15*len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset,val_dataset = random_split(train_dataset,[train_size,val_size])

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=32,num_workers=1,shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=32,num_workers=1,shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=32,num_workers=1,shuffle=False)

    return train_dataloader,val_dataloader,test_dataloader