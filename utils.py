import os
import torch
from typing import Tuple

def get_class_names(path:str) -> Tuple[list[str], dict[str, int], dict[int, str]]:

    class_names = [classes.name for classes in os.scandir(path)]
    class_to_idx = {classes:index for index,classes in enumerate(class_names)}
    idx_to_class = {index:classes for index,classes in enumerate(class_names)}
    return class_names,class_to_idx,idx_to_class


def accuracy_fn(y_true,y_pred):
    
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = 100*(correct/len(y_pred))
    return acc