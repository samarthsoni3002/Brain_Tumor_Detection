import os
import torch
from typing import Tuple,List
import matplotlib.pyplot as plt
import numpy as np

def get_class_names(path:str) -> Tuple[list[str], dict[str, int], dict[int, str]]:

    class_names = [classes.name for classes in os.scandir(path)]
    class_to_idx = {classes:index for index,classes in enumerate(class_names)}
    idx_to_class = {index:classes for index,classes in enumerate(class_names)}
    return class_names,class_to_idx,idx_to_class


def accuracy_fn(y_true,y_pred):
    
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = 100*(correct/len(y_pred))
    return acc

def evaluate(model,test_dataloader):
    
    test_acc = 0 

    for X,y in test_dataloader:
            y_pred = model(X)
            test_acc+= accuracy_fn(y,y_pred.argmax(dim=1))

    return test_acc/len(test_dataloader)


def visualization(train_loss: List[float],train_acc: List[float],valid_loss: List[float],valid_acc: List[float],epochs:int):

    
    train_loss = [t.detach().item() for t in train_loss]
    valid_loss = [t.detach().item() for t in valid_loss]


    train_num_epochs = np.linspace(1, epochs, len(train_loss))
    valid_num_epochs = np.linspace(1, epochs, len(valid_loss))


    plt.figure(figsize=(15,5))

    plt.subplot(2,2,1)
    plt.plot(train_num_epochs,train_loss,label="Train Loss",marker="o")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.title("Training Loss")

    plt.subplot(2,2,2)
    plt.plot(valid_num_epochs,valid_loss,label="Validation Loss", marker="o")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.title("Validation Loss")

    plt.subplot(2,2,3)
    plt.plot(train_num_epochs,train_acc,label="Train Accuracy", marker="s")
    plt.xlabel("Acc")
    plt.ylabel("Epochs")
    plt.title("Training Accuracy")

    plt.subplot(2,2,4)
    plt.plot(valid_num_epochs,valid_acc,label="Validation Accuracy", marker="s")
    plt.xlabel("Accuracy")
    plt.ylabel("Epochs")
    plt.title("Testing Accuracy")
 

    plt.subplots_adjust(hspace=0.5)


    plt.show()
    