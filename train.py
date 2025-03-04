import torch
import torch.nn as nn
from data_preparation import prepare_data
from model_builder import model_builder
from train_utils import train_step,test_step,train
from utils import get_class_names, accuracy_fn
from pathlib import Path
from utils import evaluate
import argparse


parser = argparse.ArgumentParser(description="Train a model on the brain tumor dataset")
parser.add_argument("-e","--epochs",type=int,default=2,help="Number of epochs to train the model")
args = parser.parse_args()

EPOCHS = args.epochs

data_path = Path("./brain_tumor_dataset")

class_name,class_to_idx,idx_to_class = get_class_names(data_path/"Training")

train_dataloader,val_dataloader,test_dataloader = prepare_data(data_path,"Training","Testing")

model = model_builder(len(class_name))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loss,train_acc,valid_loss,valid_acc = train(EPOCHS,train_dataloader,val_dataloader,model,loss_fn,optimizer,accuracy_fn)

test_acc = evaluate(model,test_dataloader)
print(f"Test Accuracy: {test_acc}")