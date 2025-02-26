import torch
import torch.nn as nn
from data_preparation import prepare_data
from model_builder import model_builder
from train_utils import train_step,test_step,train
from utils import get_class_names, accuracy_fn
from pathlib import Path


data_path = Path("./brain_tumor_dataset")

class_name,class_to_idx,idx_to_class = get_class_names(data_path/"Training")

train_dataloader,val_dataloader,test_dataloader = prepare_data(data_path,"Training","Testing")

model = model_builder(len(class_name))

img,label = next(iter(train_dataloader))
x = img[0].unsqueeze(dim=0)
print("no probs")
y = model(x)
print(y)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loss,train_acc,test_loss,test_acc = train(2,train_dataloader,val_dataloader,model,loss_fn,optimizer,accuracy_fn)

