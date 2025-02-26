import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from typing import List,Callable


def train_step(train_dataloader: DataLoader,model: nn.Module,loss_fn: Callable,optimizer: torch.optim.Optimizer,accuracy_fn: Callable,train_loss:List[float],train_acc:List[float]):
    
    
    epoch_train_loss = 0
    epoch_train_acc = 0

    for X,y in train_dataloader:

        model.train()
        y_pred = model(X)
        tr_loss = loss_fn(y_pred.squeeze(),y)
        tr_acc = accuracy_fn(y,y_pred.argmax(dim=1))

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

        epoch_train_loss += tr_loss
        epoch_train_acc += tr_acc

        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

    epoch_train_loss/=len(train_dataloader)
    epoch_train_acc/=len(train_dataloader)

    return epoch_train_loss,epoch_train_acc

 
def test_step(val_dataloader: DataLoader,model: nn.Module,loss_fn: Callable,accuracy_fn: Callable,test_loss:List[float],test_acc:List[float]):
    
   
    epoch_test_loss = 0
    epoch_test_acc = 0

    with torch.inference_mode():

        model.eval()
        for X,y in val_dataloader:
             
            y_pred = model(X)
            te_loss = loss_fn(y_pred,y)
            te_acc = accuracy_fn(y,y_pred.argmax(dim=1))
    
            epoch_test_loss += te_loss
            epoch_test_acc += te_acc
    
            test_loss.append(te_loss)
            test_acc.append(te_acc)
    epoch_test_loss = epoch_test_loss/len(val_dataloader)
    epoch_test_acc = epoch_test_acc/len(val_dataloader)
                    

    return epoch_test_loss,epoch_test_acc


def train(epochs: int,train_dataloader: DataLoader,val_dataloader: DataLoader):
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for epoch in range(epochs):

        epoch_train_loss,epoch_train_acc = train_step(train_dataloader,train_loss,train_acc)
        epoch_test_loss,epoch_test_acc = test_step(val_dataloader, test_loss, test_acc)

        print(f"\n{epoch+1} Train loss: {epoch_train_loss} | Train Accuracy: {epoch_train_acc} | Validation Loss: {epoch_test_loss} | Validation Acc: {epoch_test_acc}")