import torch
import torch.nn as nn
from torchvision import models

def model_builder():
    model = models.resnet34(weights=models.ResNet34_Weights)

    for param in model.parameters():
        param.requires_grad=False

    for param in model.layer4.parameters():
        param.requires_grad=True

    model.fc = nn.Sequential(nn.Linear(512,4))

    return model