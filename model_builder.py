import torch.nn as nn
from torchvision import models

def model_builder(classes):
    
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad=False

    for param in model.layer4.parameters():
        param.requires_grad=True

    model.fc = nn.Linear(model.fc.in_features, classes)

    return model