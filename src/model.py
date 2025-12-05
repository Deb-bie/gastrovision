from torchvision import models # type: ignore
import torch.nn as nn # type: ignore


def create_resnet18_model(num_classes=4, pretrained=True):
    """Create ResNet18 model with custom classifier"""
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model
















