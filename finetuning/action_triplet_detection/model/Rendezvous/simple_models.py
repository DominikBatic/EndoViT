import torch
import torch.nn as nn
import torchvision.models as basemodels

class ResNet18_with_classifier_head(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_with_classifier_head, self).__init__()

        self.basemodel = basemodels.resnet18(pretrained=pretrained)
        self.basemodel.fc = nn.Identity()
        self.head = nn.Linear(in_features=512, out_features=100)

    def forward(self, x):
        x = self.basemodel(x)
        x = self.head(x)

        return x
        
def get_resnet18(pretrained=True):
    return ResNet18_with_classifier_head(pretrained=pretrained)


class ResNet50_with_classifier_head(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_with_classifier_head, self).__init__()

        self.basemodel = basemodels.resnet50(pretrained=pretrained)
        self.basemodel.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Linear(in_features=1024, out_features=100),
        )

    def forward(self, x):
        x = self.basemodel(x)
        x = self.head(x)

        return x
        
def get_resnet50(pretrained=True):
    return ResNet50_with_classifier_head(pretrained=pretrained)