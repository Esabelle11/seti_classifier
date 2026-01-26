import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        num_features = self.vgg16.classifier[0].in_features
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.vgg16(x)
