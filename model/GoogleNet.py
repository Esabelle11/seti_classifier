import torch.nn as nn
from torchvision import models

class GoogleNet_model(nn.Module):
    def __init__(self, num_classes=7):
        super(GoogleNet_model, self).__init__()
        pretrained_googlenet = models.googlenet(pretrained=True)
        self.features = nn.Sequential(*list(pretrained_googlenet.children())[:-3])
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
