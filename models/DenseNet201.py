"""Single-channel DenseNet201 with the original input stem retained."""
import torch.nn as nn
from torchvision.models import densenet201


class DenseNet201_1ch(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, keep_stem=True):
        super().__init__()
        if pretrained or not keep_stem:
            raise ValueError("The manuscript configuration uses pretrained=False and keep_stem=True")
        self.model = densenet201(weights=None)
        self.model.features.conv0 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
