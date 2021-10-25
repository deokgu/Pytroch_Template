import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import timm
from torchvision import models


class ReadTimmModule(BaseModel):  #
    def __init__(self, model_arch, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_arch, num_classes=num_classes, pretrained=pretrained
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ReadTorchvisionModule(BaseModel):  #
    def __init__(self, model_arch, num_classes, pretrained=True, classifier=None):
        super().__init__()

        self.model = eval(f"models.{model_arch}(pretrained={pretrained})")
        self.model.fc = eval(classifier)

    def forward(self, x):
        x = self.model(x)
        return x
