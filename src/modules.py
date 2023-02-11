from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn
from milankalkenings.deep_learning import Module


class ImageClassifier(Module):
    def __init__(self, n_classes: int):
        super(ImageClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(in_features=512, out_features=n_classes)
        self.loss = nn.CrossEntropyLoss()

    def freeze_pretrained(self):
        # todo
        pass

    def unfreeze_pretrained(self):
        # todo
        pass

    def forward(self, x, y):
        scores = self.resnet(x)
        loss = self.loss(scores, y)
        return {"loss": loss, "scores": scores}



