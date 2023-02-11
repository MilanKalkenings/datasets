from milankalkenings.deep_learning import Module
import torch
from torch.utils.data import DataLoader


def accuracy(module: Module, loader_eval: DataLoader):
    module.eval()
    correct_count = 0
    obs_count = 0
    for batch in loader_eval:
        x, y = batch
        with torch.no_grad():
            scores = module(x=x, y=y)["scores"]
            preds = torch.argmax(scores, dim=1)
            correct_count += int((preds == y).sum())
            obs_count += len(y)
    return correct_count / obs_count