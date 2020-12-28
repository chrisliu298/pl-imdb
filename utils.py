import torch


def binary_accuracy(pred, y):
    rounded_pred = torch.round(pred)
    correct = (rounded_pred == y).float()
    return correct.sum() / len(correct)