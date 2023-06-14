import torch.nn as nn
import torch

class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, gamma=2, ignore_index=-100, reduction="mean"):
        # initialize Cross Entropy Loss
        super().__init__(weight=None, ignore_index=ignore_index, reduction="none")
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logits, targets):
        # get Cross Entropy Loss per class
        cross_entropy_loss = super().forward(logits, targets)
        # calculate Focal Loss (by reverse engineering Cross Entropy Loss)
        input_probs = torch.exp(-1 * cross_entropy_loss)
        loss        = cross_entropy_loss * torch.pow(1 - input_probs, self.gamma)

        if (self.reduction == "mean"):
            return torch.mean(loss)
        elif (self.reduction == "sum"):
            return torch.sum(loss)
        elif (self.reduction == "none"):
            return loss
        else:
            assert False, f"Focal Loss \"reduction\" parameter should be one of the following [\"mean\" | \"sum\" |\"none\"]. Got \"{self.reduction}\" instead."

class CombinedLoss(nn.CrossEntropyLoss):
    def __init__(self, gamma=1.3, switch_loss_at=7, ignore_index=-100, reduction="mean"):
        # initialize Cross Entropy Loss
        super().__init__(weight=None, ignore_index=ignore_index, reduction="none")
        self.gamma = gamma
        self.reduction = reduction
        self.switch_epoch = switch_loss_at

    def __call__(self, logits, targets, epoch):
        # get Cross Entropy Loss per class
        cross_entropy_loss = super().forward(logits, targets)

        loss = cross_entropy_loss
        if (epoch >= self.switch_epoch):
            # calculate Focal Loss (by reverse engineering Cross Entropy Loss)
            input_probs = torch.exp(-1 * cross_entropy_loss)
            loss        = cross_entropy_loss * torch.pow(1 - input_probs, self.gamma)

        if (self.reduction == "mean"):
            return torch.mean(loss)
        elif (self.reduction == "sum"):
            return torch.sum(loss)
        elif (self.reduction == "none"):
            return loss
        else:
            assert False, f"Combined loss \"reduction\" parameter should be one of the following [\"mean\" | \"sum\" |\"none\"]. Got \"{self.reduction}\" instead."