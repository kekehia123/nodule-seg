import torch
from torch import nn
from torch.nn import functional as F


class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None):
        if class_weights is not None:
            nll_weight = class_weights.to(torch.device('cuda'))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes=len(class_weights)

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            cls_weight = self.jaccard_weight / (self.num_classes-2)
            eps = 1e-15
            for cls in range(self.num_classes-2):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum() + eps
                loss += (1 - intersection / (union - intersection)) * cls_weight

            loss /= (1 + self.jaccard_weight)
        return loss