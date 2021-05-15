import torch
from torch import nn


class L1Loss(object):
    def __init__(self):
        pass

    def __call__(self, predicted: torch.tensor, gt: torch.tensor):
        return torch.abs(predicted - gt).mean()


class SigmoidFocalLoss(object):
    def __init__(self, gamma: float = 2.0):
        self._gamma = gamma

    def __call__(self, predicted: torch.tensor, gt: torch.tensor, gamma=2.0):
        pred = torch.clamp(predicted.sigmoid_(), min=1e-4, max=1 - 1e-4)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred,
                                                    gamma) * neg_inds * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return neg_loss
        return (pos_loss + neg_loss) / num_pos
