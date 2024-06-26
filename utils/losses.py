import numpy as np
from monai.losses.dice import *  # NOQA
import torch
from monai.losses.dice import DiceCELoss
from monai.networks.utils import one_hot
import torch.nn as nn
import torch.nn.functional as F

seed = 2025
torch.manual_seed(seed)
# B, C, H, W = 4, 2, 3, 3
# input = torch.rand(B, C, H, W)
# target_idx = torch.randint(low=0, high=2, size=(B, H, W)).long()
# print(target_idx.shape)
# print(target_idx[:, None, ...].shape)
# target = one_hot(target_idx[:, None, ...], num_classes=2)
# print(target.shape)
# self = DiceCELoss(reduction="mean")
# loss = self(input, target)
# print(loss)


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

class KLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean', eps=1e-10):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, output1, output2):
        output1 = F.log_softmax(output1 + self.eps, dim=1)
        output2 = F.softmax(output2 + self.eps, dim=1)
        return F.kl_div(output1, output2, reduction=self.reduction)


def distillation_loss(output1, output2, temperature=7):
    """
    Compute the distillation loss with KL Didistillation_lossvergence.

    Arguments:
    - output1 : the output tensor from the first model
    - output2 : the output tensor from the second model
    - temperature : the temperature scaling parameter

    Returns:
    - loss : the computed distillation loss
    """

    # compute the soft targets
    soft_target1 = F.softmax(output1 / temperature, dim=1)
    soft_target2 = F.softmax(output2 / temperature, dim=1)

    # compute the loss with KL Divergence
    loss = F.kl_div(soft_target2.log(), soft_target1, reduction='batchmean')

    return loss



