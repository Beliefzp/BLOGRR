import torch.nn as nn 
import torch
from torch import Tensor
import torchvision.models as models
import torch

class ThresholdNumberSelector(nn.Module):
    def __init__(self, config):
        super(ThresholdNumberSelector, self).__init__()
        self.config = config
        self.resnet18 = models.resnet18(pretrained=True)
        if self.config.whether_concatenate:
            self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.resnet18.conv1.weight, mode="fan_out", nonlinearity="relu")
        if self.resnet18.conv1.bias is not None:
            nn.init.zeros_(self.resnet18.conv1.bias)
        nn.init.xavier_normal_(self.resnet18.fc.weight)
        if self.resnet18.fc.bias is not None:
            nn.init.zeros_(self.resnet18.fc.bias)

    def Dice_loss_function(self, x_rec: Tensor, Seg: Tensor, reduce_batch_first: bool = True) -> dict:
        x_rec = x_rec.squeeze(1)
        Seg = Seg.squeeze(1)
        assert x_rec.size() == Seg.size()
        assert x_rec.dim() == 3 or not reduce_batch_first
        sum_dim = (-1, -2) if x_rec.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
        inter = 2 * (x_rec * Seg).sum(dim=sum_dim)
        sets_sum = x_rec.sum(dim=sum_dim) + Seg.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        dice = (inter + 1e-6) / (sets_sum + 1e-6)
        return 1 - dice.mean()

    def forward(self, x):
        x = self.resnet18(x)
        x = self.sigmoid(x) 
        return x

    