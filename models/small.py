"""
Replications of models from Frankle et al. Lottery Ticket Hypothesis
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import Builder
from args import args

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        builder = Builder()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, int(300 * args.width_mult), first_layer=True),
            nn.ReLU(),
            builder.conv1x1(int(300 * args.width_mult), int(100 * args.width_mult)),
            nn.ReLU(),
            builder.conv1x1(int(100 * args.width_mult), num_classes, last_layer=True),
        )

    def forward(self, x, reps=False):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        if reps:
            return self.linear[:3](out).squeeze()
        else:
            return self.linear(out).squeeze()

class FC1024(nn.Module):
    def __init__(self, num_classes):
        super(FC1024, self).__init__()
        builder = Builder()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, int(args.width_mult * 1024), first_layer=True),
            nn.ReLU(),
            builder.conv1x1(int(args.width_mult * 1024), int(args.width_mult * 1024)),
            nn.ReLU(),
            builder.conv1x1(int(args.width_mult * 1024), num_classes, last_layer=True),
        )

    def forward(self, x, reps=False):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        if reps:
            return self.linear[:3](out).squeeze()
        else:
            return self.linear(out).squeeze()

class BNNet(nn.Module):
    def __init__(self, num_classes):
        super(BNNet, self).__init__()
        builder = Builder()
        dim = 2048
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, int(dim * args.width_mult), first_layer=True),
            builder.batchnorm(int(dim * args.width_mult)),
            Swish(),
            builder.conv1x1(int(dim * args.width_mult), int(dim * args.width_mult)),
            builder.batchnorm(int(dim * args.width_mult)),
            Swish(),
            builder.conv1x1(int(dim * args.width_mult), num_classes, last_layer=True),
        )

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()
