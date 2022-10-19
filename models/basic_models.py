import math
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))


class LeNetBasic(nn.Module):
    def __init__(self, num_classes, width_mult=1):
        super(LeNetBasic, self).__init__()
        self.linear = nn.Sequential(
            nn.Conv2d(28 * 28, int(300 * width_mult), kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(300 * width_mult), int(100 * width_mult), kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(100 * width_mult), num_classes, kernel_size=1, stride=1, bias=False),
        )
        self.linear.apply(init_weights)

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()

class FC1024Basic(nn.Module):
    def __init__(self, num_classes, width_mult=1):
        super(FC1024Basic, self).__init__()

        self.linear = nn.Sequential(
            nn.Conv2d(28 * 28, int(width_mult * 1024), kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(width_mult * 1024), int(width_mult * 1024), kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(width_mult * 1024), num_classes, kernel_size=1, stride=1, bias=False),
        )
        self.linear.apply(init_weights)

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()