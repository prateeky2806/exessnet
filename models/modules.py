import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import models.module_util as module_util

from args import args as pargs

from scipy.stats import ortho_group

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d

StandardConv1d = nn.Conv1d
StandardBN1d = nn.BatchNorm1d

class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )

class MultitaskNonAffineBN(nn.Module):
    def __init__(self, dim):
        super(MultitaskNonAffineBN, self).__init__()
        self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(pargs.num_tasks)])
        self.task = 0

    def forward(self, x):
        return self.bns[self.task](x)

class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(module_util.mask_init(self))

        # Turn the gradient on the weights off

        # default sparsity
        self.sparsity = pargs.sparsity

    def forward(self, x):
        subnet = module_util.GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

# Conv from What's Hidden in a Randomly Weighted Neural Network?
class MultitaskMaskConv(nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # import pdb; pdb.set_trace()
        self.scores = nn.ParameterList( [ nn.Parameter(module_util.mask_init(self)) for _ in range(pargs.num_tasks) ] )

        self.sparsity = pargs.sparsity

    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack( [ module_util.get_subnet(self.scores[j].abs(), self.sparsity) for j in range(pargs.num_tasks) ] ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        if self.task < 0:
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = ( alpha_weights[idxs] * self.stacked[: self.num_tasks_learned][idxs] ).sum(dim=0)
        else:
            subnet = module_util.GetSubnet.apply( self.scores[self.task].abs(), self.sparsity )

        w = self.weight * subnet
        x = F.conv2d( x, w, self.bias, self.stride, self.padding, self.dilation, self.groups )
        return x

    def __repr__(self):
        return f"MultitaskMaskConv({self.in_channels}, {self.out_channels})"

class MaskConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(module_util.mask_init(self))
        self.sparsity = pargs.sparsity

class MultitaskMaskConv1d(nn.Conv1d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # import pdb; pdb.set_trace()
        self.scores = nn.ParameterList( [ nn.Parameter(module_util.mask_init(self)) for _ in range(pargs.num_tasks) ] )

        self.sparsity = pargs.sparsity

    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack( [ module_util.get_subnet(self.scores[j].abs(), self.sparsity) for j in range(pargs.num_tasks) ] ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        if self.task < 0:
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = ( alpha_weights[idxs] * self.stacked[: self.num_tasks_learned][idxs] ).sum(dim=0)
        else:
            subnet = module_util.GetSubnet.apply( self.scores[self.task].abs(), self.sparsity )

        w = self.weight * subnet
        x = F.conv1d( x, w, self.bias, self.stride, self.padding, self.dilation, self.groups )
        return x

    def __repr__(self):
        return f"MultitaskMaskConv1D({self.in_channels}, {self.out_channels})"

class NonAffineBN1d(nn.BatchNorm1d):
    def __init__(self, dim):
        super(self).__init__(dim, affine=False)

class MultitaskNonAffineBN1D(nn.Module):
    def __init__(self, dim):
        super(MultitaskNonAffineBN1D, self).__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim, affine=False) for _ in range(pargs.num_tasks)])
        self.task = 0

    def forward(self, x):
        return self.bns[self.task](x)

# Init from What's Hidden with masking from Mallya et al. (Piggyback)
class FastMultitaskMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.ParameterList(
            [nn.Parameter(module_util.mask_init(self)) for _ in range(pargs.num_tasks)]
        )


    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    module_util.get_subnet_fast(self.scores[j])
                    for j in range(pargs.num_tasks)
                ]
            ),
        )


    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        if self.task < 0:
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            subnet = module_util.GetSubnetFast.apply(self.scores[self.task])

        w = self.weight * subnet

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def __repr__(self):
        return f"FastMultitaskMaskConv({self.in_channels}, {self.out_channels})"

class IndividualHeads(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.ParameterList(
            [nn.Parameter(self.weight.data.clone()) for _ in range(pargs.num_tasks)]
        )

    def forward(self, x):
        w = self.scores[self.task]
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def __repr__(self):
        return f"IndividualHeads({self.in_channels}, {self.out_channels})"

class StackedFastMultitaskMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.ParameterList(
            [nn.Parameter(module_util.mask_init(self)) for _ in range(pargs.num_tasks)]
        )


    def forward(self, x):
        if self.task < 0:
            stacked = torch.stack(
                [
                    module_util.get_subnet_fast(self.scores[j])
                    for j in range(min(pargs.num_tasks, self.num_tasks_learned))
                ]
            )
            alpha_weights = self.alphas[: self.num_tasks_learned]
            subnet = (alpha_weights * stacked).sum(dim=0)
        else:
            subnet = module_util.GetSubnetFast.apply(self.scores[self.task])
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
