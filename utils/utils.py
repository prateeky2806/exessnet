from PIL import Image
import time
import pathlib

import torch
import torch.nn as nn
import models
import models.module_util as module_util
import torch.backends.cudnn as cudnn

from models.modules import FastMultitaskMaskConv, MultitaskMaskConv
from args import args


def cond_cache_masks(m,):
    if hasattr(m, "cache_masks"):
        m.cache_masks()


def cond_cache_weights(m, t):
    if hasattr(m, "cache_weights"):
        m.cache_weights(t)


def cond_clear_masks(m,):
    if hasattr(m, "clear_masks"):
        m.clear_masks()


def cond_set_mask(m, task):
    if hasattr(m, "set_mask"):
        m.set_mask(task)


def cache_masks(model):
    model.apply(cond_cache_masks)


def cache_weights(model, task):
    model.apply(lambda m: cond_cache_weights(m, task))


def clear_masks(model):
    model.apply(cond_clear_masks)


def set_mask(model, task):
    model.apply(lambda m: cond_set_mask(m, task))


def freeze_model_weights(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Freezing weight for {n}")
            m.weight.requires_grad_(False)

            if m.weight.grad is not None:
                m.weight.grad = None
                print(f"==> Resetting grad value for {n} -> None")


def freeze_model_scores(model: nn.Module, task_idx: int):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Freezing weight for {n}")
            m.scores[task_idx].requires_grad_(False)

            if m.scores[task_idx].grad is not None:
                m.scores[task_idx].grad = None
                print(f"==> Resetting grad value for {n} scores -> None")


def unfreeze_model_weights(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Unfreezing weight for {n}")
            m.weight.requires_grad_(True)


def unfreeze_model_scores(model: nn.Module, task_idx: int):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Unfreezing weight for {n}")
            m.scores[task_idx].requires_grad_(True)


def set_gpu(model):
    if args.multigpu is None:
        args.device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
        args.device = torch.cuda.current_device()
        cudnn.benchmark = True

    return model


def get_model(text_exp, num_classes):
    return models.__dict__[args.model](num_classes)

def write_result_to_csv(**kwargs):
    results = pathlib.Path(args.log_dir) / "results.csv"

    if not results.exists():
        results.write_text("Date Finished,Name,Current Val,Best Val,Save Directory\n")

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{name}, "
                "{curr_acc1:.04f}, "
                "{best_acc1:.04f}, "
                "{save_dir}\n"
            ).format(now=now, **kwargs)
        )


def write_adapt_results(**kwargs):
    results = pathlib.Path(args.run_base_dir) / "adapt_results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished,"
            "Name,"
            "Task,"
            "Num Tasks Learned,"
            "Current Val,"
            "Adapt Val\n"
        )
    now = time.strftime("%m-%d-%y_%H:%M:%S")
    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{name}~task={task}~numtaskslearned={num_tasks_learned}~tasknumber={task_number}, "
                "{task}, "
                "{num_tasks_learned}, "
                "{curr_acc1:.04f}, "
                "{adapt_acc1:.04f}\n"
            ).format(now=now, **kwargs)
        )

def write_adapt_results(**kwargs):
    results = pathlib.Path(args.run_base_dir) / "adapt_results.csv"

    if not results.exists():
        results.write_text( "Date Finished," "Learned Task," "Eval Task," "Last Val," "Best Mask Val," "Best Weight Val," "Adapt Val\n" )
    now = time.strftime("%m-%d-%y_%H:%M:%S")
    with open(results, "a+") as f:
        f.write( ( "{now}, " "{num_tasks_learned}, " "{task}, " "{last_acc:.04f}, " "{best_mask_acc:.04f}, " "{best_weight_acc:.04f}, " "{adapt_acc:.04f}\n" ).format(now=now, **kwargs) )


class BasicVisionDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform, target_transform):
        assert len(data) == len(targets)

        self.data = data
        self.targets = targets

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def kth_elt(x, base):
    if base == 2:
        return x.median()
    else:
        val, _ = x.flatten().sort()
        return val[(val.size(0) - 1) // base]


# TODO: Remove this with task-eval
def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [{"params": bn_params, "weight_decay": args.wd,}, {"params": rest_params, "weight_decay": args.wd},],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd, )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop( filter(lambda p: p.requires_grad, model.parameters()), lr=lr )
    return optimizer