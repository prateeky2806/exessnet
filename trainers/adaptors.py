from args import args
import torch
from torch import optim
import math

import numpy as np
import pathlib

from models import module_util
from utils.utils import kth_elt
from functools import partial


def adapt_test( model, data_loader, split, alphas=None, ):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if args.text_exp:
                data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                output = model(data, mask)
            else:
                data, target = batch[0].to(args.device), batch[1].to(args.device)
                output = model(data)
            pred = output.argmax( dim=1, keepdim=True )
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += pred.numel()

        acc = float(correct) / float(total)
    print(f"{split} Accuracy: {acc:.6f}\tCorrect: {correct}\tTotal: {total}")  
    return acc

# gt means ground truth task -- corresponds to GG
def gt( model, writer, data_loader, num_tasks_learned, task, split ):
    model.zero_grad()
    model.train()

    # changed requires_grad to False.
    alphas = ( torch.zeros( [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True ) )
    alphas.data[task] = 1

    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "task", task))

    acc = adapt_test( model, data_loader, split, alphas)
    print( f"\nLearned Task:{num_tasks_learned-1}\tTesting Task:{task}\tAccuracy:({acc:.4f}%)" )
    model.apply(lambda m: setattr(m, "alphas", None))
    return acc


def epoch_evaluate( model, data_loader, in_task, out_task, split):
    model.eval()
    if in_task is not None:
        model.apply(lambda m: setattr(m, "task", in_task))
    
    if split.lower() in ['val', 'validation']:
        loaders = data_loader.val_loaders
    elif split.lower() in ['test']:
        loaders = data_loader.test_loaders
    else:
        raise NotImplementedError(f'{split} not implemented')
    
    if out_task == 'all':
        task_list = range(args.num_tasks)
    elif type(out_task) == int:
        task_list = [out_task]
    else:
        raise NotImplementedError(f'{out_task} not implemented')

    with torch.no_grad():
        acc = np.zeros(len(task_list))
        for i, idx in enumerate(task_list):
            correct, total = 0, 0
            for batch_idx, batch in enumerate(loaders[idx]):
                if args.debug and batch_idx > 10: break
                if args.text_exp:
                    token_type_ids = None
                    if args.superglue:
                        data, mask, token_type_ids, target = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),\
                             batch['token_type_ids'].to(args.device), batch['labels'].to(args.device)
                    else:
                        data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                    output = model(data, mask, token_type_ids=token_type_ids)
                else:
                    data, target = batch[0].to(args.device), batch[1].to(args.device)
                    output = model(data)
                pred = output.argmax( dim=1, keepdim=True )
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += pred.numel()

            acc[i] = float(correct) / float(total)
            print(f"In Task: {in_task}\tOut Task: {idx}\t{split} Accuracy: {acc[i]:.6f} ")
    if len(acc) == 1:
        return acc[0]
    return acc


def adapt_all( model, test_loaders):
    
    model.eval()
    
    with torch.no_grad():
        test_acc = np.zeros(args.num_tasks)
        for idx in range(args.num_tasks):
            correct = 0
            total = 0
            for batch_idx, batch in enumerate(test_loaders[idx]):
                if args.text_exp:
                    token_type_ids = None
                    if args.superglue:
                        data, mask, token_type_ids, target = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),\
                             batch['token_type_ids'].to(args.device), batch['labels'].to(args.device)
                    else:
                        data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                    output = model(data, mask, token_type_ids=token_type_ids)
                else:
                    data, target = batch[0].to(args.device), batch[1].to(args.device)
                    output = model(data)
                pred = output.argmax( dim=1, keepdim=True )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += pred.numel()

            test_acc[idx] = float(correct) / float(total)
            print(f"Task{idx} Val Accuracy: {test_acc[idx]:.6f} ")
        
    return test_acc