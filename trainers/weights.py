from args import args

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import module_util
import numpy as np


def init(args):
    pass

def get_editable_weights_mask_dict(model, type):
    mask_dict = {}
    curr_mask_sum = {}

    for n, module in model.named_modules():
        if hasattr(module, 'scores'):
            assert isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d)
            curr_mask = module_util.get_subnet( module.scores[module.task].abs(), module.sparsity ).type(torch.bool)
            curr_mask_sum[n] = curr_mask.sum()
            editable_weight_mask = None
            
            if type == "original":
                mask_dict[n] = curr_mask
            elif type == "exclusive":                               
                if module.task == 0:
                    mask_dict[n] = curr_mask
                else:
                    for ii in range(module.task):
                        mask_ii = module_util.get_subnet( module.scores[ii].abs(), module.sparsity ).type(torch.bool)
                        if editable_weight_mask is None:
                            editable_weight_mask = (curr_mask ^ (curr_mask & mask_ii))
                        else:
                            editable_weight_mask = editable_weight_mask & (curr_mask ^ (curr_mask & mask_ii))
                    mask_dict[n] = editable_weight_mask
    return mask_dict, curr_mask_sum



class EWC:
    def __init__(self):
        pass

    def ewc_loss(self, model):
        losses = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert "weight" in n or 'bias' in n, 'Parameter is not a model weight or bias.'
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = Variable(getattr(model, '{}_mean'.format(n)))
                fisher = Variable(getattr(model, '{}_fisher'.format(n)))
                losses.append((fisher * (p-mean)**2).sum())
        return sum(losses) 

    def consolidate_fisher( self, model, criterion, optimizer, dataloader, num_samples=500):
        """
        Compute Diagonal Fisher matrix for each parameter and register the fisher matrix and old parameters.
        """
        # model.eval()
        importances = {k: torch.zeros_like(p) for k,p in model.named_parameters() if p.requires_grad}
        kwargs = {"num_workers": args.workers, "pin_memory": True} if torch.cuda.is_available() else {}
        fisher_dl = DataLoader(dataloader.dataset, batch_size=1, shuffle=True, drop_last=False, **kwargs)
        
        for batch_idx, batch in enumerate(fisher_dl):
            if batch_idx > num_samples:
                break
            optimizer.zero_grad()
            
            if args.text_exp:
                data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                output = model(data, mask)
            else:
                data, target = batch[0].to(args.device), batch[1].to(args.device)
                output = model(data)
            # set weights with respect to which you want the fisher information.
            loss = criterion(output, target)
            loss.backward()
    
            for (k1, p) in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    importances[k1] += p.grad.data.clone().pow(2) / (num_samples * fisher_dl.batch_size)

        for n, p in model.named_parameters():
            if p.grad is not None:
                assert "weight" in n or 'bias' in n, 'Parameter is not a model weight or bias.'
                new_key = n.replace('.', '__')
                model.register_buffer('{}_mean'.format(new_key), p.data.clone())
                model.register_buffer('{}_fisher'.format(new_key), importances[n].data.clone())
        pass


def train(model, writer, train_loader, optimizer, criterion, epoch, task_idx, weight_mask_dict, curr_act_dict, ):
    model.zero_grad()
    model.train()

    t_mask = len(train_loader) * (args.epochs+1) * args.batch_size
    if args.ewc_lambda != 0:
        ewc_obj = EWC()
        if epoch == 1 and task_idx > 0:
            print(f'Consolidating FISHER: TASK: {task_idx}')
            ewc_obj.consolidate_fisher(model, criterion, optimizer, train_loader, num_samples=500)
        model.train()

    # weight_mask_dict, curr_act_dict = get_editable_weights_mask_dict(model, type=args.weight_mask_type)
    assert len(optimizer.param_groups) == 1
    for batch_idx, batch in enumerate(train_loader):
        if args.debug and batch_idx > 10: break
        if args.iter_lim < 0 or len(train_loader) * (epoch - 1) + batch_idx < args.iter_lim:
            optimizer.zero_grad()
            
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

            loss = criterion(output, target)
            if args.ewc_lambda != 0 and task_idx > 0:
                ewc_loss = ewc_obj.ewc_loss(model)
                total_loss = loss + (args.ewc_lambda / 2) * ewc_loss
                if args.verbose: print(f'Total: {total_loss:.6f}\tLoss: {loss:.6f}\tEWC Loss: {ewc_loss:.6f}')
            else:
                total_loss = loss
            total_loss.backward()

            old_sums = []
            total, curr_active, exclusive_active = 0, 0, 0    
            for n, m in model.named_modules():
                if hasattr(m, 'scores'):
                    assert (isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d)) and m.task == task_idx
                    for i in range(args.num_tasks): assert m.scores[i].grad is None
                    m.weight.grad.data.copy_(m.weight.grad.data * weight_mask_dict[n])
                    assert (m.weight.grad.data * ~weight_mask_dict[n]).sum() == 0, 'gradients are not masked properly!'
                    old_sums.append((m.weight * ~weight_mask_dict[n]).sum().cpu().detach().numpy().item())
                    total += m.weight.numel()
                    curr_active += curr_act_dict[n]
                    exclusive_active += weight_mask_dict[n].sum()
            optimizer.step()
            
            # can be removed, just checking that the masked weights dont get updated at all.
            new_sums = []
            for n, m in model.named_modules():
                if hasattr(m, 'scores'):
                    new_sums.append((m.weight * ~weight_mask_dict[n]).sum().cpu().detach().numpy().item())
            assert np.linalg.norm(np.array(old_sums) - np.array(new_sums)) == 0
            
            if batch_idx % args.log_interval == 0:
                num_samples = batch_idx * len(data)
                num_epochs = len(train_loader.dataset)
                percent_complete = 100.0 * batch_idx / len(train_loader)
                print(
                    f"Task:{task_idx}\tTrain Epoch:{epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                    f"Loss:{loss.item():.6f}"
                    # f"\nExlusive Sparse Overlap: {(curr_active-exclusive_active)/curr_active * 100:.8f}\n"
                )
                t = (len(train_loader) * epoch + batch_idx) * args.batch_size + t_mask
                writer.add_scalar(f"train/task_{task_idx}/loss", loss.item(), t)
                writer.add_scalar(f"train/task_{task_idx}/lr", optimizer.param_groups[0]['lr'], t)
            pass
    pass

def batch_evaluate(model, writer, criterion, data_loader, epoch, task_idx, split='Val'):
    model.zero_grad()
    model.eval()
    correct, batch_loss = 0, 0

    if split.lower() in ['val', 'validation']:
        loader = data_loader.val_loaders[task_idx]
    elif split.lower() in ['test']:
        loader = data_loader.test_loaders[task_idx]
    else:
        raise NotImplementedError(f'{split} not implemented')

    with torch.no_grad():
        for  batch_idx, batch in enumerate(loader):
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
            
            batch_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    batch_loss /= len(loader)
    batch_acc = float(correct) / len(loader.dataset)

    print(f"Train/{split} Task: {task_idx}\t{split} loss: {batch_loss:.4f}, {split} Accuracy: ({batch_acc:.4f})")
    epoch = epoch + args.epochs
    writer.add_scalar(f"task_train_{split.lower()}/task_{task_idx}/loss", batch_loss, epoch)
    writer.add_scalar(f"task_train_{split.lower()}/task_{task_idx}/acc", batch_acc, epoch)
    return batch_acc
