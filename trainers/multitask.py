from args import args
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def init(args):
    pass


def train(model, writer, data_loader, optimizer, criterion, epoch, task_idx):
    model.zero_grad()
    model.train()
    assert len(optimizer.param_groups) == 1

    for batch_idx, batch in enumerate(zip(*data_loader.train_loaders)):
        optimizer.zero_grad()

        all_data, all_target, all_mask = [], [], []
        for bat in batch:
            if args.text_exp:
                all_data.append(bat[0])
                all_mask.append(bat[1])
                all_target.append(bat[2])
            else:
                all_data.append(bat[0])
                all_target.append(bat[1])

        if args.text_exp:
            data, mask, target = torch.cat(all_data).to(args.device), torch.cat(all_mask).to(args.device), torch.cat(all_target).to(args.device)
            output = model(data, mask)
        else:
            data, target = torch.cat(all_data).to(args.device), torch.cat(all_target).to(args.device)
            output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            t = (min([len(dl) for dl in data_loader.train_loaders]) * epoch + batch_idx) * args.batch_size
            num_samples = batch_idx * len(data)
            num_epochs = min([len(dl) for dl in data_loader.train_loaders])
            percent_complete = 100.0 * batch_idx / min([len(dl) for dl in data_loader.train_loaders])
            print(
                f"Task:{task_idx}\tTrain Epoch:{epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                f"Loss:{loss.item():.6f}"
            )
            writer.add_scalar(f"train/task_{task_idx}/loss", loss.item(), t)
            writer.add_scalar(f"train/task_{task_idx}/lr", optimizer.param_groups[0]['lr'], t)


def batch_evaluate(model, writer, criterion, data_loader, epoch, task_idx, split='Val'):
    model.zero_grad()
    model.eval()

    if split.lower() in ['val', 'validation']:
        loader = data_loader.val_loaders
    elif split.lower() in ['test']:
        loader = data_loader.test_loaders
    else:
        raise NotImplementedError(f'{split} not implemented')

    with torch.no_grad():
        test_acc = np.zeros(args.num_tasks)
        for idx in range(args.num_tasks):
            correct, batch_loss = 0, 0  
            for batch_idx, batch in enumerate(loader[idx]):
                if args.text_exp:
                    data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                    output = model(data, mask)
                else:
                    data, target = batch[0].to(args.device), batch[1].to(args.device)
                    output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                batch_loss += criterion(output, target).item()

    batch_loss /= len(loader[idx])
    batch_acc = float(correct) / len(loader[idx].dataset)

    print(f"Train/{split} Task: {task_idx}\t{split} loss: {batch_loss:.4f}, {split} Accuracy: ({batch_acc:.4f})")
    writer.add_scalar(f"task_train_{split.lower()}/task_{task_idx}/loss", batch_loss, epoch)
    writer.add_scalar(f"task_train_{split.lower()}/task_{task_idx}/acc", batch_acc, epoch)
    return batch_acc

