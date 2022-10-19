from args import args

import torch
import torch.nn as nn


def init(args):
    pass


def train(model, writer, data_loader, optimizer, criterion, epoch, task_idx):
    model.zero_grad()
    model.train()
    
    # assert len(optimizer.param_groups) == 1
    for batch_idx, batch in enumerate(data_loader.train_loader):
        if args.debug and batch_idx > 10: break
        if args.iter_lim < 0 or len(data_loader.train_loader) * (epoch - 1) + batch_idx < args.iter_lim:
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
            loss.backward()
            optimizer.step()
            t = (len(data_loader.train_loader) * epoch + batch_idx) * args.batch_size
            if batch_idx % args.log_interval == 0:
                num_samples = batch_idx * len(data)
                num_epochs = len(data_loader.train_loader.dataset)
                percent_complete = 100.0 * batch_idx / len(data_loader.train_loader)
                print(
                    f"Task:{task_idx}\tTrain Epoch:{epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                    f"Loss:{loss.item():.6f}"
                )

                t = (len(data_loader.train_loader) * epoch + batch_idx) * args.batch_size
                writer.add_scalar(f"train/task_{task_idx}/loss", loss.item(), t)
                writer.add_scalar(f"train/task_{task_idx}/lr", optimizer.param_groups[0]['lr'], t)
                

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
        for batch_idx, batch in enumerate(loader):
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
    writer.add_scalar(f"task_train_{split.lower()}/task_{task_idx}/loss", batch_loss, epoch)
    writer.add_scalar(f"task_train_{split.lower()}/task_{task_idx}/acc", batch_acc, epoch)
    return batch_acc

