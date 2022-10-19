import torch
import numpy as np, os
from args import args
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from collections import defaultdict as ddict
from models import module_util
import torch.nn as nn
import torch.optim as optim
import wandb, random
import trainers
from pathlib import Path
import math


def knn_score_transfer(model, task_accs, curr_idx, type='knn_best'):
    best_idx = np.array([])
    if type == "knn_best":
        if task_accs.max() > 1/args.num_classes:
            best_idx = np.array([task_accs.argmax()])
    elif type == "knn_always":
        best_idx = np.array([task_accs.argmax()])
    elif type == "knn_all":
        best_idx = np.where(task_accs > 1/args.num_classes)[0]

    if best_idx.size == 0:
        print(f'No Good tasks found.')
    else:
        print(f"Type: {type}\tBest Index: {best_idx}\tBest Accs: {task_accs[best_idx]}")
        for n, m in model.named_modules():
            if hasattr(m, 'scores'):
                assert isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d)
                for i, idx in enumerate(best_idx):
                    if i == 0:
                        scores = m.scores[idx].data
                    else:
                        scores += m.scores[idx].data
                scores /= len(best_idx)
                if args.verbose: print(f"=> Copying scores from the Task:{best_idx} to Task:{curr_idx} for {n}")
                m.scores[curr_idx].data.copy_(scores)

def score_transfer(model, best_idxs, curr_idx):
    for n, m in model.named_modules():
        if hasattr(m, 'scores'):
            assert isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d)
            for i, idx in enumerate(best_idxs):
                if i == 0:
                    scores = m.scores[idx].data
                else:
                    scores += m.scores[idx].data
            scores /= len(best_idxs)
            if args.verbose: print(f"=> Copying scores from the Task:{best_idxs} to Task:{curr_idx} for {n}")
            m.scores[curr_idx].data.copy_(scores)


def findSimilarTasks(model, dataloader, num_tasks_learned, type, num_topk=100):
    current_task_idx = num_tasks_learned
    if 'knn' in type:
        acc_tasks = np.zeros(num_tasks_learned)
        for task_idx in range(num_tasks_learned):
            model.apply(lambda m: setattr(m, "task", task_idx))
            
            # # get features f_{task_id}^{task_id} via model forward pass.
            # base_reps, base_labels = get_reps(model, dataloader, task_idx)
            # get features f_{task_id}^{current_task_idx} via model forward pass.
            curr_reps, curr_labels = get_reps(model, dataloader, current_task_idx)
            train_x, train_y = curr_reps[:int(len(curr_reps)*0.66)], curr_labels[:int(len(curr_reps)*0.66)]
            test_x, test_y = curr_reps[int(len(curr_reps)*0.66):], curr_labels[int(len(curr_reps)*0.66):]
            NNC = KNN(X=train_x, Y=train_y, k=num_topk, p=2)
            test_pred_labels = NNC.predict(test_x)
            acc_tasks[task_idx] = test_y.eq(test_pred_labels).sum()/len(test_y)
        
        # best_task = acc_tasks.argmax()
        # best_acc = acc_tasks.max()
        # print(f'Best Task:{best_task}\tBest Acc: {best_acc:.4f}')
        return acc_tasks
    else:
        raise NotImplementedError(f"{type} siilarity measure is not implemented!")

def get_reps(model, dataloader, task_idx, num_sim_batches=100):
    dataloader.update_task(task_idx)
    num_sim_batches = math.ceil(len(dataloader.train_loader) * 0.2)
    reps = []
    labels = []
    for i, batch in enumerate(dataloader.train_loader):
        if i > num_sim_batches: 
            break
        if args.text_exp:
            train_x, mask, train_y = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
            if args.multigpu is None:
                reps.append(model.forward(train_x, mask, True).squeeze())
            else:
                reps.append(model.module.forward(train_x.cuda(), mask, True).squeeze().cpu().detach())
        else:
            train_x, train_y = batch[0].to(args.device), batch[1].to(args.device)
            if args.multigpu is None:
                reps.append(model.forward(train_x, True).squeeze())
            else:
                reps.append(model.module.forward(train_x.cuda(), True).squeeze().cpu().detach())
        labels.append(train_y)
    
    reps = torch.vstack(reps)
    labels = torch.hstack(labels)
    return reps, labels

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, p).sum(2)
    
    return dist

class KNN:
    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.p = p
        self.k = k
        self.train(X, Y)
    
    def __call__(self, x):
        return self.predict(x)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]
        return torch.mode(votes, dim=1).values


class getOverlap:
    def __init__(self, model: nn.Module, num_tasks):
        self.model = model
        self.num_tasks = num_tasks
        self.get_masks()
        self.seq_compute_overlap()

    def get_masks(self):
        print(f"\nGetting masks for model")
        self.task_masks = ddict(lambda: ddict(lambda: {}))
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                for ti in range(self.num_tasks):
                    self.task_masks[ti][n] = module_util.get_subnet(m.scores[ti].abs(), m.sparsity).type(torch.bool).detach()
    
    def compute_overlap(self):
        self.mask_sparse_overlap = np.zeros((self.num_tasks, self.num_tasks))
        self.mask_total_overlap = np.zeros((self.num_tasks, self.num_tasks))

        for t1 in range(self.num_tasks):
            for t2 in range(self.num_tasks):
                if t1 == t2:
                    self.mask_sparse_overlap[t1,t2] = 0
                    self.mask_total_overlap[t1,t2] = 0
                    break
                total = 0
                avg_non_zero = 0
                common = 0
                for k in self.task_masks[t1].keys():
                    overlap = torch.logical_and(self.task_masks[t1][k], self.task_masks[t2][k])
                    common += overlap.sum().item()
                    total += overlap.numel()
                    avg_non_zero += (torch.count_nonzero(self.task_masks[t1][k]).item() + torch.count_nonzero(self.task_masks[t2][k]).item()) / 2
                self.mask_sparse_overlap[t1,t2] = common/avg_non_zero
                self.mask_total_overlap[t1,t2] = common/total

        self.avg_sparse_overlap = 100 * self.mask_sparse_overlap.sum() / (self.num_tasks**2 - self.num_tasks)
        self.avg_total_overlap = 100 * self.mask_total_overlap.sum() / (self.num_tasks**2 - self.num_tasks)

    def seq_compute_overlap(self):
        self.mask_sparse_overlap = np.zeros((self.num_tasks))
        self.mask_total_overlap = np.zeros((self.num_tasks))

        for t1 in range(self.num_tasks):
            if t1 == 0: continue
            total_weight = 0
            total_mask = 0
            common = 0
            for k in self.task_masks[t1].keys():
                active = None
                for t2 in range(t1):
                    if active is None:
                        active = self.task_masks[t2][k]
                    else:
                        active = active | self.task_masks[t2][k]
                overlap = self.task_masks[t1][k] & active
                common += overlap.sum().item()
                total_weight += overlap.numel()
                total_mask += self.task_masks[t2][k].sum().item()

            self.mask_sparse_overlap[t1] = common/total_mask * 100
            self.mask_total_overlap[t1] = common/total_weight * 100

        self.avg_sparse_overlap = self.mask_sparse_overlap.mean()
        self.avg_total_overlap = self.mask_total_overlap.mean()


def get_optim_and_scheduler(model, optim_type, idx):
    # Clear the grad on all the parameters.
    for p in model.parameters():
        p.grad = None
        p.requires_grad = False

    # Set gradients for required parameter to be true.
    for n, p in model.named_parameters():
        if (idx >= 0 ) and f"scores.{idx}" in n and not args.base_type!="":
            if args.verbose: print(n)
            p.requires_grad = True
        elif (args.weight_epochs > 0 and idx < 0) or args.base_type!="":
            if "weight" in n or "bias" in n:
                if args.text_exp and "cnn_model" not in n:
                    continue
                if args.verbose: print(n)
                p.requires_grad = True

    # lr = ( args.train_weight_lr if idx < 0 else args.lr )
    lr = ( args.train_weight_lr if args.weight_epochs > 0 else args.lr )
    # get optimizer, scheduler
    print(f"Using {optim_type} Optimizer")
    wd = 0 if args.base_type=="" else args.wd
    # mom = 0 if args.weight_epochs > 0 else args.momentum
    mom = args.momentum
    # wd = args.wd

    if optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD( model.parameters(), lr=lr, momentum=mom, weight_decay=wd )
    
    train_epochs = args.weight_epochs if (idx < 0 or args.base_type!="") else args.epochs
    if args.no_scheduler:
        scheduler = None
    else:
        scheduler = [ReduceLROnPlateau(optimizer), CosineAnnealingLR(optimizer, T_max=train_epochs)]
        # scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)
    print(f"LEARNING RATE: {lr}")
    return optimizer, scheduler, train_epochs

def get_train_test_function(trainer):
    trainer = getattr(trainers, trainer)
    if args.verbose: print(f"=> Using trainer {trainer}")
    train, batch_evaluate = trainer.train, trainer.batch_evaluate
    # Initialize model specific context (editorial note: avoids polluting main file)
    if hasattr(trainer, "init"):
        trainer.init(args)
    return train, batch_evaluate

def setup_wandb_logger(args):
    wandb_writer = wandb.init(project=args.project_name, save_code=False, name=args.name, config=args)#, group=args.group)

    src_dir = Path(__file__).resolve().parent
    base_path = str(src_dir.parent)
    src_dir = str(src_dir)
    return wandb_writer, src_dir, base_path


def load_best_training(model, filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(
            filepath, map_location=f"cuda:{args.multigpu[0]}"
        )
        model_dict = model.state_dict()
        pretrained_dict = checkpoint["state_dict"]
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        # best_mask_acc = checkpoint["best_mask_acc"]
        if args.verbose: print(f"=> Loaded checkpoint with accs")
    return model


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False