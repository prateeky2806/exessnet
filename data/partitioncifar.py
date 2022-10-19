import numpy as np
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import copy

from args import args
import utils.utils as utils


def partition_dataset(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label - torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionCIFAR10:
    def __init__(self):
        super(PartitionCIFAR10, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_dataset(train_dataset, 2 * i),
                partition_dataset(val_dataset, 2 * i),
            )
            for i in range(5)
        ]

        for i in range(5):
            print()
            print(f"=> Size of train split {i}: {len(splits[i][0].data)}")
            print(f"=> Size of val split {i}: {len(splits[i][1].data)}")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


def partition_datasetv2(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionCIFAR10V2:
    def __init__(self):
        super(PartitionCIFAR10V2, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_datasetv2(train_dataset, 2 * i),
                partition_datasetv2(val_dataset, 2 * i),
            )
            for i in range(5)
        ]

        for i in range(5):
            print(len(splits[i][0].data))
            print(len(splits[i][1].data))
            print("==")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


def partition_datasetv3(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i)
        or label == torch.tensor(i + 1)
        or label == torch.tensor(i + 2)
        or label == torch.tensor(i + 3)
        or label == torch.tensor(i + 4)
    ]

    newdataset.targets = [
        label - torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i)
        or label == torch.tensor(i + 1)
        or label == torch.tensor(i + 2)
        or label == torch.tensor(i + 3)
        or label == torch.tensor(i + 4)
    ]
    return newdataset


class PartitionCIFAR100V2:
    def __init__(self):
        super(PartitionCIFAR100V2, self).__init__()
        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_datasetv3(train_dataset, 5 * i),
                partition_datasetv3(val_dataset, 5 * i),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]




def partition_datasetv4(dataset, perm):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]

    newdataset.targets = [
        lperm.index(label)
        for label in newdataset.targets
        if label in lperm
    ]
    return newdataset

class RandSplitCIFAR100:
    def __init__(self):
        super(RandSplitCIFAR100, self).__init__()
        data_root = os.path.join(args.data, "cifar100")
        self.curr_task = None
        use_cuda = torch.cuda.is_available()
        self.tasks = list(range(args.num_tasks))
        self.task_classes = [args.num_classes] * args.num_tasks
        self.total_classes = sum(self.task_classes)
        self.max_classes = max(self.task_classes)

        normalize = transforms.Normalize( mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262] )
        train_transform = transforms.Compose( [ transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize, ])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        train_dataset = datasets.CIFAR100( root=data_root, train=True, download=True, transform=train_transform )
        val_dataset = datasets.CIFAR100( root=data_root, train=True, download=True, transform=test_transform )
        test_dataset = datasets.CIFAR100( root=data_root, train=False, download=True, transform=test_transform, )

        np.random.seed(args.data_seed)
        perm = np.random.permutation(100)
        if args.verbose: print(perm)

        tds = [partition_datasetv4(train_dataset, perm[self.task_classes[i] * i:self.task_classes[i] * (i+1)]) for i in range(args.num_tasks) ]
        vds = [partition_datasetv4(val_dataset, perm[self.task_classes[i] * i:self.task_classes[i] * (i+1)]) for i in range(args.num_tasks) ]
        self.test_datasets = [partition_datasetv4(test_dataset, perm[self.task_classes[i] * i:self.task_classes[i] * (i+1)]) for i in range(args.num_tasks) ]
        
        self.train_datasets = []
        self.val_datasets = []
        for tdset, vdset in zip(tds, vds):
            train_len = int(len(tdset)*0.8)
            train_idx = list(range(len(tdset)))[:train_len]
            val_idx = list(range(len(tdset)))[train_len:]
            self.train_datasets.append(Subset(tdset, train_idx))
            self.val_datasets.append(Subset(vdset, val_idx))

        for i in range(args.num_tasks):
            if args.verbose: print(perm[self.task_classes[i] * i:self.task_classes[i] * (i+1)]) 

        self.train_probs, self.train_batch_sizes = self.get_batch_sizes(self.train_datasets)
        self.val_probs, self.val_batch_sizes = self.get_batch_sizes(self.val_datasets)
        self.test_probs, self.test_batch_sizes = self.get_batch_sizes(self.test_datasets)

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        if args.base_type == 'multitask':
            kwargs['pin_memory'] = False

        if args.base_type=='multitask':
            self.train_loaders = [DataLoader( x, batch_size=int(self.train_batch_sizes[i]), shuffle=True, drop_last=False, **kwargs ) for i, x in enumerate(self.train_datasets) ]
            self.val_loaders = [DataLoader( x, batch_size=int(self.val_batch_sizes[i]), shuffle=False, drop_last=False, **kwargs ) for i, x in enumerate(self.val_datasets) ]
            self.test_loaders = [DataLoader( x, batch_size=int(self.test_batch_sizes[i]), shuffle=False, drop_last=False, **kwargs ) for i, x in enumerate(self.test_datasets) ]
        else:
            self.train_loaders = [DataLoader( x, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs ) for x in self.train_datasets ]
            self.val_loaders = [DataLoader( x, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs ) for x in self.val_datasets ]
            self.test_loaders = [DataLoader( x, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs ) for x in self.test_datasets ]
        
    def update_task(self, i):
        self.curr_task = i
        args.num_classes = self.task_classes[i]
        self.train_loader = self.train_loaders[i]
        self.val_loader = self.val_loaders[i]
        self.test_loader = self.test_loaders[i]

    def get_batch_sizes(self, datasets):
        datasets_len = np.array([len(d) for d in datasets])
        probs = datasets_len/datasets_len.sum(keepdims=True)
        batch_sizes = np.ceil(args.batch_size * probs).astype(int)
        return probs, batch_sizes