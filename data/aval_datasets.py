import torch, numpy as np, os
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, \
    RandomRotation, RandomCrop, RandomHorizontalFlip
import avalanche as avl
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from args import args


class AvalancheToPytorchDataset:
    def __init__(self, dataset='splitcifar100', return_task_id=True, val_size=0):
        self.dataset_name = args.dataname
        self.val_size = val_size
        self.curr_task = None

        self.kwargs = {"num_workers": args.workers, "pin_memory": args.pin_memory} if torch.cuda.is_available() else {}
        
        if args.dataname =='pmnist':
            self.dataset_classes = 10
            self.benchmark_original = avl.benchmarks.PermutedMNIST(n_experiences=args.num_tasks, seed=args.data_seed,
                                                                dataset_root=os.path.join(args.server_home, 'datasets/'))
        elif args.dataname =='rmnist':
            self.benchmark_original = avl.benchmarks.RotatedMNIST(n_experiences=args.num_tasks, seed=args.data_seed,
                                                                dataset_root=os.path.join(args.server_home, 'datasets/'))
        elif args.dataname =='smnist':
            self.dataset_classes = 10
            self.benchmark_original = avl.benchmarks.SplitMNIST(n_experiences=args.num_tasks, seed=args.data_seed, return_task_id=return_task_id,
                                                                dataset_root=os.path.join(args.server_home, 'datasets/'))            
        elif args.dataname =='splitcifar10':
            self.dataset_classes = 10
            self.benchmark_original = avl.benchmarks.SplitCIFAR10(n_experiences=args.num_tasks, seed=args.data_seed, 
                                                            return_task_id=return_task_id, fixed_class_order=np.arange(int(dataset.split('cifar')[-1])),
                                                            dataset_root=os.path.join(args.server_home, 'datasets/'))
        elif args.dataname =='splitcifar100':
            self.dataset_classes = 100
            self.benchmark_original = avl.benchmarks.SplitCIFAR100(n_experiences=args.num_tasks, seed=args.data_seed, 
                                                            return_task_id=return_task_id, fixed_class_order=np.arange(int(dataset.split('cifar')[-1])),
                                                            dataset_root=os.path.join(args.server_home, 'datasets/'))
        elif args.dataname =='splitcifar110':
            self.dataset_classes = 110
            self.benchmark_original = avl.benchmarks.SplitCIFAR110(n_experiences=args.num_tasks, seed=args.data_seed,
                                                         fixed_class_order=np.arange(int(dataset.split('cifar')[-1])),
                                                         dataset_root=os.path.join(args.server_home, 'datasets/'))
        elif args.dataname == 'tinyimagenet':
            self.dataset_classes = 200
            self.benchmark_original = avl.benchmarks.SplitTinyImageNet(n_experiences=args.num_tasks, seed=args.data_seed, return_task_id=True,
                                                        dataset_root=os.path.join(args.server_home, 'datasets/'))
        else:
            raise NotImplementedError(f"{dataset} Not implemented!")
            
        if val_size != 0:
            self.benchmark = benchmark_with_validation_stream(self.benchmark_original, validation_size=val_size, 
                                                            input_stream='train', output_stream='val')#, lazy_splitting=False)
        else:
            self.benchmark = self.benchmark_original

        self.tasks = list(range(args.num_tasks))
        if args.dataname in ['pmnist', 'rmnist']:
            self.task_classes = [self.dataset_classes] * args.num_tasks
            self.total_classes = sum(self.task_classes)
            assert self.total_classes == self.dataset_classes * args.num_tasks
            self.max_classes = max(self.task_classes)
        else:
            self.task_classes = [self.dataset_classes//args.num_tasks] * args.num_tasks
            self.total_classes = sum(self.task_classes)
            assert self.total_classes == self.dataset_classes
            self.max_classes = max(self.task_classes)
        # self.original_classes_in_exp = self.benchmark.original_classes_in_exp
        self.get_ds_and_dl_from_aval_benchmark()
        pass

    def get_ds_and_dl_from_aval_benchmark(self):
        self.train_datasets, self.train_loaders = self.get_ds_and_dl_from_aval_stream(self.benchmark.train_stream, shuffle=True)
        self.test_datasets, self.test_loaders = self.get_ds_and_dl_from_aval_stream(self.benchmark.test_stream, shuffle=False)
        if self.val_size == 0:
            self.val_datasets, self.val_loaders = self.test_datasets, self.test_loaders
        else:
            self.val_datasets, self.val_loaders = self.get_ds_and_dl_from_aval_stream(self.benchmark.val_stream, shuffle=False)

    def get_ds_and_dl_from_aval_stream(self, stream, shuffle):
        datasets, loaders = [], []
        for i in range(len(stream)):
            datasets.append(stream[i].dataset)
            loaders.append(torch.utils.data.DataLoader(stream[i].dataset, batch_size=args.batch_size, shuffle=shuffle, **self.kwargs))
        return datasets, loaders

    def update_task(self, i):
        self.curr_task = i
        args.num_classes = self.task_classes[i]
        self.train_loader = self.train_loaders[i]
        self.val_loader = self.val_loaders[i]
        self.test_loader = self.test_loaders[i]
        pass
