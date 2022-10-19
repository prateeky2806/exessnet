import os, sys, platform

import pathlib, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from args import args
import trainers.adaptors as adaptors
from models.gemresnet import GEMResNet18
from models.resnet import ResNet50
from models.small import LeNet
import trainers
from utils import utils
from utils.metrics import get_forgetting_metric
from utils import my_utils
from data.aval_datasets import AvalancheToPytorchDataset

def main():
    before = time.time()
    if args.seed is None:
        args.seed = np.random.randint(10000)
    print(f"SETTING SEED TO {args.seed}")
    my_utils.seed_everything(args.seed)

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}~try={str(i)}")
        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"-{platform.node().split('.')[0]}~try={i}"
            break
        i += 1
    
    args.project_name = args.log_dir.split('/')[-1]
    f = open(f'{run_base_dir}/logs.txt', 'w')
    sys.stdout = my_utils.Tee(sys.stdout, f)

    (run_base_dir / "settings.txt").write_text(str(args))
    args.run_base_dir = run_base_dir
    print(f"=> Saving data in {run_base_dir}")
    print(args)
    data_loader = AvalancheToPytorchDataset(dataset=args.dataname)
    epoch_evaluate = getattr(adaptors, 'epoch_evaluate')

    # Track accuracy on all tasks.
    best_mask_acc = [0.0 for _ in range(args.num_tasks)]
    last_acc = [0.0 for _ in range(args.num_tasks)]
    all_test_acc = np.zeros((args.num_tasks,args.num_tasks))

    # Get the model.
    # model = utils.get_model(args.text_exp, data_loader.max_classes)
    model_classes = data_loader.total_classes if args.strategy=='joint' else data_loader.max_classes
    if 'cifar' in args.dataname:
        print(f"Inititalizing Resnet18 model with {model_classes} classes!")
        model = GEMResNet18(num_classes=model_classes)
    elif 'imagenet' in args.dataname:
        print(f"Inititalizing Resnet50 model with {model_classes} classes!")
        model = ResNet50(num_classes=model_classes)
    elif 'mnist' in args.dataname:
        print(f"Inititalizing LeNet model with {model_classes} classes!")
        model = LeNet(num_classes=model_classes)
    else:
        raise NotImplementedError("")
    
    # If necessary, set the sparsity of the model of the model using the ER sparsity budget (see paper).
    if args.er_sparsity != 'no':
        for n, m in model.named_modules():
            if hasattr(m, "sparsity"):
                if args.er_sparsity == 'normal':
                    sp = args.sparsity
                elif args.er_sparsity == 'er':
                    sp = args.sparsity * (m.weight.size(0) + m.weight.size(1)) / ( m.weight.size(0) * m.weight.size(1) * m.weight.size(2) * m.weight.size(3) )
                m.sparsity = sp
                if args.verbose: print(f"Set sparsity of {n} to {m.sparsity}")
    
    # Put the model on the GPU.
    model = utils.set_gpu(model)
    criterion = nn.CrossEntropyLoss().to(args.device)
    writer = SummaryWriter(log_dir=run_base_dir)

    num_tasks_learned = 0
    # Iterate through all tasks for training.
    for curr_idx in range(args.num_tasks or 0):
        print("\nTRAINING FOR MASKS\n")
        if curr_idx > 0:
            # Load best Checkpoint of the Last Task.
            model = my_utils.load_best_training(model, run_base_dir / "local_best.pt")
        
        ## Update Dataloader and Task ##
        print(f"Task {args.set}: {curr_idx}")
        model.apply(lambda m: setattr(m, "task", curr_idx))
        assert hasattr(data_loader, "update_task" ), "[ERROR] Need to implement update task method for use with multitask experiments"
        data_loader.update_task(curr_idx)
        
        # Train for masks.
        if args.epochs > 0:
            train, batch_evaluate = my_utils.get_train_test_function("default")
            if curr_idx != 0 and curr_idx != args.num_tasks and args.sim_init != "":
                if "knn" in args.sim_init:
                    print('Performing KNN classification to find similar tasks!')
                    task_accs = my_utils.findSimilarTasks(model, data_loader, num_tasks_learned, type=args.sim_init, num_topk=args.num_topk)
                    print(f"task accs: {task_accs}")
                    best_indices = np.array([])
                    if args.sim_init == "knn_best":
                        if task_accs.max() > 1/data_loader.max_classes:
                            best_indices = np.array([task_accs.argmax()])
                    elif args.sim_init == "knn_always":
                        best_indices = np.array([task_accs.argmax()])
                    elif args.sim_init == "knn_all":
                        best_indices = np.where(task_accs > 1/data_loader.max_classes)[0]

                if best_indices.size == 0:
                    print(f'No Good tasks found.')
                else:
                    print(f"Type: {type}\tBest Index: {best_indices}")
                    my_utils.score_transfer(model, best_indices, curr_idx)
                model.apply(lambda m: setattr(m, "task", curr_idx))
                data_loader.update_task(curr_idx)
            else:
                print("No Similarity based initialization.")
            
            optimizer, scheduler, train_epochs = my_utils.get_optim_and_scheduler(model, optim_type=args.mask_opt, idx=curr_idx)

            # Train on the scores for current task.
            for epoch in range(1, train_epochs + 1):
                print('\n')
                train(model, writer, data_loader, optimizer, criterion, epoch, curr_idx,)
                utils.cache_weights(model, num_tasks_learned + 1)
                last_acc[curr_idx] = batch_evaluate( model, writer, criterion, data_loader, epoch, curr_idx, split='Val')
                if last_acc[curr_idx] > best_mask_acc[curr_idx]:
                    best_mask_acc[curr_idx] = last_acc[curr_idx]
                    torch.save( { "epoch": args.epochs, "arch": args.model, "state_dict": model.state_dict(), "best_mask_acc": best_mask_acc,
                        "last_acc": last_acc, "args": args, }, run_base_dir / "local_best.pt",)

                if scheduler:
                    scheduler[1].step()
                    scheduler[0].step(last_acc[curr_idx])
                if ( args.iter_lim > 0 and len(data_loader.train_loader) * epoch > args.iter_lim ):
                    break
            # caching masks and deleting optimizer and schedulers
            utils.cache_masks(model)
            del optimizer, scheduler

        # Train on the weights for current task.
        if args.weight_epochs > 0:
            print("\nTRAINING FOR WEIGHTS\n")
            optimizer, scheduler, train_epochs = my_utils.get_optim_and_scheduler(model, optim_type=args.weight_opt, idx=-1)
            train, batch_evaluate = my_utils.get_train_test_function('weights')
            get_editable_weights_mask_dict = getattr(trainers, "weights").get_editable_weights_mask_dict
            weight_mask_dict, curr_act_dict = get_editable_weights_mask_dict(model, type=args.weight_mask_type)
                
            for weight_epoch in range(1, args.weight_epochs+1):
                train( model, writer, data_loader.train_loader, optimizer, criterion, weight_epoch, curr_idx, weight_mask_dict, curr_act_dict)
                print('\n')
                last_acc[curr_idx] = batch_evaluate( model, writer, criterion, data_loader, weight_epoch, curr_idx, split='Val' )
                if last_acc[curr_idx] > best_mask_acc[curr_idx]:
                    best_mask_acc[curr_idx] = last_acc[curr_idx]
                    torch.save( { "epoch": args.epochs, "arch": args.model, "state_dict": model.state_dict(), "best_mask_acc": best_mask_acc,
                     "last_acc": last_acc, "args": args, }, run_base_dir / "local_best.pt",)
                if scheduler:
                    scheduler[1].step()
                    scheduler[0].step(last_acc[curr_idx])

            del optimizer, scheduler
        num_tasks_learned += 1
        model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks_learned))

        # EVALUTATION ON ALL TASKS!
        print('EPOCH END EVALUATION')
        if num_tasks_learned in args.eval_ckpts or num_tasks_learned == args.num_tasks or args.eval_all:
            # Evaluate until current task + 1 if not the last task
            eval_tasks = (num_tasks_learned + 1) if curr_idx < args.num_tasks-1 else num_tasks_learned
            for test_idx in range(eval_tasks):
                for p in model.parameters(): p.grad = None
                for b in model.buffers(): b.grad = None
                all_test_acc[curr_idx, test_idx] = epoch_evaluate(model, data_loader, in_task=test_idx, out_task=test_idx, split='Test')
                writer.add_scalar(f"task_test/acc_{test_idx}", all_test_acc[curr_idx, test_idx], curr_idx)
            print(f"Adapt Test Accuracies: {all_test_acc[curr_idx,:]}")
            print(f"Average Test Accuracy: {all_test_acc[curr_idx, :num_tasks_learned].mean():.4f}")
            utils.clear_masks(model)
            torch.cuda.empty_cache()


    # printing stuff to console
    if args.num_tasks > 1:
        test_forgetting = get_forgetting_metric(all_test_acc, bwt=True)
        print(f'Test Forgetting: {test_forgetting.tolist()}')
        print(f'Average Test Forgetting: {test_forgetting.mean():.4f}')
    
        overlap_obj = my_utils.getOverlap(model, args.num_tasks)
        print(f"Sparse Overalp: {overlap_obj.mask_sparse_overlap.tolist()}")
        print(f"Total Overalp: {overlap_obj.mask_total_overlap.tolist()}")
        print(f"Avg. Sparse Overalp: {overlap_obj.avg_sparse_overlap:.8f}")
        print(f"Avg. Total Overalp: {overlap_obj.avg_sparse_overlap:.8f}")
    
    print( f"Finished experiment in {str((time.time() - before) / 60.0)} minutes." )
    return all_test_acc


if __name__ == "__main__":
    main()
    pass
