import utils.parser as _parser
import platform
import argparse
import sys
import yaml

args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="ExSSNeT")

    # Dataloaders
    parser.add_argument("--set", type=str, help="Which dataset to use")
    parser.add_argument( "--data-seed", type=int, default=0, metavar="S", help="random seed (default: 1)" )
    parser.add_argument( "--debug", action="store_true", default=False,)
    parser.add_argument( "--server-home", required=True, type=str, help="Home folder for the server", )
    parser.add_argument( "--data", default='./data', type=str, help="Location to store data", )
    parser.add_argument( "--num-tasks", default=None, type=int, help="Number of tasks, None if no adaptation is necessary", )
    parser.add_argument( "--num-classes", default=None, type=int, help="Num Classes", )

    parser.add_argument( "--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 64)", )
    parser.add_argument( "--test-batch-size", type=int, default=128, metavar="N", help="input batch size for testing (default: 128)", )
    parser.add_argument("--workers", type=int, default=4, help="how many cpu workers")
    parser.add_argument("--pin-memory", action="store_false", default=True)
    

    # model params
    parser.add_argument("--model", type=str, help="Type of model.")
    
    parser.add_argument( "--conv_type", type=str, default="StandardConv", help="Type of conv layer" )
    parser.add_argument( "--bn_type", type=str, default="StandardBN", help="Type of batch norm layer." )
    parser.add_argument( "--conv-init", type=str, default="default", help="How to initialize the conv weights.", )
    
    # parser.add_argument( "--output-size", type=int, default=10, help="how many total neurons in last layer", )
    parser.add_argument( "--width-mult", type=float, default=1.0, help="how wide is each layer" )
    
    parser.add_argument( "--sparsity", type=float, default=0.1, help="how sparse is each layer, when using MultitaskMaskConv" )
    parser.add_argument("--er-sparsity", type=str, default='no')
    
    parser.add_argument( "--nonlinearity", default="relu", help="Nonlinearity used by initialization" )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")

    # Training
    parser.add_argument( "--trainer", default=None, type=str, help="Which trainer to use, default in trainers/default.py", )
    parser.add_argument( "--config", type=str, default=None, help="Config file to use, YAML format" )
    parser.add_argument( "--epochs", type=int, default=100, metavar="N", help="number of epochs to train (default: 100)", )
    parser.add_argument( "--iter-lim", default=-1, type=int, )
    parser.add_argument( "--save", action="store_true", default=False, help="save checkpoints" )
    parser.add_argument("--resume", type=str, default=None, help='optionally resume')
    parser.add_argument( "--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")], help="Which GPUs to use for multigpu training", )

    
    # Evaluation
    parser.add_argument( "--adaptor", default="gt", help="Which adaptor to use, see adaptors.py", )
    parser.add_argument( "--eval-all", action="store_true", default=False, help="Evaluate on all task for all checkpoints.")
    parser.add_argument(
        "--eval-ckpts",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="After learning n tasks for n in eval_ckpts we perform evaluation on all tasks learned so far",
    )

    
    # optimization
    parser.add_argument( "--mask_opt", type=str, default="adam", help="Which optimizer to use for masks" )
    parser.add_argument( "--weight_opt", type=str, default="adam", help="Which optimizer to use for weights" )
    parser.add_argument( "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)", )
    parser.add_argument("--lr-policy", default=None, help="Scheduler to use")
    parser.add_argument("--no-scheduler", action="store_true", help="constant LR")
    parser.add_argument( "--momentum", type=float, default=0.9, metavar="M", help="Momentum (default: 0.9)", )
    parser.add_argument( "--wd", type=float, default=0.0001, metavar="M", help="Weight decay (default: 0.0001)", )

    
    # weight training
    parser.add_argument( "--weight-epochs", type=int, default=0, metavar="N", help="number of epochs to train weights (default: 0)", )
    parser.add_argument( "--weight-mask-type", type=str, default="original", help="Mask type specifying which weights are updated." )
    parser.add_argument( "--ewc-lambda", type=float, default=0, metavar="LR", help="loss mixing coefficients.", )
    parser.add_argument( "--train-weight-lr", default=0.0001, type=float, help="While training the weights, which LR to use.", )
    
    
    # Conditioning mode
    parser.add_argument("--sim-init", type=str, default="")
    parser.add_argument("--num-topk", type=int, default=200)
    

    # Text model params
    parser.add_argument( "--text-exp", action="store_true", default=False,)
    parser.add_argument("--text-tasks", default=None, type=lambda x: [str(a) for a in x.split(",")], help="Tasks for the text setup",)
    parser.add_argument( "--emb-model", type=str, default='glove', help="" )
    parser.add_argument( "--cnn-model", type=str, default='cnnstatic', help="" )
    parser.add_argument( "--vdcnn-depth", type=int, default=9, help="" )
    parser.add_argument( "--max-length", type=int, default=256, help="" )
    parser.add_argument( "--lin-hidden", type=int, default=2048, help="" )
    parser.add_argument( "--train-class-size", type=int, default=2000, help="" )
    parser.add_argument( "--val-class-size", type=int, default=2000, help="" )
    parser.add_argument( "--num-filters", type=int, default=200, help="" )

    # Baseline flag
    ### Avalanche args
    parser.add_argument( "--dataname", type=str, default='splitcifar100', help="Dataset from avalanche")
    parser.add_argument( "--scenario", type=str, default='custom', help="")
    parser.add_argument( "--base_type", type=str, default='', help="Type of baseline experiments")
    parser.add_argument( "--strategy", type=str, default='', help="Avalanche strategy.")
    parser.add_argument( "--ewc_mode", type=str, default='online', help="EWC Mode.")
    parser.add_argument( "--mem-size", type=int, default=None, help="Memory Size of dataset Size.")
    parser.add_argument( "--mem-frac", type=float, default=0.1, help="Memory fraction of dataset Size.")
    parser.add_argument( "--lwf_alpha", type=float, default=0.1, help="Penalty hyperparameter for LwF.", )
    parser.add_argument( "--si-eps", type=float, default=0.001, help="SI EPS", )
    parser.add_argument( "--temperature", default=1, type=int)
    parser.add_argument( "--patterns_per_exp", default=256, type=float, help='agem, gem')
    parser.add_argument( "--memory_strength", default=0.5, type=float, help='gem')
    parser.add_argument( "--sample_size", default=256, type=float, help='agem')
    parser.add_argument( "--si_lambda", default=0.0001, type=float, help='synaptic intelligence.')
    
    
    # reproducibility and logging
    parser.add_argument( "--seed", type=int, default=None, metavar="S", help="random seed (default: 1)" )
    parser.add_argument( "--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status", )
    parser.add_argument("--name", type=str, default="default", help="Experiment id.")
    parser.add_argument( "--log-dir", default='.', type=str, help="Location to logs", )
    parser.add_argument( "--verbose", action="store_true", default=False,)

    parser.add_argument( "--project-name", type=str, help="Wandb project name", )
    parser.add_argument( "--group", default='exp1', type=str, help="experiment groups.", )
    parser.add_argument( "--check-dir", default='./checkpoint', type=str, help="Location to checkpoints", )


    

    # miscellaneous    
    parser.add_argument( "--warmup-length", default=0, type=int, )
    parser.add_argument( "-f", "--dummy", default=None, help="Dummy to use for ipython notebook compatibility", )

    # # Used for BatchE 
    # parser.add_argument( "--train-weight-tasks", type=int, default=0, metavar="N", help="number of tasks to train the weights, e.g. 1 for batchensembles. -1 for all tasks", )
    parser.add_argument( "--individual-heads", action="store_true", help="Seperate head for each batch_ensembles task!", )


    # #used for other adaptors
    # parser.add_argument( "--real-neurons", type=int, default=10, help="how many real neurons" )
    # parser.add_argument("--gamma", type=float, default=0.0)
    # parser.add_argument( "--hop-weight", type=float, default=1e-3, help="how wide is each layer" )
    # parser.add_argument( "--log-base", default=2, type=int, help="keep the bottom 1/log_base elements during binary optimization", )
    # parser.add_argument( "--reinit-most-recent-k", default=None, type=int, help="Whether or not to include a memory buffer for reinit training. Currently only works with binary reinit_adaptor", )
    # parser.add_argument( "--reinit-adapt", type=str, default="binary", help="Adaptor for reinitialization experiments", ) #used for other adaptors
    # parser.add_argument( "--data-to-repeat", default=1, type=int, ) #used for other adaptors
    # parser.add_argument( "--unshared_labels", action="store_true", default=False, ) #used for other adaptors
    
    # parser.add_argument( "--ortho-group", action="store_true", default=False, ) # Used for hopfield recovery
    # parser.add_argument( "--replace-task-always", action="store_true", default=False, )
    
    args = parser.parse_args()

    # Allow for use from notebook without config file
    if args.config is not None:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
