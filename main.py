import sys
import argparse
import os
import random
import numpy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import agents
import utils

numpy.set_printoptions(edgeitems=5, linewidth=160, formatter={'float': '{:0.6f}'.format})
torch.set_printoptions(edgeitems=5, precision=6, linewidth=160)
pd.options.display.float_format = '{:,.6f}'.format
pd.set_option('display.width', 160)

parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions (default: 1)')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-bs', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-lr', '--original-learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-lrdr', '--learning-rate-decay-rate', default=0.1, type=float, metavar='LRDR', help='learning rate decay rate')
parser.add_argument('-lrde', '--learning-rate-decay-epochs', default="150 200 250", metavar='LRDE', help='learning rate decay epochs')
parser.add_argument('-lrdp', '--learning-rate-decay-period', default=500, type=int, metavar='LRDP', help='learning rate decay period')
parser.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-wd', '--weight-decay', default=1*1e-4, type=float, metavar='W', help='weight decay (default: 1*1e-4)')
parser.add_argument('-pf', '--print-freq', default=1, type=int, metavar='N', help='print frequency (default: 1)')
parser.add_argument('-gpu', '--gpu-id', default='0', type=int, help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('-ei', '--exps-inputs', default="", type=str, metavar='PATHS', help='Inputs paths for the experiments')
parser.add_argument('-et', '--exps-types', default="", type=str, metavar='EXPERIMENTS', help='Experiments types to be performed')
parser.add_argument('-ec', '--exps-configs', default="", type=str, metavar='CONFIGS', help='Experiments configs to be used')
parser.add_argument('-sd', '--seed', default=42, type=int, metavar='N', help='Seed (default: 42)')

args = parser.parse_args()

args.exps_inputs = args.exps_inputs.split(":")
args.exps_types = args.exps_types.split(":")
args.exps_configs = args.exps_configs.split(":")
args.learning_rate_decay_epochs = [int(item) for item in args.learning_rate_decay_epochs.split()]

random.seed(args.seed)
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("seed", args.seed)

cudnn.benchmark = False
if args.executions == 1:
    cudnn.deterministic = True
    print("Deterministic!!!")
else:
    cudnn.deterministic = False
    print("No deterministic!!!")

torch.cuda.set_device(args.gpu_id)
print('\n__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__Number CUDA Devices:', torch.cuda.device_count())
print('Active CUDA Device: GPU', torch.cuda.current_device())


def main():
    print("\n\n\n\n\n\n")
    print("***************************************************************")
    print("***************************************************************")
    print("***************************************************************")
    print("***************************************************************")

    all_experiment_results = {}
    for args.exp_input in args.exps_inputs:
        for args.exp_type in args.exps_types:
            for args.exp_config in args.exps_configs:
                print("\n\n\n\n")
                print("***************************************************************")
                print("EXPERIMENT INPUT:", args.exp_input)
                print("EXPERIMENT TYPE:", args.exp_type)
                print("EXPERIMENT CONFIG:", args.exp_config)

                args.experiment_path = os.path.join("experiments", args.exp_input, args.exp_type, args.exp_config)

                if not os.path.exists(args.experiment_path):
                    os.makedirs(args.experiment_path)
                print("EXPERIMENT PATH:", args.experiment_path)

                args.executions_best_results_file_path = os.path.join(args.experiment_path, "results_best.csv")
                args.executions_raw_results_file_path = os.path.join(args.experiment_path, "results_raw.csv")

                for config in args.exp_config.split("+"):
                    config = config.split("~")
                    if config[0] == "data":
                        args.dataset = str(config[1])
                        print("DATASET:", args.dataset)
                    elif config[0] == "model":
                        args.model_name = str(config[1])
                        print("MODEL:", args.model_name)
                    elif config[0] == "loss":
                        args.loss = str(config[1])
                        print("LOSS:", args.loss)

                args.number_of_model_classes = None
                if args.dataset == "cifar10":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
                elif args.dataset == "cifar100":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 100
                elif args.dataset == "svhn":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10

                print("***************************************************************")
                for args.execution in range(1, args.executions + 1):
                    print("\n\n################ EXECUTION:", args.execution, "OF", args.executions, "################")
                    args.best_model_file_path = os.path.join(args.experiment_path, "model" + str(args.execution) + ".pth")

                    utils.save_dict_list_to_csv([vars(args)], args.experiment_path, args.exp_type+"_args")
                    print("\nARGUMENTS:", dict(utils.load_dict_list_from_csv(args.experiment_path, args.exp_type+"_args")[0]))

                    cnn_agent = agents.ClassifierAgent(args)
                    cnn_agent.train_classify()


                experiment_results = pd.read_csv(os.path.join(os.path.join(args.experiment_path, "results_best.csv")))
                print("\n################################\n", "EXPERIMENT RESULTS", "\n################################")
                print(args.experiment_path)
                print("\n", experiment_results.transpose())
                print("\n", experiment_results.describe())

                all_experiment_results[args.experiment_path] = experiment_results

    print("\n\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", "ALL EXPERIMENT RESULTS", "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    for key in all_experiment_results:
        print("\n", key)
        print("\n", all_experiment_results[key].transpose())
        print("\n", all_experiment_results[key].describe().reindex(['count', 'avg', 'std']))


if __name__ == '__main__':
    main()
