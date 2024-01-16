import os
import sys
import random
import pickle
import logging
import argparse

import torch
import wandb
import numpy as np

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data_preprocess")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../model")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../model")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))


from data_preprocess.mnist.data_loader import distributed_pmnist
from data_preprocess.cifar100.data_loader import distributed_cifar100
from data_preprocess.five_datasets.data_loader import distributed_5datasets
from data_preprocess.mini_imagenet.data_loader import distributed_imagenet

from resnet import get_resnet18
from mlp import get_MLP
from alexnet import get_alexnet
from resnet_imagenet import get_resnet_18

from trainer.cifar_trainer import CifarTrainer
from trainer.mnist_trainer import MNISTTrainer
from trainer.resnet_trainer import ResnetTrainer
from trainer.imagenet_trainer import ImagenetTrainer

from FedML.fedml_api.distributed.fedavg_seq_cont.FedAvgSeqAPI import (
    FedML_init,
    FedML_FedAvgSeq_distributed,
)

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="audio_conv_rnn",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="gcommand",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir", type=str, default="../data/", help="data directory"
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=2118,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=10,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument("--gpu_worker_num", type=int, default=8, help="total gpu num")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--backend", type=str, default="MPI", help="Backend for Server and Client"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--fl_algorithm",
        type=str,
        default="FedAvgSeq",
        help="Algorithm list: FedAvg; FedOPT; FedProx; FedAvgSeq ",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=30,
        help="how many round of communications we should use",
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=1,
        metavar="mu",
        help="variable for FedProx",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=0,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=10,
        help="the frequency of the algorithms",
    )

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument(
        "--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server"
    )

    parser.add_argument("--starting_gpu", type=int, default=0)

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default=None,
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key",
        type=str,
        default="mapping_default",
        help="the key in gpu utilization file",
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )


    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help='server_optimizer')

    parser.add_argument('--server_lr', type=float, default=0.001,
                        help='server_lr')


    #Arguments that we added
    parser.add_argument( "--algo", type=str, default="normal",  
                        help="Algorithm list: normal, orth", )
        
    parser.add_argument('--num_task', type=int, default=10,
                    help='Number od tasks')

    parser.add_argument('--epsilon', type=float, default=0.98,  help='Epsilon for orth training')
    parser.add_argument('--eps_inc', type=float, default=0.0,
                    help='How much epsilon will increase after each task')

    parser.add_argument('--iid', type=int, default=1,
                    help='1: iid dist, 0: non-iid dist')

    parser.add_argument('--test_seed', type=int, default=0,
                    help='Seed')
    
    parser.add_argument('--task_pct_per_client', type=float, default=1.0,
                    help='Average pertange of total tasks each client see. 1.0 everyone sees aevery task, \
                    0.5 each client see 50 percent of tasks on average')
                


    args = parser.parse_args()
    return args



def load_data(args, dataset_name):

    if dataset_name == "mnist_10ts":
        data_loader = distributed_pmnist

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
            available_clients_per_ts,
        ) = data_loader(
            task_num=10,
            num_clients=args.client_num_in_total,
            batch_size=args.batch_size, 
            iid=args.iid==1,
            task_perc=args.task_pct_per_client,
            seed = args.test_seed,
        )
    elif dataset_name == "splitCifar100":  
        data_loader = distributed_cifar100

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
            available_clients_per_ts,
        ) = data_loader(
            batch_size=args.batch_size, 
            task_num=args.num_task,
            num_client=args.client_num_in_total, 
            iid=args.iid==1,
            task_perc=args.task_pct_per_client,
            seed = args.test_seed,
        )

    elif dataset_name == "imagenet": 
        data_loader = distributed_imagenet

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
            available_clients_per_ts,
        ) = data_loader(
            batch_size=args.batch_size, 
            task_num=args.num_task,
            num_client=args.client_num_in_total, 
            iid=args.iid==1,
            task_perc=args.task_pct_per_client,
            seed = args.test_seed,
        )

    elif dataset_name == "five_datasets":
        data_loader = distributed_5datasets

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
            available_clients_per_ts,
        ) = data_loader(
            num_client=args.client_num_in_total,
            batch_size=args.batch_size, 
            iid=args.iid==1,
            task_perc=args.task_pct_per_client,
            seed = args.test_seed,
        )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        np.array(test_data_global),
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    logging.info("finish data loading")
    return dataset, available_clients_per_ts


def create_model(args, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (args.model, output_dim)
    )
    model = None
    if args.model == "resnet":
        model = get_resnet18(5, output_dim)
    elif args.model == "mlp":
        model = get_MLP(args=args)
    elif args.model == "alexnet":
        model = get_alexnet(args.num_task, output_dim, args=args)
    elif args.model == "resnet_imagenet":
        model = get_resnet_18(args=args)
    return model

def custom_model_trainer(args, model):
    if args.dataset == "mnist_10ts":
        return MNISTTrainer(model)
    elif args.dataset == "splitCifar100":
        return CifarTrainer(model)
    elif args.dataset == "five_datasets":
        return ResnetTrainer(model)
    elif args.dataset == "imagenet":
        return ImagenetTrainer(model)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_fl_algorithm_initializer(alg_name):
    if alg_name == "FedAvgSeq":
        fl_algorithm = FedML_FedAvgSeq_distributed
        print("Fedavg ***********************************************************")
    else:
        raise Exception("please do sanity check for this algorithm.")
    return fl_algorithm


if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    args = add_args(argparse.ArgumentParser(description="CFL"))
    logger.info(args)

    #load epoch numbers for each time stamp
    epoch_number_dic = {}
    with open("epochs_" + args.dataset + ".txt") as f:
        for line in f:
            (key, val) = line.split()
            epoch_number_dic[int(key)] = int(val)
    args.comm_round = np.sum(list(epoch_number_dic.values()))+1
    task_start_indices = np.cumsum(list(epoch_number_dic.values()), dtype=int)[:-1]


    if process_id == 0:
        iid_print = "IID" if args.iid == 1 else "nonIID"
        wandb.init(
            mode="disabled",
            project="FOT",
            entity="your-entity-name",
            name=str(args.fl_algorithm)
            + "-r"
            + str(args.comm_round)
            + "-tc"
            + str(args.client_num_in_total)
            + "-c"
            + str(args.client_num_per_round)
            + "-algo_"
            + str(args.algo)
            + "-e"
            + str(args.epochs)
            + "-lr"
            + str(args.lr)
            + "-bs"
            + str(args.batch_size)
            + "-"
            + args.model
            + "-"
            + args.dataset
            + "-"
            + iid_print
            + "-seed"
            + str(args.test_seed)
            + "-het"
            + str(int(args.task_pct_per_client *100)),
            config=args,
            settings= wandb.Settings(disable_git=True,disable_code = True)
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    set_seed(args.test_seed)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    # logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # server_device_index = args.starting_gpu
    if process_id == 0:
        device = torch.device(
            "cuda:" + str(args.starting_gpu) if torch.cuda.is_available() else "cpu"
        )
    else:
        process_gpu_dict = dict()
        for client_index in range(args.gpu_worker_num):
            gpu_index = client_index % args.gpu_num_per_server + args.starting_gpu
            process_gpu_dict[client_index] = gpu_index

        logging.info(process_gpu_dict)
        device = torch.device(
            "cuda:" + str(process_gpu_dict[process_id - 1])
            if torch.cuda.is_available()
            else "cpu"
        )
    args.device = device
    logger.info(device)

    # load data and model
    dataset, available_clients_per_ts = load_data(args, args.dataset)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # load model and trainer
    model = create_model(args,output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    seen_tasks_per_ts = []
    for i in range(len(train_data_local_dict[0])):
        arr = []
        for j in range(len(train_data_local_dict)):
            if train_data_local_dict[j][i] is not None:
                arr.append(train_data_local_dict[j][i][1])
        if len(seen_tasks_per_ts) == 0:
            seen_tasks_per_ts.append(np.unique(arr))
        else:
            seen_tasks_per_ts.append(np.union1d(np.unique(arr), seen_tasks_per_ts[-1]))


    # start "federated averaging (FedAvg)"
    fl_alg = get_fl_algorithm_initializer(args.fl_algorithm)
    fl_alg(
        process_id,
        worker_number,
        device,
        comm,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        args,
        model_trainer,
        available_clients_per_ts,
        seen_tasks_per_ts,
        task_start_indices,
    )
