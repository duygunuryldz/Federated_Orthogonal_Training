import logging

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def distributed_cifar100(batch_size,task_num, num_client, iid, task_perc=1.0, seed=0):

    total_client_num = int(num_client/task_perc)

    np.random.seed(seed)
    transform = transforms.Compose(  [transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4824, 0.4467),(0.2471, 0.24353, 0.2616))])
    trainset = torchvision.datasets.CIFAR100(root='../data/Cifar100', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='../data/Cifar100', train=False, download=True, transform=transform)
    num_class_per_task = int(100/task_num)
    
    y = np.array(trainset.targets)
    data_loader_dict = {}
    available_clients = []
    for j in range(0, 100, num_class_per_task):
        selected_clients = np.sort(np.random.choice(total_client_num, size=num_client, replace=False))
        available_clients.append(selected_clients)
        all_x = []
        all_y = []
        for i in range(j, j+num_class_per_task):
            ind = np.where(y == i)[0]
            for k in ind:
                all_x.append(trainset[k][0].numpy())
            all_y.append([int(i%num_class_per_task)]*len(ind))
        
        X = torch.Tensor(np.array(all_x))
        Y = torch.tensor(np.array(all_y).reshape(-1),dtype=int) 

        #distribute data to the clients
        if iid:
            num_data_per_clients = int(len(X) / num_client)
            arr = np.arange(len(X))
            np.random.shuffle(arr)
            counter = 0
            for i in range(total_client_num):
                if i in selected_clients:
                    ind = arr[counter*num_data_per_clients:(counter+1)*num_data_per_clients]
                    counter += 1
                    if j == 0:
                        data_loader_dict[i] = [(DataLoader(TensorDataset(X[ind],Y[ind]), batch_size=batch_size, shuffle=True), 0)]
                    else:
                        data_loader_dict[i].append((DataLoader(TensorDataset(X[ind],Y[ind]), batch_size=batch_size, shuffle=True), int(j/num_class_per_task)))
                else:
                    if j == 0:
                        data_loader_dict[i] = [(None, 0)]
                    else:
                        data_loader_dict[i].append((None, int(j/num_class_per_task)))

        else:
            num_shards = num_client * 2
            shard_len = int(len(X) / num_shards)
            sorted_indices = torch.argsort(Y)
            X = X[sorted_indices]
            Y = Y[sorted_indices]
            arr = np.arange(num_shards)
            np.random.shuffle(arr)
            counter = 0
            for i in range(total_client_num):
                if i in selected_clients:
                    k = arr[counter*2] #first shard
                    X_local = X[k*shard_len:(k+1)*shard_len]
                    Y_local = Y[k*shard_len:(k+1)*shard_len]
                    k = arr[counter*2+1] #second shard
                    X_local = torch.cat((X_local, X[k*shard_len:(k+1)*shard_len]))
                    Y_local = torch.cat((Y_local, Y[k*shard_len:(k+1)*shard_len])) 
                    counter += 1

                    if j == 0:
                        data_loader_dict[i] = [(DataLoader(TensorDataset(X_local,Y_local), batch_size=batch_size, shuffle=True), 0)]
                    else:
                        data_loader_dict[i].append((DataLoader(TensorDataset(X_local,Y_local), batch_size=batch_size, shuffle=True), int(j/num_class_per_task)))
                else:
                    if j == 0:
                        data_loader_dict[i] = [(None, 0)]
                    else:
                        data_loader_dict[i].append((None, int(j/num_class_per_task)))
                
    #test data
    y = np.array(testset.targets)
    test_data_global = []
    for j in range(0, 100, num_class_per_task):
        all_x = []
        all_y = []
        for i in range(j, j+num_class_per_task):
            ind = np.where(y == i)[0]
            for k in ind:
                all_x.append(testset[k][0].numpy())
            all_y.append([int(i%num_class_per_task )]*len(ind))
        
        X = torch.Tensor(np.array(all_x))
        Y = torch.tensor(np.array(all_y).reshape(-1),dtype=int) 
        test_data_global.append(DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=True))

    train_data_global = None
    class_num = int(100/task_num)
    train_data_num = int(50000/task_num)
    test_data_num = int(10000/task_num)
    data_local_num_dict = dict()
    test_data_local_dict = dict()

    return (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            data_loader_dict,
            test_data_local_dict,
            class_num,
            available_clients,
        )

















    
