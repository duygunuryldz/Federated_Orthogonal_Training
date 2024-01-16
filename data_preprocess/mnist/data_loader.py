import logging

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import os

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_mnist_data():
    transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_X = []
    train_y = []
    for i in range(len(dataset1)):
        x, y = dataset1[i]
        train_X.append(x.numpy())
        train_y.append(y)
    train_X = np.array(train_X).reshape(-1, 28, 28)
    train_y = np.array(train_y)

    test_X = []
    test_y = []
    for i in range(len(dataset2)):
        x, y = dataset2[i]
        test_X.append(x.numpy())
        test_y.append(y)
    test_X = np.array(test_X).reshape(-1, 28, 28)
    test_y = np.array(test_y)

    np.savez('../data/MNIST/mnist.npz', X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y)


def distributed_pmnist(task_num = 10, num_clients = 128, batch_size=32, iid=True, task_perc=1.0, seed=0):

    total_client_num = int(num_clients/task_perc)

    data = np.load('../data/MNIST/mnist.npz', allow_pickle=True)
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    train_data_global = None
    test_data_global = []
    train_data_local_dict = {}

    available_clients = []
    #add permuted task
    for t in range(task_num):
        np.random.seed(t+seed)
        selected_clients = np.sort(np.random.choice(total_client_num, size=num_clients, replace=False))
        available_clients.append(selected_clients)

        perm = np.random.permutation(28*28) if t != 0 else np.arange(28*28)
        X_perm = X_train.reshape(-1, 28*28)[:, perm].reshape(-1, 28, 28)
        X_perm_test = X_test.reshape(-1, 28*28)[:, perm].reshape(-1, 28, 28)
        test_data_global.append(DataLoader(TensorDataset(torch.Tensor(X_perm_test), torch.tensor(y_test,dtype=int)), batch_size=batch_size, shuffle=True))

        #distribute data to the clients
        if iid:
            num_data_per_clients = int(len(X_perm) / num_clients)
            arr = np.arange(len(X_perm))
            np.random.shuffle(arr)
            counter = 0
            for i in range(total_client_num):
                if i in selected_clients:
                    ind = arr[counter*num_data_per_clients:(counter+1)*num_data_per_clients]
                    counter += 1
                    if t == 0:
                        train_data_local_dict[i] = [(DataLoader(TensorDataset(torch.Tensor(X_perm[ind]),torch.tensor(y_train[ind],dtype=int)), batch_size=batch_size, shuffle=True), 0)]
                    else:
                        train_data_local_dict[i].append((DataLoader(TensorDataset(torch.Tensor(X_perm[ind]),torch.tensor(y_train[ind],dtype=int)), batch_size=batch_size, shuffle=True), t))
                else:
                    if t == 0:
                        train_data_local_dict[i] = [(None, 0)]
                    else:
                        train_data_local_dict[i].append((None, t))
        else:
            num_shards = num_clients * 2
            shard_len = int(len(X_perm) / num_shards)
            sorted_indices = np.argsort(y_train)
            X = X_perm[sorted_indices]
            Y = y_train[sorted_indices]
            arr = np.arange(num_shards)
            np.random.shuffle(arr)
            counter = 0
            for i in range(total_client_num):
                if i in selected_clients:
                    k = arr[counter*2] #first shard
                    X_local = X[k*shard_len:(k+1)*shard_len]
                    Y_local = Y[k*shard_len:(k+1)*shard_len]
                    k = arr[counter*2+1] #second shard
                    X_local = np.vstack((X_local, X[k*shard_len:(k+1)*shard_len]))
                    Y_local = np.hstack((Y_local, Y[k*shard_len:(k+1)*shard_len])) 
                    counter += 1
                    if t == 0:
                        train_data_local_dict[i] = [(DataLoader(TensorDataset(torch.Tensor(X_local),torch.tensor(Y_local,dtype=int)), batch_size=batch_size, shuffle=True), 0)]
                    else:
                        train_data_local_dict[i].append((DataLoader(TensorDataset(torch.Tensor(X_local),torch.tensor(Y_local,dtype=int)), batch_size=batch_size, shuffle=True), t))
                else:
                    if t == 0:
                        train_data_local_dict[i] = [(None, 0)]
                    else:
                        train_data_local_dict[i].append((None, t))

    class_num = len(np.unique(y_train))
    train_data_num = task_num * 60000
    test_data_num = task_num * 10000
    # get local dataset
    data_local_num_dict = dict()
    test_data_local_dict = dict()
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
        available_clients,
    )


















    
