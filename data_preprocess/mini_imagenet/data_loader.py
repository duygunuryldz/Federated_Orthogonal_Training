import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from data_preprocess.mini_imagenet.seq_miniimagenet import SequentialMiniImagenet


class ExtendedTensorDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors,transforms=transforms.Compose([])) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms(self.tensors[0][index]), self.tensors[1][index]

    def __len__(self):
        return self.tensors[0].size(0)



def distributed_imagenet(batch_size,task_num, num_client, iid, task_perc=1.0, seed=0):
    total_client_num = int(num_client/task_perc)
    np.random.seed(seed)
    MEAN = (0.47313006, 0.44905752, 0.40378186)
    STD = (0.27292014, 0.26559181, 0.27953038)
    train_transform = transforms.Compose(
                [transforms.RandomCrop(84, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(MEAN,
                                    STD)])
    test_transform = transforms.Compose(
                [transforms.Normalize(MEAN,
                                    STD)])
    class Args():
        validation=False
    args = Args()
    Mini =  SequentialMiniImagenet(args)
    train,test= Mini.get_data_loaders()

    #they are scaled but not normalized
    trainset = train.dataset
    testset = test.dataset
    
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
                        data_loader_dict[i] = [(DataLoader(ExtendedTensorDataset(X[ind],Y[ind],transforms=train_transform), batch_size=batch_size, shuffle=True), 0)]
                    else:
                        data_loader_dict[i].append((DataLoader(ExtendedTensorDataset(X[ind],Y[ind],transforms=train_transform), batch_size=batch_size, shuffle=True), int(j/num_class_per_task)))
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
                        data_loader_dict[i] = [(DataLoader(ExtendedTensorDataset(X_local,Y_local,transforms=train_transform), batch_size=batch_size, shuffle=True), 0)]
                    else:
                        data_loader_dict[i].append((DataLoader(ExtendedTensorDataset(X_local,Y_local,transforms=train_transform), batch_size=batch_size, shuffle=True), int(j/num_class_per_task)))
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
        test_data_global.append(DataLoader(ExtendedTensorDataset(X,Y,transforms=test_transform), batch_size=batch_size, shuffle=True))

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
