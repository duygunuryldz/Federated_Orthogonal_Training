import os
import sys
import numpy as np
from torchvision import datasets, transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from data_preprocess.mini_imagenet.seq_miniimagenet import SequentialMiniImagenet
from data_preprocess.five_datasets.data_loader import get

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

##################### MNIST #########################
if os.path.exists('../data/MNIST/mnist.npz') == False:
    save_mnist_data()

##################### Mini Imagent #########################
class Args():
    validation=False
args = Args()
Mini =  SequentialMiniImagenet(args)
_,_= Mini.get_data_loaders()

##################### CIFAR100 #########################
transform = transforms.Compose(  [transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4824, 0.4467),(0.2471, 0.24353, 0.2616))])
datasets.CIFAR100(root='../data/Cifar100', train=True, download=True, transform=transform)
datasets.CIFAR100(root='../data/Cifar100', train=False, download=True, transform=transform)

##################### 5Datasets #########################
_,_,_ = get(pc_valid=0)

