import torch
from torch import nn
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from torch import optim


class AlexNet(nn.Module):
    def __init__(self,num_class, task_num,args=None):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.fisher_dict = {}
        self.lamda = 0.1
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s=self.compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        s=self.compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        s=self.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.map.extend([2048])
        self.last=torch.nn.ModuleList()
        for t in range(task_num):
            self.last.append(torch.nn.Linear(2048,num_class,bias=False))

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = num_class

        self.opt = optim.SGD(self.parameters(), args.lr, momentum=0.9)


        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = x.view(bsz,3,32,32)
        self.act['conv1.weight']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv2.weight']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv3.weight']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(x)))

        x=x.view(bsz,-1)
        self.act['fc1.weight']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2.weight']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(x))
        y=[]
        for t in range(len(self.last)):
            y.append(self.last[t](x))
        return y

    def compute_conv_output_size(self, Lin,kernel_size,stride=1,padding=0,dilation=1):
        return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
    

def get_alexnet(num_task, num_class,args=None):
    return AlexNet(num_class=num_class, task_num=num_task, args=args)

