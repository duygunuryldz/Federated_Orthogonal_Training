import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.nn.functional import relu, avg_pool2d

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, space1= [None], space2= [None]):
        if space1[0] is not None or space2[0] is not None:
            self.count = self.count % 2 
            self.act['conv_{}'.format(self.count)] = x
            self.count +=1
            out = relu(self.bn1(self.conv1(x, space1=space1[0], space2 = space2[0])))
            self.count = self.count % 2 
            self.act['conv_{}'.format(self.count)] = out
            self.count +=1
            out = self.bn2(self.conv2(out, space1=space1[1], space2 = space2[1]))
            out += self.shortcut(x)
            out = relu(out)
        else:
            self.count = self.count % 2 
            self.act['conv_{}'.format(self.count)] = x
            self.count +=1
            out = relu(self.bn1(self.conv1(x)))
            self.count = self.count % 2 
            self.act['conv_{}'.format(self.count)] = out
            self.count +=1
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = relu(out)            
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, space1 = [None], space2 = [None]):
        
        bsz = x.size(0)
        if space1[0] is not None or space2[0] is not None:
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84), space1=space1[0], space2 = space2[0]))) 
  
            out = self.layer1[0](out, space1=space1[1:3], space2 = space2[1:3])
            out = self.layer1[1](out, space1=space1[3:5], space2 = space2[3:5])
            out = self.layer2[0](out, space1=space1[5:8], space2 = space2[5:8])
            out = self.layer2[1](out, space1=space1[8:10], space2 = space2[8:10])
            out = self.layer3[0](out, space1=space1[10:13], space2 = space2[10:13])
            out = self.layer3[1](out, space1=space1[13:15], space2 = space2[13:15])
            out = self.layer4[0](out, space1=space1[15:18], space2 = space2[15:18])
            out = self.layer4[1](out, space1=space1[18:20], space2 = space2[18:20])

            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y=[]
            for t,i in self.taskcla:
                y.append(self.linear[t](out))
        else:
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84))))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y=[]
            for t,i in self.taskcla:
                y.append(self.linear[t](out))      
        return y

    def compute_conv_output_size(self, Lin,kernel_size,stride=1,padding=0,dilation=1):
        return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


def get_resnet_18(args):
    nf = 32
    taskcla = [(0, 5),
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
                (5, 5),
                (6, 5),
                (7, 5),
                (8, 5),
                (9, 5),
                (10, 5),
                (11, 5),
                (12, 5),
                (13, 5),
                (14, 5),
                (15, 5),
                (16, 5),
                (17, 5),
                (18, 5),
                (19, 5)]
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)