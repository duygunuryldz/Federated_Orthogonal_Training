import logging
import time
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


def normal_train(model,train_data,device,args,round_idx,client_idx,time_stamp, task_end):
    start = time.time()
    model.to(device)
    model.train()
    logging.info(" Client ID " + str(client_idx) + " round Idx " + str(round_idx) + ' Time Stamp ' + str(time_stamp))
    criterion = nn.CrossEntropyLoss().to(device) 
    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_loss = []
    for epoch in range(args.epochs):
        batch_loss = []
        #  data, labels, lens
        for batch_idx, (data, labels) in enumerate(train_data):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)[time_stamp]
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

    logging.info(
            "Client Index = {}\tEpoch: {}\tBatch Loss: {:.6f}\tBatch Number: {}".format(client_idx, epoch, loss, batch_idx)
            )
    epoch_loss.append(sum(batch_loss) / len(batch_loss))
    end = time.time()
    print(f'1 local epoch time: {end-start}')
    return epoch_loss


def collect_activations(net,train_data_loader, device, orth_set): #distributed
    start = time.time()

    layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 
    'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 
    'layer2.0.shortcut.0.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 
    'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.0.shortcut.0.weight', 
    'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 
    'layer4.0.shortcut.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']

    stride_list = [1, 1,1,1,1, 2,1,2,1,1, 2,1,2,1,1, 2,1,2,1,1]   
    map_list    = [84,42,42,42,42,42,21,42,21,21,21,11,21,11,11,11,6,11,6,6]
    in_channel  = [3, 32, 32, 32, 32, 32, 64, 32, 64, 64, 64, 128, 64, 128, 128, 128, 256, 128, 256, 256]

    net.eval()
    activation = {}
    for key in layer_names:
        activation[key] = []
        
    for batch_index, (x, _) in enumerate(train_data_loader):
        # if batch_index > 2: break
        _ = net(x.to(device))
        act_list = [net.act['conv_in'], 
            net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
            net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[0].act['conv_0'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
            net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[0].act['conv_0'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
            net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[0].act['conv_0'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']]
        for j, key in enumerate(layer_names):
            activation[key].append(act_list[j].detach().cpu())
    for name in activation.keys():
        activation[name] = torch.cat(activation[name],dim=0)
        if "shortcut" not in name:
            activation[name] = F.pad(activation[name], (1, 1, 1, 1), "constant", 0)
    
    # bsz = 2*64
    bsz = len(train_data_loader.dataset)
    for i in range(len(stride_list)):
        layer_name = layer_names[i]
        k=0
        ksz= 3
        pad=1
        st = stride_list[i]
        act = activation[layer_name]
        if "shortcut" in layer_name:
            ksz=1
            pad=0
        
        unfolder = torch.nn.Unfold(ksz, dilation=1, padding=0, stride= st)
        mat = unfolder(act.to(device))
        mat = mat.permute(0,2,1)
        mat = mat.reshape(-1, mat.shape[2])
        mat = mat.T

        
        mat = mat.to(device)
        ratio = 1
        if orth_set[layer_name] is not None:
            U = orth_set[layer_name].to(device)
            projected = U @ U.T @ mat
            remaining = mat - projected
            rem_norm = torch.norm(remaining)
            orj_norm = torch.norm(mat)
            ratio = (rem_norm / orj_norm).cpu()
            mat = remaining
        activation[layer_name] = [(mat @ (torch.normal(0, 1, size=(mat.shape[1], mat.shape[0] * 5))).to(device)).cpu(), ratio, bsz]
    
    end = time.time()
    print(f'Activations collection time {end-start}')

    return activation

 
class ImagenetTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(
            ImagenetTrainer,
            self,
        ).__init__(*args, **kwargs)
        self.param_size = sum(p.numel() for p in self.model.parameters())
        

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_activations(self, train_data, device, orth_set):
        return collect_activations(self.model,train_data,device, orth_set)

    def train(self, train_data, device, args, round_idx, client_idx, time_stamp, task_end=False, consolidate=False):
        print(f'task end : {task_end}')
        normal_train(self.model,train_data,device,args,round_idx,client_idx,time_stamp, task_end)



    def test(self, test_data, device, args):
        print('inside test')
        model = self.model

        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
        all_metrics = []
        
        with torch.no_grad():
            for i, task_loader in enumerate(test_data):
                metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
                for _, (data, labels) in enumerate(tqdm(task_loader)):
                    data, labels = data.to(device), labels.to(device)
                    output = model(data)[i]
                    loss = criterion(output, labels).data.item()
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct = pred.eq(labels.data.view_as(pred)).sum()

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss * labels.size(0)
                    metrics["test_total"] += labels.size(0)

                print('test finito')

                all_metrics.append(metrics)
        return all_metrics

    def test_on_the_server(
            self,
            train_data_local_dict,
            test_data_local_dict,
            device,
            args=None,
            round_idx=None,
    ) -> bool:
        return False