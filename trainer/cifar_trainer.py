import sys
import time
import logging
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


def collect_activations(model,train_data_loader, device, orth_set):
    start = time.time()
    model.eval()
    activation = {}
    layer_names = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'fc1.weight', 'fc2.weight']
    for key in layer_names:
        activation[key] = []
    for batch_index, (x, y) in enumerate(train_data_loader):
        _ = model(x.to(device))
        for key in model.act.keys():
             activation[key].append(model.act[key].detach().cpu())
    for name in activation.keys():
        activation[name] = torch.cat(activation[name],dim=0)
    
    
    act_key = list(activation.keys())
    bsz = len(train_data_loader.dataset)
    for i in range(len(model.map)):
        k=0
        if i<3: #conv layers
            ksz= model.ksize[i]
            act = activation[act_key[i]]
            # print(act.device)
            
            unfolder = torch.nn.Unfold(ksz, dilation=1, padding=0, stride= 1)
            mat = unfolder(act.to(device))
            mat = mat.permute(0,2,1)
            mat = mat.reshape(-1, mat.shape[2])
            mat = mat.T

            mat = mat.to(device)
    
            ratio = 1
            if orth_set[act_key[i]] is not None:
                U = orth_set[act_key[i]].to(device)
                projected = U @ U.T @ mat
                remaining = mat - projected
                rem_norm = torch.norm(remaining)
                orj_norm = torch.norm(mat)
                ratio = (rem_norm / orj_norm).cpu()
                mat = remaining
            activation[act_key[i]] = [(mat @ (torch.normal(0, 1, size=(mat.shape[1], mat.shape[0]))).to(device)).cpu(), ratio, bsz]
        else:
            mat = activation[act_key[i]].T.to(device)
            ratio = 1
            if orth_set[act_key[i]] is not None:
                U = orth_set[act_key[i]].to(device)
                projected = U @ U.T @ mat
                remaining = mat - projected
                rem_norm = torch.norm(remaining)
                orj_norm = torch.norm(mat)
                ratio = (rem_norm / orj_norm).cpu()
                mat = remaining
            activation[act_key[i]] = [(mat @ (torch.normal(0, 1, size=(mat.shape[1], mat.shape[0] * 5))).to(device)).cpu(), ratio, bsz]
    end = time.time()
    print(f'Activations collection time {end-start}')
    print(f'Activations size: {sys.getsizeof(activation)}')
    
    return activation

 
class CifarTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(
            CifarTrainer,
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