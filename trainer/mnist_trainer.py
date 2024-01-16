import logging
import torch
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import time

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
            # data = torch.squeeze(data, 1)
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            data = data.view(len(data), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

    logging.info(
            "Client Index = {}\tEpoch: {}\tBatch Loss: {:.6f}\tBatch Number: {}".format(
                    client_idx, epoch, loss, batch_idx
                )
            )
    epoch_loss.append(sum(batch_loss) / len(batch_loss))
    end = time.time()
    print(f'1 local epoch time: {end-start}')
    return epoch_loss


def collect_activations(model,train_data_loader, device, orth_set):
    start = time.time()
    activation = {}
    layer_names = ["layers.0.weight", "layers.3.weight", "layers.6.weight", "layers.9.weight"]
    activation_indices = [-1, 1, 4, 7]
    def getActivation(id):
        # the hook signature
        def hook(model, input, output):
            activation[id].append(output.detach())
        return hook

    hooks = []
    for i, name in enumerate(layer_names):
        activation[name] = []
        if i > 0:
            hooks.append(model.layers[activation_indices[i]].register_forward_hook(getActivation(name)))
    
    num_samples = 0.0
    for batch_index, (x, y) in enumerate(train_data_loader):
        data_size = len(x)
        x = x.view(data_size, -1)
        activation[layer_names[0]].append(x.to(device))
        x = Variable(x).to(device)
        _ = model(x)
        num_samples += data_size

    for hook in hooks:
        hook.remove()
    for i, name in enumerate(layer_names):
        activation[name] = torch.cat(activation[name],dim=0).T
        ratio = 1
        if orth_set[name] is not None:
            U = orth_set[name].to(device)
            projected = U @ U.T @ activation[name]
            remaining = activation[name] - projected
            rem_norm = torch.norm(remaining)
            orj_norm = torch.norm(activation[name])
            ratio = (rem_norm / orj_norm).cpu()
            activation[name] = remaining
        activation[name] = [(activation[name] @ (torch.rand((activation[name].shape[1], activation[name].shape[0])) * 2 - 1).to(device)).cpu(), ratio, num_samples]
    end = time.time()
    print(f'Activations collection time {end-start}')
    return activation     
 
class MNISTTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(
            MNISTTrainer,
            self,
        ).__init__(*args, **kwargs)
        self.param_size = sum(p.numel() for p in self.model.parameters())
        

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    def set_info(self, info):
        print("buraya girmemeli")
        # self.model.fisher_dict, self.model.memory_x, self.model.memory_y = info


    def get_activations(self, train_data, device, orth_set):
        return collect_activations(self.model,train_data,device, orth_set)

    def train(self, train_data, device, args, round_idx, client_idx, time_stamp, task_end=False, consolidate=False):
        print(f'task end : {task_end}')
        normal_train(self.model,train_data,device,args,round_idx,client_idx,time_stamp, task_end)

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
        all_metrics = []
        
        with torch.no_grad():
            for task_loader in test_data:
                metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
                for _, (data, labels) in enumerate(tqdm(task_loader)):
                    # for data, labels, lens in test_data:
                    labels = labels.type(torch.LongTensor)
                    data, labels = data.to(device), labels.to(device)
                    data = data.view(len(data), -1)
                    output = model(data)
                    loss = criterion(output, labels).data.item()
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct = pred.eq(labels.data.view_as(pred)).sum()

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss * labels.size(0)
                    metrics["test_total"] += labels.size(0)

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