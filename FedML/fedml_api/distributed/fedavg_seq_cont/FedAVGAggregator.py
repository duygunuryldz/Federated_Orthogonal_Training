import copy
import logging
import random
import time
import sys
import pickle

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class FedAVGAggregator(object):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
        available_clients_per_ts,
        seen_tasks_per_ts,
        task_start_indices,
    ):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        
        self.aggregation_time = 0
        self.total_time = 0

        self.available_clients_per_ts = available_clients_per_ts
        self.seen_tasks_per_ts = seen_tasks_per_ts
        self.task_finish_indices=np.array(task_start_indices)-1
        self.first_acc_of_tasks = []
        self.last_acc_of_tasks = []
        self.activation_dict = dict()
        self.orth_set = {}
        if self.args.model == "mlp":
            self.orth_layer_names = ["layers.0.weight", "layers.3.weight", "layers.6.weight", "layers.9.weight"]
        elif self.args.model == "alexnet":
            self.orth_layer_names = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'fc1.weight', 'fc2.weight']
        elif self.args.model == "resnet":
            self.orth_layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', \
    'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.shortcut.0.weight', \
        'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', \
            'layer3.0.shortcut.0.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', \
                'layer4.0.conv2.weight', 'layer4.0.shortcut.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']
        elif self.args.model == 'resnet_imagenet':
            self.orth_layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight',
             'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.shortcut.0.weight', 
             'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 
             'layer3.0.shortcut.0.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 
             'layer4.0.conv2.weight', 'layer4.0.shortcut.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']
        for name in self.orth_layer_names:
            self.orth_set[name] = None
        self.epsilon = args.epsilon

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def workload_estimate(self, client_indexes, mode="simulate"):
        if mode == "simulate":
            client_samples = [
                self.train_data_local_num_dict[client_index]
                for client_index in client_indexes
            ]
            workload = client_samples
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return workload

    def memory_estimate(self, client_indexes, mode="simulate"):
        if mode == "simulate":
            memory = np.ones(self.worker_num)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return memory

    def resource_estimate(self, mode="simulate"):
        if mode == "simulate":
            resource = np.ones(self.worker_num)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return resource

    def record_client_runtime(self, worker_id, client_runtimes):
        self.activation_dict.update(client_runtimes)

    def client_schedule(self, round_idx, client_indexes, mode="simulate"):

        client_schedule = np.array_split(client_indexes, self.worker_num)
        return client_schedule

    def get_average_weight(self, client_indexes):
        average_weight_dict = {}
        for client_index in client_indexes:
            average_weight_dict[client_index] = (
                1 / len(client_indexes)
            )
        return average_weight_dict
    
    def expand_orth_set(self):
        print("Expand orth set")
        ratios = {}
        num_samples = {}
        activations = {}
        act_list = list(self.activation_dict.values())
        keys = act_list[0].keys()
        for k in keys:
            for i in range(0, len(act_list)):
                local_act = act_list[i]
                if i == 0:
                    activations[k] = local_act[k][0]
                    ratios[k] = [local_act[k][1]]
                    num_samples[k] = [local_act[k][2]]
                else:
                    activations[k] += local_act[k][0]
                    ratios[k].append(local_act[k][1])
                    num_samples[k].append(local_act[k][2])

        for key in activations.keys():
            weights = np.array(num_samples[key]) / np.sum(num_samples[key])
            weighted_avg = np.sum(weights * np.array(ratios[key]))
            org_eps = self.epsilon
            new_eps = (weighted_avg - (1 - org_eps)) / weighted_avg
            #find svds of remaining
            U, S, V = torch.svd(activations[key])
            #find how many singular vectors will be used
            total = torch.norm(activations[key])**2 
            for i in range(len(S)):
                hand = torch.norm(S[0:i+1])**2
                if hand / total > new_eps:
                    break

            if self.orth_set[key] == None:
                self.orth_set[key] = U[:,0:i+1]
            else:
                self.orth_set[key] = torch.cat((self.orth_set[key], U[:,0:i+1]),dim=1)
            
            self.orth_set[key], _ = torch.qr(self.orth_set[key])
        self.epsilon += self.args.eps_inc

    def aggregate(self, round_idx):
        start_time = time.time()
        model_list = []

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append(self.model_dict[idx])
        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        averaged_params = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_model_params = model_list[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k]
                else:
                    averaged_params[k] += local_model_params[k]

        global_params = self.get_global_model_params()
        global_gradients = model_list[0]
        for k in global_params.keys():
            global_gradients[k] = global_params[k] - averaged_params[k]

        #Apply projected gradient descent
        for key in self.orth_layer_names:
            if self.orth_set[key] == None: continue
            if "conv" in key or "shortcut" in key:
                grad = global_gradients[key]
                projected = self.orth_set[key] @ self.orth_set[key].T @ grad.view(grad.size(0), -1).T
                global_gradients[key] = grad - projected.T.view(grad.size())
            else:
                grad = global_gradients[key]
                projected = self.orth_set[key] @ self.orth_set[key].T @ grad.T
                global_gradients[key] = grad - projected.T

        for k in global_params.keys():
            averaged_params[k] = global_params[k] - global_gradients[k]

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        if list(self.activation_dict.values())[0] is not None:
            print("Expanding orth set")
            self.expand_orth_set()
            for key in self.orth_layer_names:
                if self.orth_set[key] == None: continue
                print(self.orth_set[key].shape)
                shape1, shape2 = self.orth_set[key].shape
                wandb.log({f"Space/{key}": shape2/shape1, "task": round_idx})

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        self.activation_dict = dict()

        return averaged_params

    def pos_normal(self, mean, sigma):
        x = np.random.normal(mean, sigma)
        if x < mean:
            x = mean
        elif x > mean + 2*sigma:
            x = mean + 2*sigma
        return x
    
    def get_completion_time(self, idx, local_profile, epochs, partial_ratio):
        compute_speed = local_profile['computation'] * self.pos_normal(1, 0.3)
        bandwidth = local_profile['communication']
        upload_size = sys.getsizeof(pickle.dumps(self.trainer.model))/1024.0*8.
        download_size = upload_size
        compute_time = (3.0 * self.train_data_local_num_dict[idx] * epochs * partial_ratio * float(compute_speed)/1000)
        comm_time = (upload_size+download_size) * partial_ratio/float(bandwidth)
        total_time = compute_time + comm_time
        return total_time

    def client_sampling(self, round_idx, time_stamp, client_num_per_round):
        print("Client selection...")
        print(self.available_clients_per_ts[time_stamp])

        print(f'available client list length {len(self.available_clients_per_ts[time_stamp])}')
        print(f'available client list  {(self.available_clients_per_ts[time_stamp])}')
        if len(self.available_clients_per_ts[time_stamp]) <= client_num_per_round:
            client_indexes = self.available_clients_per_ts[time_stamp]
        else:
            num_clients = min(client_num_per_round, len(self.available_clients_per_ts[time_stamp]))
            np.random.seed( round_idx )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice( self.available_clients_per_ts[time_stamp], num_clients, replace=False)

        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(
                range(test_data_num), min(num_samples, test_data_num)
            )
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(
                subset, batch_size=self.args.batch_size
            )
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx, time_stamp):
        if self.trainer.test_on_the_server(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args ):
            return

        flag = True if round_idx in self.task_finish_indices or round_idx == (self.args.comm_round-2) else False
        if round_idx % self.args.frequency_of_the_test == 0 or flag:

            logging.info( "################test_on_server_for_all_clients : {}".format(round_idx) )
            metrics = self.trainer.test(self.test_global[self.seen_tasks_per_ts[time_stamp]], self.device, self.args)
            print(metrics)

            
            self.last_acc_of_tasks = []
            sum_acc = 0
            for i, task in enumerate(self.seen_tasks_per_ts[time_stamp]):

                test_tot_correct, test_num_sample, test_loss = (
                    metrics[i]["test_correct"],
                    metrics[i]["test_total"],
                    metrics[i]["test_loss"],
                )

                # test on test dataset
                test_acc = (test_tot_correct) / (test_num_sample)
                sum_acc += test_acc
                test_loss = (test_loss) / (test_num_sample)
                wandb.log({f"Test/Acc task {task}": test_acc, "round": round_idx})
                wandb.log({f"Loss/Loss task {task}": test_loss, "round": round_idx})
                stats = {"test_acc": test_acc, 
                        "test_loss": test_loss}
                logging.info(stats)

                if task != time_stamp:
                    self.last_acc_of_tasks.append(test_acc)
                if flag and task == time_stamp:
                    print(f"end of task {task}")
                    self.first_acc_of_tasks.append(test_acc)
                    self.last_acc_of_tasks.append(test_acc)

            wandb.log({f"Test/Avg Accuracy": sum_acc/len(metrics), "round": round_idx})
            print(self.first_acc_of_tasks)
            print(self.last_acc_of_tasks)

            forgetting = np.array(self.first_acc_of_tasks) - np.array(self.last_acc_of_tasks)
            print(forgetting)
            print(np.mean(forgetting))
            if flag or round_idx == (self.args.comm_round-2):
                bwt = np.mean(forgetting[:-1])
            else:
                bwt = np.mean(forgetting)
            print(bwt)
            wandb.log({f"Test/BWT": bwt, "round": round_idx})

