import logging
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor


class FedAVGClientManager(ClientManager):
    def __init__(
        self,
        args,
        trainer,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        time_stamp_num=0,
        task_start_indices = None
    ):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.worker_id = self.rank - 1
        self.time_stamp = 0
        self.task_start_indices=task_start_indices
        self.task_finish_indices=np.array(task_start_indices)-1

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)

        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)
        self.round_idx = 0
        self.__train(global_model_params, client_indexes, average_weight_dict, orth_set={})

    def start_training(self):
        self.round_idx = 0

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)

        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        orth_set = msg_params.get(MyMessage.MSG_ARG_KEY_ORTH_SPACE)
        client_indexes = client_schedule[self.worker_id]

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(global_model_params)

        self.round_idx += 1
        if self.round_idx in self.task_start_indices:
            self.time_stamp += 1
        self.__train(global_model_params, client_indexes, average_weight_dict, orth_set)
        if self.round_idx == self.num_rounds - 1:
            self.finish()


    def send_result_to_server(self, receive_id, weights, client_runtime_info, ):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info)
        self.send_message(message)


    def add_client_model(self, local_agg_model_params, model_params, weight=1.0):
        # Add params that needed to be reduces from clients
        for name, param in model_params.items():
            if torch.sum(torch.isnan(param)) > 0:
                logging.info('LOCAL NaN Error')
                sys.exit()

            if name not in local_agg_model_params:
                local_agg_model_params[name] = param * weight
            else:
                local_agg_model_params[name] += param * weight

            if torch.sum(torch.isnan(local_agg_model_params[name])) > 0:
                logging.info('Aggregated NaN Error')
                sys.exit()


    def __train(self, global_model_params, client_indexes, average_weight_dict, orth_set):
        logging.info("#######training########### round_id = %d  timestamp = %d" % (self.round_idx, self.time_stamp))
        local_agg_model_params = {}
        client_runtime_info = {}
        for client_index in client_indexes:
            logging.info("#######training########### Simulating client_index = %d, average weight: %f " % \
                (client_index, average_weight_dict[client_index]))
            start_time = time.time()
            self.trainer.update_model(global_model_params, client_index)
            self.trainer.update_dataset(int(client_index), self.time_stamp)
            weights, local_sample_num, activations = \
                self.trainer.train(self.round_idx, self.time_stamp, self.round_idx in self.task_finish_indices, orth_set)
            
            self.add_client_model(local_agg_model_params, weights, weight=average_weight_dict[client_index])

            end_time = time.time()
            client_runtime = end_time - start_time
            client_runtime_info[client_index] = activations
            logging.info("#######training########### End Simulating client_index = %d, consuming time: %f" % \
                (client_index, client_runtime))

        self.send_result_to_server(0, local_agg_model_params, client_runtime_info)