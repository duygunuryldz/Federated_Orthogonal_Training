from .utils import transform_tensor_to_list


class FedAVGTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer
        self.round_idx = 0
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

    def update_model(self, weights, client_index):
        self.trainer.set_model_params(weights)

    def update_round(self, round_idx, client_idx):
        self.round_idx = round_idx
        self.client_index = client_idx

    def update_dataset(self, client_index, time_stamp):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index][time_stamp][0]
        self.local_sample_number = 400
        self.test_local = None

    def train(self, round_idx=None, time_stamp = None, task_end=False, orth_set = None):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args, round_idx, int(self.client_index), time_stamp, task_end)
        activations = None
        if self.args.algo == "orth" and task_end:
            print("Collect activations!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            activations = self.trainer.get_activations(self.train_local, self.device, orth_set)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number, activations

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )