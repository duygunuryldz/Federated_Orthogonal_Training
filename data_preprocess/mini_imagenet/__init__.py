from data_preprocess.mini_imagenet.seq_miniimagenet import SequentialMiniImagenet
from data_preprocess.mini_imagenet.utils.continual_dataset import ContinualDataset
from argparse import Namespace


def get_dataset() -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """()
    return SequentialMiniImagenet
