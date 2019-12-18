__author__ = ["Shayan Fazeli"]
__email__ = ["shayan@cs.ucla.edu"]
__credit__ = ["ER Lab - UCLA"]

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pickle
import warnings
import os

warnings.filterwarnings("ignore")
plt.ioff()  # turning off the interactive mode


class SampleTaskDataset(Dataset):
    """
    The :class:`SampleTaskDataset` is the sample PyTorch dataset (not Iterable Dataset) that I
    have created mainly for this mock project. Note that in complicated cases, batch sampling or
    even using the new :class:`IterableDataset` class in PyTorch can make things a lot easier for us.
    """
    def __init__(self, group):
        path_to_dataset = os.path.join(os.path.dirname(__file__), '../data/')
        with open(os.path.join(path_to_dataset, "sample_task_observations_{}.pkl".format(group)), 'rb') as handle:
            self.task_observations = pickle.load(handle)
        with open(os.path.join(path_to_dataset, "sample_task_labels_{}.pkl".format(group)), 'rb') as handle:
            self.task_labels = pickle.load(handle)

    def __len__(self):
        return len(self.task_labels)

    def __getitem__(self, index):
        return {
            'observation': self.task_observations[index],
            'label': self.task_labels[index]
        }
