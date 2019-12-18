__author__ = ["Shayan Fazeli"]
__email__ = ["shayan@cs.ucla.edu"]
__credit__ = ["ER Lab - UCLA"]

import unittest
import pytest
from models.resnet import get_state_of_the_art_ILSVRC_resnet_model
from datasets.sample_task import SampleTaskDataset
from trainers.basic import train_the_model
import torch
import torch.utils.data


class GeneralTests(unittest.TestCase):
    """
    The :class:`GeneralTests` which inherits from :class:`unittest.TestCase` includes
    the test cases for this mock sample. These test cases can be executed using
    `pytest` and target different aspects of this simple example. In general,
    it might not make sense to write test cases on applications of this scale, but as
    the code becomes larger, test cases play vital roles in determining the robustness
    of our frameworks.

    One thing to note is to make sure torch.hub includes the cache files before proceeding.
    """
    def setUp(self):
        """
        The main initialization function that prepares the built-in variables
        and models for the tests to begin.
        """
        super().setUp()
        datasets = {x: SampleTaskDataset(group=x) for x in ['train', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(
            datasets[x],
            batch_size=16,
            shuffle=(x == 'train'),
            num_workers=4,
            pin_memory=True
        ) for x in ['train', 'test']}

        self.model = get_state_of_the_art_ILSVRC_resnet_model(model_name='resnet18')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.device = torch.device('cpu')

    def test_model_building(self):
        """
        The :func:`test_model_building` tests the ability of the system to return
        an instance of resnet model in model generation scheme.
        """
        model = get_state_of_the_art_ILSVRC_resnet_model('resnet18')

    def test_dataset_building(self):
        """
        The :func:`test_dataset_building` tests building train and test datasets.
        """
        dataset = SampleTaskDataset(group='test')
        dataset = SampleTaskDataset(group='train')

    def trainer_can_run(self):
        """
        The :func:`trainer_can_run` tests the ability of the system in running the training scheme.
        """
        _ = train_the_model(model=self.model, dataloaders=self.dataloaders, device=self.device,
                            criterion=self.criterion, optimizer=self.optimizer)

    @pytest.mark.skipif(not torch.cuda.is_available())
    def trainer_can_run_on_gpu(self):
        """
        If there is GPU availability and the first gpu is free, this scheme will be used in :meth:`trainer_can_run_on_gpu`
        to test the execution of training scheme on GPU.
        """
        _ = train_the_model(model=self.model, dataloaders=self.dataloaders, device=torch.device('cuda:0'), criterion=self.criterion, optimizer=self.optimizer)


if __name__ == '__main__':
    unittest.main()
