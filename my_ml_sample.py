__author__ = ["Shayan Fazeli"]
__email__ = ["shayan@cs.ucla.edu"]
__credit__ = ["ER Lab - UCLA"]


import argparse
import torch
import torch.nn
import torch.optim
import torch.nn.functional
from models.resnet import get_state_of_the_art_ILSVRC_resnet_model
from datasets.sample_task import SampleTaskDataset
from trainers.basic import train_the_model


def main(args):
    """
    args: `argparse.Namespace`, required
        The argument namespace that is passed when the application is called.
    """
    datasets = {x: SampleTaskDataset(group=x) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x],
        batch_size=16,
        shuffle=(x=='train'),
        num_workers=4,
        pin_memory=True
    ) for x in ['train', 'test']}

    model = get_state_of_the_art_ILSVRC_resnet_model(model_name=args.model)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    device = torch.device('cuda:0')
    _ = train_the_model(model=model, dataloaders=dataloaders, device=device, criterion=loss, optimizer=optimizer)
    print("end of training.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("my simple ML approach")
    parser.add_argument(
        "--model",
        type=str,
        default='resnet18'
    )

    args = parser.parse_args()
    main(args=args)
