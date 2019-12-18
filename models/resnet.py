__author__ = ["Shayan Fazeli"]
__email__ = ["shayan@cs.ucla.edu"]
__credit__ = ["ER Lab - UCLA"]

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models
import torch.hub


def get_state_of_the_art_ILSVRC_resnet_model(
        model_name: str
) -> Tuple[nn.Module, int]:
    """
    The :func:`get_state_of_the_art_ILSVRC_resnet_model` is designed based on PyTorch's documentation to provide us
    with the state-of-the-art computer vision models for the task of image recognition.

    Parameters
    ----------
    model_name: `str`, required
        The model name which is one of the list: `["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]`
    Returns
    -----------
    This method returns a `torch.nn.Module` containing the model and the input size (one dimension of it only) for it.
    """

    model_ft = None
    input_size = 0

    # certain sanity check
    assert model_name.startswith("resnet"), "this version only works with resnet models."

    if model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        """ Resnet Family
        """
        model_ft = torch.hub.load('pytorch/vision:v0.4.2', model_name, pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 10)

    return model_ft
