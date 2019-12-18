__author__ = ["Shayan Fazeli"]
__email__ = ["shayan@cs.ucla.edu"]
__credit__ = ["ER Lab - UCLA"]

import torch


def train_the_model(
    model,
    dataloaders,
    device,
    criterion,
    optimizer
):
    """
    The main trainer function that is responsible
    for training the model for 50 epochs. In this simple example
    I have not considered a large degree of freedom (for example one should normally be
    able to adjust the maximum number of epochs, optimization parameters, etc.). In larger
    examples, these must be noted as well.
    """

    for epoch in range(0, 50):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            elif phase == 'test':
                model.eval()

            running_loss = 0.0
            for batch_data in dataloaders[phase]:
                inputs = batch_data["observations"].to(device)
                labels = batch_data["labels"].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print("{}: loss={:.4f}\n".format(phase, epoch_loss))
    return model
