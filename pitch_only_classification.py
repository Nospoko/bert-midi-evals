from typing import Optional

import numpy as np
import torch.optim
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PitchSeqNN
from utils import test_model, plot_loss_curves
from data.dataset import ComposerClassificationDataset

BATCH_SIZE = 16
N_EPOCHS = 25
SEQUENCE_LENGTH = 64


def prepare_samples_pitch_only(dataset: ComposerClassificationDataset):
    """
    Prepares the samples by selecting only the pitch information from them and converting it
    into a format suitable for training.

    Args:
        dataset (ComposerClassificationDataset): An instance of YourDatasetClass containing samples with 'notes'
                                    and 'composer' information.

    Returns:
        list: list of tuples (pitches, label) for ComposerClassificationDataset.samples
    """
    df = pd.DataFrame(dataset.samples)
    df = df[["notes", "composer"]]
    # change classnames to numbers
    df["composer"] = df["composer"].apply(lambda x: 0 if x == dataset.selected_composers[0] else 1)
    # take only ['pitch'] column as an input tensor
    df["pitches"] = df["notes"].apply(lambda x: x["pitch"])
    samples = [(torch.tensor(row["pitches"], dtype=torch.float32), torch.tensor(row["composer"])) for _, row in df.iterrows()]

    # normalization into [-0.5, 0.5]
    for pitches, _ in samples:
        pitches.data = (pitches.data - 64) / 127.0

    # pop some of the samples, to fit to batch size
    while len(samples) % BATCH_SIZE != 0:
        samples.pop(-1)
    return samples


def train_model(model: nn.Module, train_data: DataLoader, lr=1e-5, val_data: Optional[DataLoader] = None, n_epochs=N_EPOCHS):
    """
    Train a classifier using the provided train_data. Evaluates on optional val_data

     Parameters:
        model (nn.Module): The neural network model to be trained.
        train_data (DataLoader): The DataLoader containing the training dataset.
        lr (float, optional): The learning rate for the Adam optimizer. Default is 1e-5.
        val_data (DataLoader, optional): The DataLoader containing the validation dataset. Default is None.
        n_epochs (int): Number of epochs.

    Returns:
        nn.Module: Trained model after completing the training loop.
        pd.DataFrame: history of training containing loss, accuracy, val_loss and val_accuracy columns
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # history container
    if val_data is not None:
        history = pd.DataFrame(columns=["loss", "accuracy", "val_loss", "val_accuracy"])
    else:
        history = pd.DataFrame(columns=["loss", "accuracy"])

    # training loop
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            total += labels.size(0)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            preds = output.argmax(1)
            for pred, label in zip(preds, labels):
                if pred == label:
                    correct += 1
            running_loss += loss.item()

        running_loss = running_loss / len(train_data)
        running_accuracy = correct / total

        if val_data is not None:
            val_loss, val_accuracy = evaluate(model, val_data)
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        data={
                            "loss": running_loss,
                            "accuracy": running_accuracy,
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                        },
                        index=[epoch],
                    ),
                ]
            )
            print(
                f"[{epoch + 1}] loss: {running_loss:.3f} accuracy: {running_accuracy:.3f}"
                + f" val_loss: {val_loss:.3f} val_accuracy: {val_accuracy:.3f}"
            )
        else:
            history = pd.concat([history, pd.DataFrame({"loss": running_loss, "accuracy": running_accuracy}, index=[epoch])])
            print(f"[{epoch + 1}] loss: {running_loss:.3f} accuracy: {running_accuracy:.3f}")

    print("Finished Training")
    return model, history


def evaluate(model, test_dataloader: DataLoader):
    """
    Evaluates the performance of a given model

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_dataloader (DataLoader): The test dataset loader containing notes and labels in batches.

    Returns:
        Returns:
        tuple: A tuple containing the evaluation results.
            - val_loss (float): The average loss of the model on the test dataset.
            - val_accuracy (float): The accuracy of the model on the test dataset, as a percentage.
    """
    # get test data
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0
    correct = 0
    total = 0
    # test
    with torch.no_grad():
        for data in test_dataloader:
            notes, labels = data
            total += labels.size(0)
            out = model(notes)
            loss = criterion(out, labels)
            loss_sum += loss.item()
            preds = out.argmax(1)
            correct += (preds == labels).sum()
    val_loss = loss_sum / len(test_dataloader)
    val_accuracy = correct / total
    return val_loss, val_accuracy


def get_dataloader(split="train", sequence_length=64, batch_size=BATCH_SIZE):
    """Create and prepare a DataLoader from ComposerClassificationDataset.

    Parameters:
        split (str): dataset split. Possible splits: "train", "test", "validation"
        sequence_length (int): Length of a note sequence to be extracted.
        batch_size (int): A batch size for DataLoader.
    """
    data = ComposerClassificationDataset(split=split, sequence_length=sequence_length)
    data.samples = prepare_samples_pitch_only(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    # Versions to check
    model_layers = [[64, 128, 64, 32, 8, 2], [64, 128, 256, 64, 2]]

    # Data to use
    train_dataloader = get_dataloader(split="train", sequence_length=64, batch_size=16)
    validation_dataloader = get_dataloader(split="validation", sequence_length=64, batch_size=16)
    test_dataloader = get_dataloader(split="test", sequence_length=64, batch_size=16)

    # Check each version
    results = []
    for layers in model_layers:
        model = PitchSeqNN(layers)
        model, history = train_model(model, train_data=train_dataloader, lr=3e-4, val_data=validation_dataloader)

        results.append(evaluate(model, test_dataloader))
        plot_loss_curves(history)
        test_model(model, test_dataloader)

    print(results)
    print(model_layers[np.argmin(results[:, 0])])


if __name__ == "__main__":
    main()
