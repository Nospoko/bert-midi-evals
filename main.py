from typing import Optional

import numpy as np
import torch.optim
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ComposerClassificationDataset
from model import PitchSeqNN, PitchSeqNNv2_64, PitchSeqNNv3_64, PitchSeqNNv4_64

BATCH_SIZE = 8
N_EPOCHS = 50
SEQUENCE_LENGTH = 64


def prepare_samples_pitch_only(dataset: ComposerClassificationDataset):
    """
    Prepares the samples by selecting only the pitch information from them and converting it
    into a format suitable for training.

    Args:
        dataset (ComposerClassificationDataset): An instance of YourDatasetClass containing samples with 'notes'
                                    and 'composer' information.

    Returns:
        samples (list): list of tuples (pitches, label) for ComposerClassificationDataset.samples
    """
    df = pd.DataFrame(dataset.samples)
    df = df[["notes", "composer"]]
    # change classnames to numbers
    df["composer"] = df["composer"].apply(lambda x: 0 if x == dataset.selected_composers[0] else 1)
    # take only ['pitch'] column as an input tensor
    df["pitches"] = df["notes"].apply(lambda x: x["pitch"])
    # print(df.groupby("composer").size())
    samples = [(torch.tensor(row["pitches"], dtype=torch.float32), torch.tensor(row["composer"])) for _, row in df.iterrows()]
    # normalization into [-0.5, 0.5)
    for pitches, _ in samples:
        pitches.data = (pitches.data - 64) / 127.0
    # pop some of the samples, to fit to batch size
    while len(samples) % BATCH_SIZE != 0:
        samples.pop(-1)
    return samples


def train_model(model: nn.Module, train_data: DataLoader, lr=1e-5, val_data: Optional[DataLoader] = None):
    """
    Train a classifier using the provided train_data. Evaluates on optional val_data

     Parameters:
        model (nn.Module): The neural network model to be trained.
        train_data (DataLoader): The DataLoader containing the training dataset.
        lr (float, optional): The learning rate for the Adam optimizer. Default is 1e-5.
        val_data (DataLoader, optional): The DataLoader containing the validation dataset. Default is None.

    Returns:
        nn.Module: Trained model after completing the training loop.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # training loop
    for epoch in range(N_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            # comment for accumulating gradients
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Periodically print the running loss for monitoring the training progress.
        running_loss = running_loss / len(train_data)
        if val_data is not None:
            val_loss, val_accuracy = evaluate(model, val_data)
            print(f"[{epoch + 1}] loss: {running_loss:.3f} val loss: {val_loss:.3f} val accuracy: {val_accuracy:.3f}")
        else:
            print(f"[{epoch + 1}] loss: {running_loss:.3f}")

    print("Finished Training")
    return model


def evaluate(model, test_dataloader: DataLoader):
    """
    Evaluates the performance of a given model

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_dataloader (DataLoader): The test dataset loader containing notes and labels in batches.

    Returns:
        val_loss, val_accuracy (Tuple([float, float])): The average loss value calculated over the test dataset.
    """
    # get test data
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            notes, labels = data
            total += labels.size(0)
            out = model(notes)
            loss += nn.CrossEntropyLoss()(out, labels)
            _, preds = torch.max(out, 1)
            for pred, label in zip(preds, labels):
                if pred == label:
                    correct += 1
    val_loss = loss/len(test_dataloader)
    val_accuracy = correct / total
    return val_loss, val_accuracy


def test_model(model: nn.Module, test_data: DataLoader, path: Optional[str] = None, classnames=None):
    """
    Evaluate the performance of the trained ComposerClassifier model with the provided test data loader.

    Args:
        model (PitchSeqNN): The trained ComposerClassifier model to be evaluated.
        path (str, optional): If specified, the function will load the model's state from the
                              provided path.
    """
    if classnames is None:
        classnames = [0, 1]
    # if path is specified load state from file
    if path is not None:
        model.load_state_dict(torch.load(path))
    # containers for counting prediction
    correct_pred = {classname: 0 for classname in classnames}
    total_pred = {classname: 0 for classname in classnames}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            notes, labels = data
            out = model(notes)
            loss = nn.CrossEntropyLoss()(out, labels)
            print(loss)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[classnames[label]] += 1
                total_pred[classnames[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
    print(f"Accuracy of the network on the test data: {(100 * correct / total):.3f} %")
    print(f"correctly predicted : {correct_pred}")
    return correct / total


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
    models = [PitchSeqNN(), PitchSeqNNv2_64(), PitchSeqNNv3_64(), PitchSeqNNv4_64()]
    losses = []
    train_dataloader = get_dataloader(split="train", sequence_length=64, batch_size=16)
    validation_dataloader = get_dataloader(split="validation", sequence_length=64, batch_size=16)
    test_dataloader = get_dataloader(split="test", sequence_length=64, batch_size=16)
    for model in models:
        model = train_model(model, train_data=train_dataloader, lr=3e-4, val_data=validation_dataloader)
        losses.append(evaluate(model, test_dataloader))
    print(losses)
    torch.save(models[np.argmax(losses)].state_dict(), "best.pth")
    # path = "model1.pth"
    # torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
