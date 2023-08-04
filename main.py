from typing import Optional

import torch.optim
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PitchSeqNNv2
from data.dataset import ComposerClassificationDataset

BATCH_SIZE = 16
N_EPOCHS = 50
SEQUENCE_LENGTH = 64


def prepare_dataset_pitch_only(dataset: ComposerClassificationDataset):
    """
    Prepares the dataset by selecting only the pitch information from the samples and converting it
    into a format suitable for training.

    Args:
        dataset (ComposerClassificationDataset): An instance of YourDatasetClass containing samples with 'notes'
                                    and 'composer' information.

    Returns:
        None: Modifies the 'samples' attribute of the dataset in-place.
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
    dataset.samples = samples
    # print(dataset.samples[0])


def train():
    """
    Train the ComposerClassifier model using the ComposerClassificationDataset.

    Returns:
        PitchSeqNN: Trained model after completing the training loop.
    """
    # get train data
    train_data = ComposerClassificationDataset(split="train", sequence_length=SEQUENCE_LENGTH)
    prepare_dataset_pitch_only(train_data)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = ComposerClassificationDataset(split="validation", sequence_length=SEQUENCE_LENGTH)
    prepare_dataset_pitch_only(test_data)
    # initialize model
    model = PitchSeqNNv2()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # training loop
    for epoch in range(N_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            # comment for accumulating gradients
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Periodically print the running loss for monitoring the training progress.
        running_loss = running_loss / len(train_dataloader)
        val_loss = evaluate(model, test_data)
        print(f"[{epoch + 1}] loss: {running_loss:.3f} val loss: {val_loss:.3f}")

    print("Finished Training")
    return model


def evaluate(model, test_data: ComposerClassificationDataset):
    """
        Evaluates the performance of a given model

        Args:
            model (torch.nn.Module): The trained model to be evaluated.
            test_data (ComposerClassificationDataset): The test dataset containing notes and labels.

        Returns:
            val_loss: The average loss value calculated over the test dataset.
        """
    # get test data
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            notes, labels = data
            out = model(notes)
            loss += nn.CrossEntropyLoss()(out, labels)
            _, preds = torch.max(out, 1)
    return loss / len(test_dataloader)


def test_model(model: Optional[nn.Module] = None, path: Optional[str] = None):
    """
    Evaluate the performance of the trained ComposerClassifier model on the test dataset.

    Args:
        model (PitchSeqNN): The trained ComposerClassifier model to be evaluated.
        path (str, optional): If specified, the function will load the model's state from the
                              provided path.
    """
    # get test data
    test_data = ComposerClassificationDataset(split="test", sequence_length=SEQUENCE_LENGTH)
    prepare_dataset_pitch_only(test_data)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # if path is specified load state from file
    if model is None:
        model = PitchSeqNNv2()
    if path is not None:

        model.load_state_dict(torch.load(path))
    # containers for counting prediction
    correct_pred = {classname: 0 for classname in test_data.selected_composers}
    total_pred = {classname: 0 for classname in test_data.selected_composers}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            notes, labels = data
            out = model(notes)
            loss = nn.CrossEntropyLoss()(out, labels)
            print(loss)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[test_data.selected_composers[label]] += 1
                total_pred[test_data.selected_composers[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
    print(f"Accuracy of the network on the test data: {100 * correct / total} %")
    print("correctly predicted :" + str(correct_pred))
    return correct / total


def main():
    torch.manual_seed(78)
    model = train()
    path = "model1.pth"
    # torch.save(model.state_dict(), path)
    test_model(model)


if __name__ == "__main__":
    main()
