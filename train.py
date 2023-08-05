from typing import Callable

import torch.optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import PitchSeqNNv2
from data.dataset import BagOfPitches

BATCH_SIZE = 32
N_EPOCHS = 50
LR = 1e-4


def main():
    dataset = BagOfPitches()
    v_dataset = BagOfPitches(split="validation")
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    v_dataloader = DataLoader(v_dataset, batch_size=BATCH_SIZE)

    model = PitchSeqNNv2(input_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(range(N_EPOCHS), desc="Training started!")
    for epoch in pbar:
        train_stats = training_epoch(
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
        )

        v_stats = validation_epoch(
            loader=v_dataloader,
            model=model,
            criterion=criterion,
        )
        bar = "Train: loss={t_loss:.3f}, acc={t_acc:.2f}, Val: loss={v_loss:.3f}, acc={v_acc:.2f}".format(
            t_loss=train_stats["loss"],
            t_acc=train_stats["accuracy"],
            v_loss=v_stats["loss"],
            v_acc=v_stats["accuracy"],
        )
        pbar.set_description(bar)


def training_epoch(
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    criterion: Callable,
):
    correct = 0
    running_loss = 0
    for batch in train_dataloader:
        inputs = batch["data"]
        labels = batch["label"]

        optimizer.zero_grad()

        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()

        optimizer.step()

        correct += (pred.argmax(1) == labels).sum().item()
        running_loss += loss.item()

    running_loss = running_loss / len(train_dataloader)
    accuracy = correct / len(train_dataloader.dataset)
    stats = {
        "loss": running_loss,
        "accuracy": accuracy,
    }
    return stats


def validation_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: Callable,
):
    v_loss = 0
    correct = 0
    for batch in loader:
        inputs = batch["data"]
        labels = batch["label"]

        pred = model(inputs)
        loss = criterion(pred, labels)

        v_loss += loss.item()
        correct += (pred.argmax(1) == labels).sum().item()
    v_loss = v_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    stats = {
        "loss": v_loss,
        "accuracy": accuracy,
    }
    return stats


if __name__ == "__main__":
    main()
