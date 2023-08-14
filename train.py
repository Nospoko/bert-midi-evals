import uuid
import os.path
import itertools
from typing import Callable

import hydra
import numpy as np
import torch.optim
import pandas as pd
import torch.nn as nn
import streamlit as st
from tqdm import tqdm
from fortepyan import MidiPiece
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from model import PitchSeqNN
from data.dataset import BagOfPitches
from utils import test_model, piece_av_files


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model, run_id = run_experiment(cfg, classnames=cfg.model.composers)
    print(run_id)


def wrong_preds(model: nn.Module, test_data: DataLoader) -> dict:
    # get data from testing
    data = test_model(model, test_data=test_data)

    # select samples with false predictions
    filtered_indices = [index for index, value in enumerate(data["pred"]) if value != data["label"][index]]
    wrong_predictions = {key: [values[index] for index in filtered_indices] for key, values in data.items()}
    return wrong_predictions


def pair_comparison_main(cfg: DictConfig):
    """
    Finds composers with at least 5 pieces in the validation split,
    then trains model for each pair by calling run_experiment

    Args:
        cfg (DictConfig): DictConfig passed from hydra.main function containing hyperparameters and model specification.
    """

    composers_to_check = find_composers_to_check()
    composers = [pair for pair in itertools.combinations(composers_to_check, r=2)]
    print(f"pairs to check: {len(composers)}")
    # container for trained models
    models = []

    # train model for each pair
    for pair in composers:
        print(f"{pair[0]} vs {pair[1]}")
        model = run_experiment(cfg=cfg, classnames=pair)
        models.append((model, pair))


def find_composers_to_check() -> list[str]:
    """
    Finds well-represented composers in the dataset.

    Returns:
        list [str]: list of composers with at least 5 pieces in the validation split of the dataset.
    """
    # load data
    dataset = BagOfPitches(split="validation")

    # get composers with at least 5 pieces in the database
    count = dataset.df.groupby(["composer"]).size()
    composers_with_most_data = count[count.values > 5]
    composers_to_check = composers_with_most_data.index.tolist()
    return composers_to_check


def initialize_wandb(cfg: DictConfig, classnames: list[str]):
    # initialize experiment on WandB with unique run id
    run_id = str(uuid.uuid1())[:8]
    name = classnames[0].replace(" ", "_").lower()
    for classname in classnames[1:]:
        name += f"-{classname.replace(' ', '_').lower()}"
    wandb.init(
        project=cfg.logger.project,
        name=f"{name}-{run_id}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run_id


def run_experiment(cfg: DictConfig, classnames: list[str]) -> nn.Module:
    """
    Run whole experiment using the provided configuration and classnames from the dataset.

    Parameters:
        cfg (DictConfig): Configuration containing hyperparameters and model specifications.
        classnames (list[str]): A list containing the class names corresponding to their indices.

    Returns:
        PitchSeqNN: The trained PitchSeqNN model.
        int: Run id.

    This function initializes an experiment on WandB, initializes the model, dataloaders and optimizer, and performs
    the training and validation loops. It logs the training and validation statistics to WandB.
    """
    # loading data
    dataset = BagOfPitches(selected_composers=classnames)
    v_dataset = BagOfPitches(split="validation", selected_composers=classnames)
    train_dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    v_dataloader = DataLoader(v_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # initialize experiment on WandB with unique run id
    run_id = initialize_wandb(cfg, classnames)

    # initialize model, optimizer and loss criterion
    model = PitchSeqNN(cfg.model.hidden_layers, 128, len(classnames))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(cfg.train.num_epochs), desc="Training started!")

    # training loop
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

        wandb.log({**train_stats, **v_stats})

        bar = (
            "loss={t_loss:.3f}, acc={t_acc:.2f}, f1_score={f1:.2f}, val_loss={v_loss:.3f}, val_acc={v_acc:.2f}, "
            "val_f1_score={v_f1:.2f}"
        ).format(
            t_loss=train_stats["train/loss"],
            t_acc=train_stats["train/accuracy"],
            f1=train_stats["train/f1_score"],
            v_loss=v_stats["test/loss"],
            v_acc=v_stats["test/accuracy"],
            v_f1=v_stats["test/f1_score"],
        )
        pbar.set_description(bar)
    wandb.finish()
    path = "models/"
    for classname in classnames:
        path += f"{classname.lower().replace(' ', '_')}-"
    path += f"{run_id}.pt"

    torch.save(
        {
            "epoch": cfg.train.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    return model, run_id


def training_epoch(
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    criterion: Callable,
):
    correct = 0
    running_loss = 0
    truths = torch.tensor([])
    preds = torch.tensor([])

    for batch in train_dataloader:
        inputs = batch["data"]
        labels = batch["label"]

        optimizer.zero_grad()

        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()

        preds = torch.concatenate((preds, pred.argmax(1)))
        truths = torch.concatenate((truths, labels))
        optimizer.step()

        correct += (pred.argmax(1) == labels).sum().item()

        running_loss += loss.item()

    running_loss = running_loss / len(train_dataloader)
    accuracy = correct / len(train_dataloader.dataset)
    f1 = f1_score(truths, preds, average="weighted")
    stats = {"train/loss": running_loss, "train/accuracy": accuracy, "train/f1_score": f1}
    return stats


def validation_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: Callable,
):
    v_loss = 0
    correct = 0
    truths = torch.tensor([])
    preds = torch.tensor([])
    for batch in loader:
        inputs = batch["data"]
        labels = batch["label"]

        pred = model(inputs)
        loss = criterion(pred, labels)

        truths = torch.concatenate((truths, labels))
        preds = torch.concatenate((preds, pred.argmax(1)))

        v_loss += loss.item()
        correct += (pred.argmax(1) == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    v_loss = v_loss / len(loader.dataset)
    f1 = f1_score(truths, preds, average="weighted")
    stats = {"test/loss": v_loss, "test/accuracy": accuracy, "test/f1_score": f1}
    return stats


if __name__ == "__main__":
    main()
