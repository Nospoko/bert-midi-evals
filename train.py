import uuid
import itertools
from typing import Callable

import hydra
import torch.optim
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from model import PitchSeqNN
from data.dataset import BagOfPitches


def run_one_experiment(cfg: DictConfig, train_data, val_data):
    run_id = str(uuid.uuid1())[:8]
    wandb.init(
        project="MIDI-18-bag-of-pitches",
        name=f"{cfg.logger.run_name}-{run_id}",
        config={
            "learning_rate": cfg.hyperparameters.lr,
            "n_epochs": cfg.hyperparameters.num_epochs,
            "architecture": "NN",
            "batch_size": cfg.hyperparameters.batch_size,
        },
    )

    model = PitchSeqNN(cfg.model.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(range(cfg.hyperparameters.num_epochs), desc="Training started!")
    for epoch in pbar:
        train_stats = training_epoch(
            train_dataloader=train_data,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
        )

        v_stats = validation_epoch(
            loader=val_data,
            model=model,
            criterion=criterion,
        )

        wandb.log({**train_stats, **v_stats})

        bar = "loss={t_loss:.3f}, acc={t_acc:.2f}, val_loss={v_loss:.3f}, val_acc={v_acc:.2f}".format(
            t_loss=train_stats["loss"],
            t_acc=train_stats["accuracy"],
            v_loss=v_stats["val_loss"],
            v_acc=v_stats["val_accuracy"],
        )
        pbar.set_description(bar)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dataset = BagOfPitches()
    v_dataset = BagOfPitches(split="validation")
    train_dataloader = DataLoader(dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    v_dataloader = DataLoader(v_dataset, batch_size=cfg.hyperparameters.batch_size)
    count = dataset.df.groupby(["composer"]).size()
    composers_with_most_data = count[count.values > 10]
    composers_to_check = composers_with_most_data.index.tolist()
    composers = [pair for pair in itertools.combinations(composers_to_check, r=2)]
    print(f"pairs to check: {len(composers)}")
    for pair in composers:
        print(f"{pair[0]} vs {pair[1]}")
        run_id = str(uuid.uuid1())[:8]
        wandb.init(
            project="MIDI-18-different-composers",
            name=f"{pair[0]}-vs-{pair[1]}-{run_id}",
            config={
                "learning_rate": cfg.hyperparameters.lr,
                "n_epochs": cfg.hyperparameters.num_epochs,
                "architecture": "NN",
                "batch_size": cfg.hyperparameters.batch_size,
            },
        )

        model = PitchSeqNN(cfg.model.layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

        criterion = nn.CrossEntropyLoss()
        pbar = tqdm(range(cfg.hyperparameters.num_epochs), desc="Training started!")
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

            bar = "loss={t_loss:.3f}, acc={t_acc:.2f}, val_loss={v_loss:.3f}, val_acc={v_acc:.2f}".format(
                t_loss=train_stats["loss"],
                t_acc=train_stats["accuracy"],
                v_loss=v_stats["val_loss"],
                v_acc=v_stats["val_accuracy"],
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
        "val_loss": v_loss,
        "val_accuracy": accuracy,
    }
    return stats


if __name__ == "__main__":
    main()
