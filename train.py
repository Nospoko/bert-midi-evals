import uuid
import os.path
import itertools
from typing import Callable

import hydra
import torch.optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from model import PitchSeqNN
from data.dataset import BagOfPitches
from utils import test_model, make_confusion_matrix


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):  # base_acc
    composers = [
        ["Alexander Scriabin", "Johann Sebastian Bach"],  # 0.58
        ["Franz Liszt", "Johann Sebastian Bach"],  # 0.76
        ["Alexander Scriabin", "Sergei Rachmaninoff"],  # 0.63
        ["Robert Schumann", "Sergei Rachmaninoff"],  # 0.54
        ["Robert Schumann", "Ludwig van Beethoven"],  # 0.59
        ["Alexander Scriabin", "Johann Sebastian Bach", "Franz Liszt", "Ludwig van Beethoven"],
    ]
    for classnames in composers:
        print(classnames)
        print(test_and_confusion_matrix_main(cfg, classnames))


def test_and_confusion_matrix_main(cfg: DictConfig, classnames: list[str]):
    """
    Evaluate a neural network model on test data, generate a confusion matrix,
    and return the validation loss and accuracy.

    Args:
        cfg (DictConfig): Configuration settings for the experiment.
        classnames (list[str]): List of class names or labels.

    Returns:
        Dict: Validation loss and accuracy achieved on the test data.
    """
    path = classnames[0].replace(" ", "_").lower()
    for classname in classnames[1:]:
        path += f"-{classname.replace(' ', '_').lower()}"
    path += ".pth"
    if not os.path.isfile(f"models/{path}"):
        model = run_experiment(cfg, "MIDI-18-bag-of-pitches-pairs", classnames=classnames)
        if cfg.model.save_model == "y":
            torch.save(model.state_dict(), f"models/{path}.pth")
    else:
        model = PitchSeqNN(cfg.model.hidden_layers, 128, len(classnames))
        model.load_state_dict(torch.load(f"models/{path}.pth"))

    test_data = BagOfPitches(split="test", selected_composers=classnames)
    test_dataloader = DataLoader(test_data, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    true, pred = test_model(model, test_data=test_dataloader)
    make_confusion_matrix(y_true=true, y_pred=pred, classes=classnames)
    return validation_epoch(loader=test_dataloader, model=model, criterion=nn.CrossEntropyLoss())


def one_run(cfg: DictConfig):
    """
    Function for testing
    """
    dataset = BagOfPitches(split="validation")
    # get composers to classify against each other
    count = dataset.df.groupby(["composer"]).size()
    composers_with_most_data = count[count.values > 5].index.tolist()

    run_experiment(cfg, classnames=composers_with_most_data)


def composer_comparison_main(cfg: DictConfig):
    """
    Finds composers with at least 5 pieces in the validation split,
    then trains model for each pair by calling run_experiment

    Args:
        cfg (DictConfig): DictConfig passed from hydra.main function containing hyperparameters and model specification.
    """

    composers_to_check = find_composers_to_check()
    composers = [pair for pair in itertools.combinations(composers_to_check, r=2)]
    print(f"pairs to check: {len(composers)}")
    project = "MIDI-18-bag-of-pitches-pairs"
    # container for trained models
    models = []

    # train model for each pair
    for pair in composers:
        print(f"{pair[0]} vs {pair[1]}")
        model = run_experiment(cfg=cfg, project=project, classnames=pair)
        models.append((model, pair))
        first_composer = pair[0].replace(" ", "_").lower()
        other_composer = pair[1].replace(" ", "_").lower()
        path = f"models/{first_composer}-{other_composer}.pth"
        if cfg.model.save_model == "y":
            torch.save(model.state_dict(), path)


def find_composers_to_check():
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


def initialize_wandb(cfg: DictConfig, project: str, classnames: list[str]):
    # initialize experiment on WandB with unique run id
    run_id = str(uuid.uuid1())[:8]
    name = classnames[0].replace(" ", "_").lower()
    for classname in classnames[1:]:
        name += f"-{classname.replace(' ', '_').lower()}"
    run = wandb.init(
        project=project,
        name=f"{name}-{run_id}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def run_experiment(cfg: DictConfig, project, classnames: list[str]):
    """
    Run an experiment using the provided configuration and classnames from the dataset.

    Parameters:
        cfg (DictConfig): Configuration containing hyperparameters and model specifications.
        classnames (list[str]): A list containing the class names corresponding to their indices.

    Returns:
        PitchSeqNN: The trained PitchSeqNN model.

    This function initializes an experiment on WandB, initializes the model, dataloaders and optimizer, and performs
    the training and validation loops. It logs the training and validation statistics to WandB.
    """
    # loading data
    dataset = BagOfPitches(selected_composers=classnames)
    v_dataset = BagOfPitches(split="validation", selected_composers=classnames)
    train_dataloader = DataLoader(dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    v_dataloader = DataLoader(v_dataset, batch_size=cfg.hyperparameters.batch_size)

    # initialize experiment on WandB with unique run id
    run = initialize_wandb(cfg, project, classnames)

    # initialize model, optimizer and loss criterion
    model = PitchSeqNN(cfg.model.hidden_layers, 128, len(classnames))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(cfg.hyperparameters.num_epochs), desc="Training started!")

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

        run.log({**train_stats, **v_stats})

        bar = "loss={t_loss:.3f}, acc={t_acc:.2f}, val_loss={v_loss:.3f}, val_acc={v_acc:.2f}".format(
            t_loss=train_stats["loss"],
            t_acc=train_stats["accuracy"],
            v_loss=v_stats["val_loss"],
            v_acc=v_stats["val_accuracy"],
        )
        pbar.set_description(bar)
    run.finish()
    return model


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

    accuracy = correct / len(loader.dataset)
    stats = {
        "val_loss": v_loss,
        "val_accuracy": accuracy,
    }
    return stats


if __name__ == "__main__":
    main()
