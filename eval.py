import os.path

import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch.optim
import pandas as pd
import torch.nn as nn
from model import PitchSeqNN
from data.dataset import BagOfPitches
import utils as U
from train import validation_epoch


@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # load checkpoint of desired run
    checkpoint = load_checkpoint(cfg.run_id)
    model_config = checkpoint["config"]

    # initialize model from checkpoint
    model = PitchSeqNN(model_config["model"]["hidden_layers"], 128, len(model_config["model"]["composers"]))
    model.load_state_dict(checkpoint["model_state_dict"])

    # initialize dataloader
    test_data = BagOfPitches(split="test", selected_composers=model_config["model"]["composers"])
    test_dataloader = DataLoader(test_data, batch_size=model_config["train"]["batch_size"], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    # evaluate model on test data
    print(validation_epoch(model=model, loader=test_dataloader, criterion=criterion))


def load_checkpoint(run_id):
    # find path with desired run_id
    path = None
    for file in os.listdir("models"):
        for word in file.split('-'):
            if word == f"{run_id}.pt":
                path = file
                break
    if path is None:
        print("no run with this id found")
        return None
    path = "models/" + path
    print(path)
    # load checkpoint from path
    checkpoint = torch.load(path)
    return checkpoint


if __name__ == "__main__":
    main()
