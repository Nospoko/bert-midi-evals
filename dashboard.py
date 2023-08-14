import os
import torch
import hydra
from eval import load_checkpoint
import streamlit as st
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from model import PitchSeqNN
from data.dataset import BagOfPitches
from utils import test_model, piece_av_files
import numpy as np
import pandas as pd
from fortepyan import MidiPiece


def main():
    # choose which model to plot samples of
    path = "models/" + st.selectbox(label="model", options=os.listdir("models"))
    option = st.selectbox(label="right/wrong predictions", options=("right", "wrong"))
    checkpoint = torch.load(path)
    model_config = checkpoint["config"]

    # initialize model from checkpoint
    model = PitchSeqNN(model_config["model"]["hidden_layers"], 128, len(model_config["model"]["composers"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    classnames = model_config["model"]["composers"]

    test_data = BagOfPitches(split="test", selected_composers=classnames)
    test_dataloader = DataLoader(test_data, batch_size=model_config["train"]["batch_size"], shuffle=True)

    # test model and get predicted samples
    if option == "wrong":
        bad_preds = wrong_predictions(model, test_dataloader)
        create_dashboard(bad_preds)
    else:
        good_preds = right_predictions(model, test_dataloader)
        create_dashboard(good_preds)


def wrong_predictions(model: nn.Module, test_data: DataLoader) -> dict:
    # get data from testing
    data = test_model(model, test_data=test_data)

    # select samples with false predictions
    filtered_indices = [index for index, value in enumerate(data["pred"]) if value != data["label"][index]]
    bad_predictions = {key: [values[index] for index in filtered_indices] for key, values in data.items()}
    return bad_predictions


def right_predictions(model: nn.Module, test_data: DataLoader) -> dict:
    # get data from testing
    data = test_model(model, test_data=test_data)
    # select samples with false predictions
    filtered_indices = [index for index, value in enumerate(data["pred"]) if value == data["label"][index]]
    good_predictions = {key: [values[index] for index in filtered_indices] for key, values in data.items()}
    return good_predictions


def create_dashboard(preds):
    cols = st.columns(2)
    n_samples = 20
    it = 0
    # choose random index to plot
    indexes = np.random.randint(low=0, high=len(preds["label"]) - 1, size=n_samples)
    print(indexes)
    for index in indexes:
        # choose column
        col = it % 2
        it += 1
        # dataframe with notes
        notes = preds["notes"][index]

        # normalize notes to plot filled pianoroll
        start_time = np.min(notes["start"])
        notes["start"] -= start_time
        notes["end"] -= start_time

        piece = MidiPiece(pd.DataFrame(notes))
        piece.source["midi_filename"] = preds["midi_filename"][index]
        piece.source["composer"] = preds["composer"][index]
        piece.source["title"] = preds["title"][index]
        paths = piece_av_files(piece)
        with cols[col]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(piece.source)


if __name__ == "__main__":
    main()
