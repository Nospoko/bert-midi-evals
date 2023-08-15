import os

import torch
import numpy as np
import pandas as pd
import streamlit as st
from datasets import Dataset
from fortepyan import MidiPiece
from torch.utils.data import DataLoader

from model import PitchSeqNN
from data.dataset import BagOfPitches
from utils import piece_av_files, samples_with_pred


def main():
    # choose which model to plot samples of
    path = "models/" + st.selectbox(label="model", options=os.listdir("models"))
    option = st.selectbox(label="display mode", options=("accuracy per piece", "right", "wrong"))

    checkpoint = torch.load(path)
    model_config = checkpoint["config"]

    # initialize model from checkpoint
    model = PitchSeqNN(model_config["model"]["hidden_layers"], 128, len(model_config["model"]["composers"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    classnames = model_config["model"]["composers"]

    test_data = BagOfPitches(split="test", selected_composers=classnames)
    # I use unshuffled data so I can plot each sample individually (indexes are unambiguous)
    test_dataloader = DataLoader(test_data, batch_size=model_config["train"]["batch_size"], shuffle=False)
    # get predictions for each sample
    pieces_data = samples_with_pred(model, test_dataloader)

    if option == "wrong":
        bad_preds = wrong_predictions(pieces_data)
        create_samples_dashboard(bad_preds, classnames)
    elif option == "right":
        good_preds = right_predictions(pieces_data)
        create_samples_dashboard(good_preds, classnames)
    elif option == "accuracy per piece":
        acc_per_piece = evaluate_pieces(pieces_data)
        create_pieces_dashboard(test_data.dataset, acc_per_piece)


def wrong_predictions(samples_data: dict) -> dict:
    # select samples with false predictions
    filtered_indices = [index for index, value in enumerate(samples_data["pred"]) if value != samples_data["label"][index]]
    bad_predictions = {key: [values[index] for index in filtered_indices] for key, values in samples_data.items()}
    return bad_predictions


def right_predictions(samples_data: dict) -> dict:
    # select samples with false predictions
    filtered_indices = [index for index, value in enumerate(samples_data["pred"]) if value == samples_data["label"][index]]
    good_predictions = {key: [values[index] for index in filtered_indices] for key, values in samples_data.items()}
    return good_predictions


def evaluate_pieces(samples_data: dict) -> dict:
    # select columns from samples_data
    keys = ["midi_filename", "pred", "label"]
    info = {key: samples_data[key] for key in keys}
    data_df = pd.DataFrame(info)
    pieces = data_df.groupby(["midi_filename"])
    acc_per_piece = {}
    # calculate accuracy for each piece
    for piece in pieces.groups.values():
        new_dict = {info["midi_filename"][piece[0]]: (info["pred"][piece] == info["label"][piece]).sum() / len(piece)}
        acc_per_piece.update(new_dict)
    return acc_per_piece


def create_pieces_dashboard(dataset: Dataset, accuracy_per_piece: dict):
    pieces = []
    # for each pair ('midi_filename', 'accuracy')
    for key, value in accuracy_per_piece.items():
        # select record with filename from dict
        record = dataset.filter(lambda rec: rec["midi_filename"] == key)[0]
        midi_piece = MidiPiece.from_huggingface(record)
        # append accuracy to MidiPiece source dict
        midi_piece.source["accuracy"] = value
        pieces.append(midi_piece)
    cols = st.columns(2)
    n_samples = len(pieces)

    # plot pieces
    for it in range(n_samples):
        col = it % 2
        paths = piece_av_files(pieces[it])
        with cols[col]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(pieces[it].source)


def create_samples_dashboard(preds, classnames):
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
        name = preds["midi_filename"][index].split("/")[0] + "/" + str(index)
        piece.source["midi_filename"] = name + os.path.basename(preds["midi_filename"][index])
        piece.source["title"] = preds["title"][index]
        piece.source["composer"] = preds["composer"][index]
        piece.source["predicted"] = classnames[preds["pred"][index]]
        paths = piece_av_files(piece)
        with cols[col]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(piece.source)


if __name__ == "__main__":
    main()
