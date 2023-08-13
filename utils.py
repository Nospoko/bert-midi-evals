import os
import itertools
from typing import Optional

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan import MidiPiece
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from fortepyan.audio import render as render_audio


def plot_loss_curves(history: pd.DataFrame):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
      history: pd.Dataframe with ['loss', 'accuracy', 'val_loss', 'val_accuracy'] columns
    """
    loss = history["loss"]
    val_loss = history["val_loss"]

    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    epochs = range(len(history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def make_confusion_matrix(y_true, y_pred, classes: Optional[list[str]] = None, figsize=(16, 10), text_size=15, norm=False):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels.
      y_pred: Array of predicted labels.
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).

    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalization
    n_classes = cm.shape[0]  # find the number of classes

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),  # create enough axis slots for each class
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.0

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(
                j,
                i,
                f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size,
            )
        else:
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size,
            )
    plt.show()
    # path = str()
    # for name in classes:
    #     path += name.replace(' ', '-').lower()
    # path += '.png'
    # plt.savefig(f"plots/{path}")


def test_model(model: nn.Module, test_data: DataLoader):
    """
    Creates a shuffled copy of the input dataset in as a dict, with additional 'pred' column for model predictions.

    Args:
        model (PitchSeqNN): The trained ComposerClassifier model to be evaluated.
        test_data (DataLoader): Data for the model to be tested on
    Returns:
        dict: all sample data updated with "pred" key.
    """
    # containers for predictions and truths:
    predicted = torch.tensor([])
    true = torch.tensor([])
    data = {}
    with torch.no_grad():
        for batch in test_data:
            labels = batch["label"]
            out = model(batch["data"])
            preds = out.argmax(1)
            all_info = batch.copy()
            # updating sample data with predictions
            all_info.update({"pred": preds})

            # reshaping notes so that there is one list of notes for each sample instead of batch
            notes = [
                {key: [lst[i].item() for lst in lsts] for key, lsts in batch["notes"].items()} for i in range(len(batch["data"]))
            ]
            all_info["notes"] = notes

            # merging dictionaries to store all the data in one dict
            data = merge_dictionary(data, all_info)
            predicted = torch.concatenate((predicted, preds))
            true = torch.concatenate((true, labels))

    return data


def piece_av_files(piece: MidiPiece) -> dict:
    # stolen from Tomek
    midi_file = os.path.basename(piece.source["midi_filename"])
    mp3_path = midi_file.replace(".midi", ".mp3")
    mp3_path = os.path.join("tmp", mp3_path)
    if not os.path.exists(mp3_path):
        render_audio.midi_to_mp3(piece.to_midi(), mp3_path)

    pianoroll_path = midi_file.replace(".midi", ".png")
    pianoroll_path = os.path.join("tmp", pianoroll_path)
    if not os.path.exists(pianoroll_path):
        ff.view.draw_pianoroll_with_velocities(piece)
        plt.tight_layout()
        plt.savefig(pianoroll_path)
        plt.clf()

    paths = {
        "mp3_path": mp3_path,
        "pianoroll_path": pianoroll_path,
    }
    return paths


def merge_dictionary(dict_1, dict_2):
    # merge dictionaries by concatenating lists in corresponding keys
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.concatenate((value, dict_1[key]))
    return dict_3
