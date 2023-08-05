import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset


class ComposerClassificationDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        sequence_length: int = 64,
        # chose these composers due to the similar total number of notes in maestro dataset
        selected_composers: list[str] = ["Frédéric Chopin", "Ludwig van Beethoven"],
    ):
        self.sequence_length = sequence_length
        self.selected_composers = selected_composers
        self.dataset = load_dataset("roszcz/maestro-v1-sustain", split=split)
        self.df = pd.DataFrame(self.dataset.select_columns(["composer", "title"]))
        self._build()

    def _build(self):
        # This is much faster than filtering HF datasets directly
        ids = self.df.composer.isin(self.selected_composers)
        print("Records with selected composers:", ids.sum())

        composer_dataset = self.dataset.select(np.where(ids)[0])

        self.samples = []
        for record in composer_dataset:
            self.samples += self._process_record(record)

    def _process_record(self, record: dict) -> list[dict]:
        # Separate MIDI notes from maestro metadata
        df = pd.DataFrame(record.pop("notes"))

        start = 0
        finish = self.sequence_length
        new_records = []
        while finish < df.shape[0]:
            part = df[start:finish].reset_index(drop=True)
            new_record = {"notes": part.to_dict(orient="list")}
            # Keep the metadata
            new_record |= record
            new_records.append(new_record)
            start += self.sequence_length
            finish += self.sequence_length

        return new_records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class BagOfPitches(ComposerClassificationDataset):
    def __getitem__(self, idx: int):
        pitch = self.samples[idx]["notes"]["pitch"]

        histogram = [pitch.count(it) for it in range(128)]

        # For seq len 64, 44 is the maximum count of a note in the train split, wtf
        data = torch.tensor(histogram).float() / 44
        composer = self.samples[idx]["composer"]
        label = self.selected_composers.index(composer)

        sample = {
            "data": data,
            "label": label,
            "composer": composer,
        }

        return sample
