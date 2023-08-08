# import itertools

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


class ComposerClassificationDataset:
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
        data = torch.tensor(histogram).float()
        composer = self.samples[idx]["composer"]
        label = self.selected_composers.index(composer)

        sample = {
            "data": data,
            "label": label,
            "composer": composer,
        }

        return sample


if __name__ == "__main__":
    train_dataset = BagOfPitches(split="train")
    count = train_dataset.df.groupby(["composer"]).size()
    composers_with_most_data = count[count.values > 10]
    composers_to_check = composers_with_most_data.index.tolist()
    dataset_with_all_composers = BagOfPitches(split="train", selected_composers=composers_to_check)
    df = pd.DataFrame()
    maxes = []
    i = 0
    pbar = tqdm(dataset_with_all_composers)
    for sample in pbar:
        df = pd.concat([df, pd.DataFrame(sample)], ignore_index=True)
        maxim = torch.max(sample["data"])
        maxes.append(maxim)
        i += 1
        pbar.set_description(str(i))

    print(np.max(maxes))
    print(df.groupby("composer").size())
    print(df.head())
    # print(composers_to_check)
    # for comp1, comp2 in itertools.combinations(composers_to_check, r=2):
    # print([comp1, comp2])

    # pieces of composers:
    # Alexander Scriabin          22 # val: None
    # Claude Debussy              37
    # Domenico Scarlatti          18
    # Felix Mendelssohn           24 #test: 1 val: 3
    # Franz Liszt                 93
    # Franz Schubert             159
    # Frédéric Chopin            145
    # Johann Sebastian Bach      114
    # Johannes Brahms             20 #test: 1 val: 5
    # Joseph Haydn                29
    # Ludwig van Beethoven       110
    # Robert Schumann             33
    # Sergei Rachmaninoff         29
    # Wolfgang Amadeus Mozart     27
