import torch
import numpy as np
import pandas as pd
from datasets import load_dataset


class ComposerClassificationDataset:
    def __init__(
        self,
        split: str = "train",
        sequence_length: int = 64,
        # chose these composers due to the similar total number of notes in maestro dataset
        selected_composers: list[str] = ["Frédéric Chopin", "Johann Sebastian Bach"],
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
        data = torch.tensor(histogram).float() / 64  # found out there is a sample with 64 same notes :O
        composer = self.samples[idx]["composer"]
        label = self.selected_composers.index(composer)

        sample = {
            "data": data,
            "label": label,
            "composer": composer,
        }

        return sample

    # samples for each of chosen composers on train:
    # composer
    # Alexander Scriabin          294272
    # Claude Debussy              310528
    # Domenico Scarlatti           67712
    # Felix Mendelssohn           358272
    # Franz Liszt                1489536
    # Franz Schubert             2096896
    # Frédéric Chopin            1508096
    # Johann Sebastian Bach       613120
    # Johannes Brahms             419328
    # Joseph Haydn                223232
    # Ludwig van Beethoven       1441152
    # Robert Schumann             897024
    # Sergei Rachmaninoff         305280
    # Wolfgang Amadeus Mozart     230528
    # on validation:
    # composer
    # Alexander Scriabin          40960
    # Felix Mendelssohn           22656
    # Franz Liszt                182144
    # Franz Schubert              17536
    # Frédéric Chopin            179200
    # Johann Sebastian Bach       56576
    # Johannes Brahms            107136
    # Joseph Haydn                29056
    # Ludwig van Beethoven       198144
    # Robert Schumann            138880
    # Sergei Rachmaninoff        116608
    # Wolfgang Amadeus Mozart     54784
    # on test:
    # composer
    # Alexander Scriabin          33024
    # Claude Debussy              68352
    # Domenico Scarlatti          39168
    # Felix Mendelssohn           16512
    # Franz Liszt                230272
    # Franz Schubert             255616
    # Frédéric Chopin            224640
    # Johann Sebastian Bach       62848
    # Johannes Brahms             26624
    # Joseph Haydn                31488
    # Ludwig van Beethoven       228480
    # Robert Schumann            112256
    # Sergei Rachmaninoff         70144
    # Wolfgang Amadeus Mozart     34688
    #
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
