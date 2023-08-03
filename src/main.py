import pandas as pd
import torch.optim
import numpy as np
from src.data.dataset import *
import torch.nn as nn
from torch.utils.data import DataLoader


def prepare_dataset(dataset):
    df = pd.DataFrame(dataset.samples)
    df = df[['notes', 'composer']]
    df['composer'] = df['composer'].apply(lambda x: [1.0, 0.0] if x == "Frédéric Chopin" else [0.0, 1.0])
    df['notes'] = df['notes'].apply(lambda x: x['pitch'])

    samples = [(row['notes'], row['composer']) for _, row in df.iterrows()]
    dataset.samples = samples


def main():
    train_data = ComposerClassificationDataset(split='train')
    test_data = ComposerClassificationDataset(split='test')
    prepare_dataset(train_data)
    prepare_dataset(test_data)
    print(train_data.__getitem__(0))
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()