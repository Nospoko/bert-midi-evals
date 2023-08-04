import pandas as pd
import torch.optim
import numpy as np
from src.data.dataset import *
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ComposerClassifier


def prepare_dataset(dataset):
    def divide(lst):
        return [num / 127. for num in lst]
    df = pd.DataFrame(dataset.samples)
    df = df[['notes', 'composer']]
    df['composer'] = df['composer'].apply(
        lambda x: 0 if x == dataset.selected_composers[0] else 1)
    df['notes'] = df['notes'].apply(lambda x: x['pitch'])
    df['notes'] = df['notes'].apply(divide)
    print(df.groupby('composer').size())
    samples = [(torch.tensor(row['notes'], dtype=torch.float32), torch.tensor(row['composer'])) for
               _, row in df.iterrows()]
    while len(samples) % 16 != 0:
        samples.pop(-1)
    dataset.samples = samples

def train():
    train_data = ComposerClassificationDataset(split='train')
    prepare_dataset(train_data)
    print(train_data.__getitem__(0))
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    n_epochs = 100
    model = ComposerClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            # optimizer.zero_grad()
            pred = model(inputs)
            # print(pred)
            # print(labels)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Finished Training')
    path = "model1.pth"
    torch.save(model.state_dict(), path)
def test_model(model, path):
    test_data = ComposerClassificationDataset(split='test')
    prepare_dataset(test_data)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)
    model.load_state_dict(torch.load(path))
    dataiter = iter(test_dataloader)
    notes, labels = next(dataiter)
    outputs = model(notes)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{test_data.selected_composers[predicted[j]]:5s}'
                                  for j in range(4)))
    correct_pred = {classname: 0 for classname in test_data.selected_composers}
    total_pred = {classname: 0 for classname in test_data.selected_composers}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            notes, labels = data
            out = model(notes)
            # print(out)
            _, preds = torch.max(out, 1)
            print(out)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[test_data.selected_composers[label]] += 1
                total_pred[test_data.selected_composers[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print(f'Accuracy of the network on the test data: {100 * correct // total} %')
    print('correctly predicted :' + str(correct_pred))


def main():

    # train()
    model = ComposerClassifier()
    test_model(model, 'model1.pth')

if __name__ == '__main__':
    main()
