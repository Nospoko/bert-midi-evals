import torch.nn as nn
import torch.nn.functional as F
class ComposerClassifier(nn.Module):
    def __init__(self):
        super(ComposerClassifier, self).__init__()
        self.dense1 = nn.Linear(128, 256)
        self.dense2 = nn.Linear(256, 512)
        self.dense3 = nn.Linear(512, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = self.dense5(x)
        return x


if __name__ == '__main__':
    model = ComposerClassifier()
    params = list(model.parameters())
    print(params[i].size() for i in range(len(params)))
