import torch.nn as nn
import torch.nn.functional as F


class PitchSeqNN(nn.Module):
    """
    Simple sequential neural network for classifying pitch tensors of size (128, 1) into two classes.
    (performs poorly - pretty much randomly)
    """

    def __init__(self):
        super(PitchSeqNN, self).__init__()
        self.dense1 = nn.Linear(64, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 256)
        self.dense4 = nn.Linear(256, 64)
        self.dense5 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = self.dense5(x)
        return x


class PitchSeqNNv2_64(nn.Module):
    """
    Simple sequential neural network for classifying pitch tensors of size (64, 1) into two classes.
    (performs poorly - pretty much randomly)
    """

    def __init__(self):
        super(PitchSeqNNv2_64, self).__init__()
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class PitchSeqNNv3_64(nn.Module):
    """
    Simple sequential neural network for classifying pitch tensors of size (64, 1) into two classes.
    """

    def __init__(self):
        super(PitchSeqNNv3_64, self).__init__()
        self.dense1 = nn.Linear(64, 256)
        self.dense2 = nn.Linear(256, 16)
        self.dense3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class PitchSeqNNv4_64(nn.Module):
    """
    Simple sequential neural network for classifying pitch tensors of size (64, 1) into two classes.
    """

    def __init__(self):
        super(PitchSeqNNv4_64, self).__init__()
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense4 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        return x


class PitchSeqNNv5_32(nn.Module):
    """
    Simple sequential neural network for classifying pitch tensors of size (32, 1) into two classes.
    """

    def __init__(self):
        super(PitchSeqNNv5_32, self).__init__()
        self.dense1 = nn.Linear(32, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


if __name__ == "__main__":
    model = PitchSeqNN()
    params = list(model.parameters())
    print(params[i].size() for i in range(len(params)))
