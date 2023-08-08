import torch.nn as nn
import torch.nn.functional as F


class PitchSeqNN(nn.Module):
    """
    A sequential neural network model for processing pitch data.

    Parameters:
        layers (list): A list specifying the number of neurons in each layer of the neural network.
                       The first element should represent the input size, the last element represents
                       the output size, and the intermediate elements represent the hidden layer sizes.
    """

    def __init__(self, layers: list):
        super(PitchSeqNN, self).__init__()

        self.input_layer = nn.Linear(layers[0], layers[1])
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(1, len(layers) - 2)])
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    model = PitchSeqNN([64, 128, 64, 32, 2])
    params = list(model.parameters())
    print(params[i].size() for i in range(len(params)))
