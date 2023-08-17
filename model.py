import torch.nn as nn
import torch.nn.functional as F


class PitchSeqNN(nn.Module):
    """
    A sequential neural network model for processing pitch data.

    Parameters:
        hidden_layers (list): A list specifying the number of neurons in each hidden layer of the neural network.
        input_size (int): The input size.
        num_classes (int): The output size.
    """

    def __init__(self, hidden_layers: list, input_size: int, num_classes: int):
        super(PitchSeqNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        list_of_layers = [nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in range(0, len(hidden_layers) - 1)]
        self.layers = nn.ModuleList(list_of_layers)
        self.output_layer = nn.Linear(hidden_layers[-1], num_classes)

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
