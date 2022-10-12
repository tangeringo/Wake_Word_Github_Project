from torch import nn
import torch


# DEFINING HYPER PARAMETERS
model_params = {
    "num_classes": 1, "input_size": 40, "hidden_size": 128,
    "num_layers": 1, "bidirectional": False
}


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, num_classes, bidirectional, device='cpu'):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.directions = 2 if bidirectional else 1
        self.device = device
        self.layer_norm = nn.LayerNorm(input_size)  # normalize the data

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * self.directions, num_classes)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return (torch.zeros(n*d, batch_size, hs).to(self.device),
                torch.zeros(n*d, batch_size, hs).to(self.device))

    def forward(self, x):
        x = self.layer_norm(x)
        hidden = self._init_hidden(x.size()[1])
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn)
        return out
