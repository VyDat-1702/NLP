
import torch.nn as nn

class Neuron_NW(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Neuron_NW, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout layer with 20% dropout
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)  # Apply Dropout
        out = self.l2(out)
        return out
