import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, state=None):
        outputs, state = self.lstm(x, state)
        outputs = self.fc(outputs)
        return outputs, state
