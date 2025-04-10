import torch.nn as nn
from models.xLSTM import sLSTM, mLSTM, xLSTM

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, state=None):
        outputs, state = self.lstm(x, state)
        return self.fc(outputs), state


def build_models(input_size, head_size=32, num_heads=2):
    return {
        "xLSTM": xLSTM(input_size=input_size, head_size=head_size, num_heads=num_heads, layers='msm', batch_first=True),
        "LSTM": CustomLSTM(input_size=input_size, hidden_size=head_size, batch_first=True),
        "sLSTM": sLSTM(input_size=input_size, head_size=head_size, num_heads=num_heads, batch_first=True),
        "mLSTM": mLSTM(input_size=input_size, head_size=head_size, num_heads=num_heads, batch_first=True)
    }