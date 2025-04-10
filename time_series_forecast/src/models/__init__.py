from .CustomLSTM import CustomLSTM
from .sLSTM import sLSTM
from .mLSTM import mLSTM
from .xLSTM import xLSTM

def build_models(input_size, head_size=32, num_heads=2):
    return {
        "xLSTM": xLSTM(input_size=input_size, head_size=head_size, num_heads=num_heads, layers='msm', batch_first=True),
        "LSTM": CustomLSTM(input_size=input_size, hidden_size=head_size, batch_first=True),
        "sLSTM": sLSTM(input_size=input_size, head_size=head_size, num_heads=num_heads, batch_first=True),
        "mLSTM": mLSTM(input_size=input_size, head_size=head_size, num_heads=num_heads, batch_first=True)
    }