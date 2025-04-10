import torch
import numpy as np

def create_dataset(dataset, seq_len):
    dataX, dataY = [], []
    for i in range(len(dataset) - seq_len):
        a = dataset[i:(i + seq_len)]
        dataX.append(a)
        dataY.append(dataset[i + seq_len])

    return torch.tensor(np.array(dataX), dtype=torch.float32), torch.tensor(np.array(dataY), dtype=torch.float32)
