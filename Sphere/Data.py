import numpy as np
from torch.utils.data import Dataset
import torch
import datetime as dt
import  pandas as pd
from torch import nn
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


class DatasetLoader(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.x, self.y = make_blobs(n_samples=10_000, n_features=20, random_state = 0, centers=5)
        y = []
        m = max(self.y)
        for Y in self.y:
            y.append([1 if k==m else 0 for k in range(m+1)])
        #self.y = np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(self.y[index], dtype=torch.float).unsqueeze(0)