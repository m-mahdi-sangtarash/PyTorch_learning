import torch
import torch.nn as nn


class SimpleNNsequential(nn.Module):
    def __init__(self):
        super.__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(in_features=784, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )


    def forward(self, x):
        x = self.flatten(x)
        x = self.sequential(x)

        return x


seqential_model = SimpleNNsequential()