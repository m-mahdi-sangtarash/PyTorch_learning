import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data = datasets.FashionMNIST(root='./train',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())

test_data = datasets. FashionMNIST(root='./test',
                                   train=False,
                                   download=True,
                                   transform=transforms.ToTensor())

class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


model = SimpleNN()

