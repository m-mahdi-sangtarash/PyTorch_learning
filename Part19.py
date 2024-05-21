import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt

SEED = 101

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


train_transforms = transforms.Compose([transforms.ToTensor,
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])

test_transforms = transforms.Compose([transforms.ToTensor, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])

train_data = datasets.CIFAR10(
    root='./train_data',
    train=True,
    download=True,
    transform=train_transforms
)

test_data = datasets.CIFAR10(
    root='./test_data',
    train=False,
    download=True,
    transform=test_transforms
)

# Training and validation sets

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])


valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms


# Part 19 - PyTorch training Course

# Creating train, validation and test dataloaders

BATCH_SIZE = 64

train_loader = data.DataLoader(train_data,
                               shuffle=True,
                               batch_size=BATCH_SIZE)

valid_loader = data.DataLoader(valid_data,
                               batch_size=BATCH_SIZE)

test_loader = data.DataLoader(test_data,
                              batch_size=BATCH_SIZE)


# LeNet Implementation

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_dim)


    def forward(self, x):
        # Feature Extraction
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)

        x = x.view(-1, 16 * 5 * 5)

        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
