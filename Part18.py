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