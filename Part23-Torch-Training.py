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
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from torchinfo import summary

SEED = 101

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])

test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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


# Part 20 - PyTorch Course Training videos

OUTPUT_DIM = 10
model = LeNet(OUTPUT_DIM)

summary(model, input_size=(1, 3, 32, 32))

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)


# helper function to calculate accuracy

def calculate_accuracy(pred, y):
    top_pred = pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# Part 21 - PyTorch Course training

# helper function to perform training epochs

def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for features, labels in tqdm(dataloader, desc='Training Phase', leave=False):
        # Sending features and labels to device
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass; making predictions and calculating loss
        pred = model(features)
        loss = criterion(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracies
        acc = calculate_accuracy(pred, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


# Helper function to perform evaluation epoch

def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Evaluation Phase', leave=False):
            features = features.to(device)
            labels = labels.to(device)

            pred = model(features)

            loss = criterion(pred, labels)
            acc = calculate_accuracy(pred, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


# Part 22 - PyTorch Course Videos Training

# Training on train set

EPOCHS = 15

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in trange(EPOCHS, desc="Epoch Number"):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    history['val_loss'].append(valid_loss)
    history['val_acc'].append(valid_acc)

    print(f"Epoch: {epoch + 1:02}")
    print(f"\tTrain Loss: {train_loss:>7.3f} | Training Accuracy: {train_acc * 100:>7.2f}%")
    print(f"\tValidation Loss: {valid_loss:>7.3f} | Validation Accuracy: {valid_acc * 100:>7.2f}%")

# Saving Model

torch.save(model.state_dict(), 'cifar10.pt')

# Evaluating on the Test set
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc * 100:.2f}%")


# Part 23 - PyTorch Course Training

# Plotting Confusion Matrix
def get_preds(model, dataloader, device):
    model.eval()

    image_lst, labels_lst, probs_lst = [], [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)

            pred = model(features)

            prob = F.softmax(pred, dim=-1)

            image_lst.append(features.cpu())
            labels_lst.append(labels.cpu())
            probs_lst.append(prob.cpu())

    images = torch.cat(image_lst, dim=0)
    labels = torch.cat(labels_lst, dim=0)
    probs = torch.cat(probs_lst, dim=0)

    return images, labels, probs


images, labels, probs = get_preds(model, test_loader, device)

pred_labels = torch.argmax(probs, 1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
cm = confusion_matrix(labels, pred_labels)
cm = ConfusionMatrixDisplay(cm, display_labels=range(10))
cm.plot(values_format='d', cmap='BuGn', ax=ax)
plt.show()
