import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)


class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.squential = nn.Sequential(
            nn.Linear(in_features=784, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.squential(x)
        return x


model = NNModel()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
batch_size = 64
n_epochs = 3

optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):

        # generate predictions
        pred = model(X)

        # calculate loss
        loss = loss_fn(pred, y)

        # compute gradients
        loss.backward()

        # update parameters w, b
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"Loss: {loss:>4f} ----  [{current:>5d}] out of {len(dataloader.dataset):>5d}")


def test(dataloader, model, loss_fn):
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            loss = loss_fn(pred, y)

            test_loss += loss.item()

            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        test_loss /= len(dataloader.dataset)
        correct /= len(dataloader.dataset)

        print(f"\n Results: Accuracy: {(100 * correct):>0.1f}%, Average Loss: {test_loss:>8f} \n ")


for epoch in range(n_epochs):
    print(f"[Epoch {epoch + 1}]\n")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Completed!")

# part 17 - PyTorch Course - save model weights

torch.save(model.state_dict(), 'model_weights.pth')

model_1 = NNModel()
model_1.load_state_dict(torch.load('model_weights.pth'))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

model_1.eval()

x, y = test_data[2][0], test_data[2][1]

with torch.no_grad():
    pred = model_1(x)
    predicted, actual = classes[pred[0].argmax(dim=0)], classes[y]
    print(f"Prediction: {predicted}, Actual: {actual}")
