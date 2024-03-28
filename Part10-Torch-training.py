import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

labels_map = {0: 'T-shirt/top',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle boot'
              }


figure = plt.figure(figsize=(7, 7))
cols, rows = (3, 3)

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap='binary')
plt.show()


# DataLoader

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True)

train_feature, train_labels = next(iter(train_data_loader))

print(f"Feature batch size: {train_feature.size()}")
print(f"Labels shape: {train_labels.size()}")