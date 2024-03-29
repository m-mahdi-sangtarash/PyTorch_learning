import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt

train_trans = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip()])


cifar_train = CIFAR10(
    root='train',
    train=True,
    download=True,
    transform=train_trans
)


def visualize(train_data, labels_map):
    figure = plt.figure(figsize=(7, 7))
    cols, rows = (3, 3)

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img, cmap='binary')
    plt.show()


labels = {0: 'airplane',
              1: 'automobile',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck'}

visualize(cifar_train, labels)