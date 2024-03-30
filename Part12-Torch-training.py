import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import io

data_path = Path.cwd()
image_path = data_path / "Cats_dogs_images"

class MyDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        image_path = Path('.').joinpath(self.image_dir, self.annotations.iloc[index, 0])
        image = io.imread(image_path)
        label = self.annotations.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


train_data = MyDataset(csv_file='./Cats_dogs_images/annotations.csv',
                       image_dir='./Cats_dogs_images/cats_dogs')

train_data_loader = DataLoader(train_data,
                               batch_size=1,
                               shuffle=True)

train_features, train_label = next(iter(train_data_loader))

print(train_features)
print(train_label)