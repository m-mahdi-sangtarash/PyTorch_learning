from torchvision.datasets import FashionMNIST

# Work with datasets

train_data = FashionMNIST(
    root='./train',
    train=True,
    download=True
)

# Shape

print(f"Data shape: {train_data.data.shape}")
print("====================================\n")


# Targets

print(f"Date targets: {train_data.targets}")
print("====================================\n")


# Classes

print(f"Data classes: {train_data.classes}")
print("====================================\n")


# Classes and index

print(f"Classes and index: {train_data.class_to_idx}")
print("====================================\n")


# Type

print(f"Type: {type(train_data[0])}")
print("====================================\n")


# Data Len

print(f"Len: {len(train_data[0])}")
print("====================================\n")


# Labels

data, label = train_data[0]
print(data, label)
print("====================================\n")


# Data types
print(type(data))
print("====================================\n")


