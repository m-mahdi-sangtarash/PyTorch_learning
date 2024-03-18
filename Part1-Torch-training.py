import torch
import numpy as np

data = [[1, 2], [3, 4]]


# Create my first torch tensor

data_tensor = torch.tensor(data)

print(f"My Torch tensor : {data_tensor}")
print("===============================\n")

# From Numpy array to Torch tensor

data_array = np.array(data)
num_to_tensor = torch.from_numpy(data_array)

print(f"From Numpy array to Torch tensor: {num_to_tensor}")

