import numpy as np
import torch

# Numpy array to PyTorch tensor

np_array = np.ones((2, 3))
print(f"Numpy array: {np_array}")
print("====================================\n")

pytorch_tensor = torch.from_numpy(np_array)
print(f"Torch tensor: {pytorch_tensor}")
print("====================================\n")
