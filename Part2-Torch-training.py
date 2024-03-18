import torch
import numpy

# Create a zeros tensor

zeros_tensor = torch.zeros(3, 4)
print(f"Zeros tensor: {zeros_tensor}")
print("============================\n")


# Create a ones tensor

ones_tensor = torch.ones(4, 4)
print(f"Ones tensor: {ones_tensor}")
print("============================\n")


# Create a eye tensor

eye_tensor = torch.eye(4, 4)
print(f"Eye tensor: {eye_tensor}")
print("============================\n")


# Create a rand tensor

rand_tensor = torch.rand(5, 5)
print(f"Rand tensor: {rand_tensor}")
print("============================\n")


# Create a randn tensor

randn_tensor = torch.randn(3, 3)
print(f"Randn tensor: {randn_tensor}")
print("============================\n")


# Create a randint tensor

randint_tensor = torch.randint(0, 10, (3, 3))
print(f"Randint tensor: {randint_tensor}")
print("============================\n")


# Create a rand_like tensor

base = [[5, 2, 6],
        [4, 0, 3],
        [8, 1, 9]]

base_tensor = torch.tensor(base, dtype=torch.float32)

rand_like_tensor = torch.rand_like(base_tensor)
print(f"Rand_like tensor: {rand_like_tensor}")
print("============================\n")


# Create a randn_like tensor

randn_like_tensor = torch.randn_like(base_tensor)
print(f"Randn_like tensor: {randn_like_tensor}")
print("============================\n")


# Create a randint_like tensor

randint_like_tensor = torch.randint_like(base_tensor, 10, 20)
print(f"Randint_like tensor: {randint_like_tensor}")
print("============================\n")


# Create a Zeros_like tensor

zeros_like_tensor = torch.zeros_like(base_tensor)
print(f"Zeros_like tensor: {zeros_like_tensor}")
print("============================\n")


# Create a empty tensor

empty_tensor = torch.empty(3, 3)
print(f"Empty tensor: {empty_tensor}")
print("============================\n")


# Create a diag tensor

line_tensor = torch.randn(3)

diag_tensor = torch.diag(line_tensor)
print(f"Line tensor: {line_tensor}\n")
print(f"Diag tensor: {diag_tensor}")
print("============================\n")