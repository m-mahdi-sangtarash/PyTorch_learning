import torch


# Indexing and Slicing


my_tensor = torch.randint(1, 10, (4, 4))
print(my_tensor)
print("=========================\n")
print(f"First row: {my_tensor[0]}")
print("=========================\n")
print(f"First column: {my_tensor[:, 0]}")
print("=========================\n")


# Joining tensors

a = torch.randint(2, 16, (4, 4))
b = torch.randint(5, 23, (4, 4))

cat_tensor = torch.cat((a, b), dim=1)
print(f"cat tensor: {cat_tensor}")
print("=========================\n")

stack_tensor = torch.stack((a, b), dim=1)
print(f"stack tensor: {stack_tensor}")
print("=========================\n")

# Reshaping a tensor

base_tensor = torch.rand(16)
print(f"base tensor: {base_tensor}")
print("=========================\n")
view_tensor = base_tensor.view(4, 4)
print(f"reshape tensor with view: {view_tensor}")
print("=========================\n")
reshape_tensor = torch.reshape(view_tensor, (8, 2))
print(f"reshape tensor with torch.reshape: {reshape_tensor}")
print("=========================\n")
