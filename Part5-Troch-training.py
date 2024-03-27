import torch

tensor_a = torch.randint(2, 12, (3, 4))
tensor_b = torch.randint(5, 19, (3, 4))
print(f"Tensor A: {tensor_a}\n")
print(f"Tensor B: {tensor_b}\n")
print("=========================================\n")


# Addition

print(f"Tensors addition: ", tensor_a + tensor_b)
print("=========================================\n")


# Subtraction

print(f"Tensors subtraction: ", tensor_a - tensor_b)
print("=========================================\n")


# Multiplication

print(f"Tensors multiplication: ", tensor_a * tensor_b)
print("=========================================\n")


# Division

print(f"Tensors division: ", tensor_a / tensor_b)
print("=========================================\n")


# Exponent

print(f"Tensors exponent: ", tensor_a ** tensor_b)
print("=========================================\n")



