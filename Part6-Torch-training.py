import torch

my_tensor = torch.rand(2, 3)

# Clone


print(f"Base tensor: {my_tensor}")
print("==================================")

b = my_tensor
b[0][2] = 45

print(torch.eq(my_tensor, b))
print("==================================\n")

c = my_tensor.clone()
c[1][1] = 89

print(torch.eq(my_tensor, c))
print("==================================\n")

new_tensor = torch.rand(4, 3, requires_grad=True)
print(f"New tensor(with grad_fn): {new_tensor}\n")
n_tensor_clone = new_tensor.clone()
print(f"Clone Tensor: {n_tensor_clone}")
print("==================================\n")

tensor_no_grad = new_tensor.detach().clone()
print(f"Tensor no grad: {tensor_no_grad}")
print("==================================\n")


# Unsqueeze

sample_tensor = torch.rand(3, 256, 256)
us_tensor = sample_tensor.unsqueeze(0)

print(f"Sample_tensor: {sample_tensor.shape}")
print(f"Unsqueeze tensor: {us_tensor.shape}")
print("==================================\n")


# Squeeze

tensor = torch.rand(1, 15)
print(f"Tensor: {tensor}\n")

sq_tensor = tensor.squeeze(0)
print(sq_tensor.shape)
print(f"Squeeze tensor: {sq_tensor}")
print("==================================\n")



