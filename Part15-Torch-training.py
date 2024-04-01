import torch


# Autograd

a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([4.], requires_grad=True)

y = a**3 + b**2
print(y)
print("========================\n")

y.backward()

print(a.grad, b.grad)
