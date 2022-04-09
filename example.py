import torch
from build import torch_extension

a = torch.ones(10)
b = torch.ones(10)
c = torch.empty_like(a)

torch_extension.torch_add_two(c, a, b)
print(c)
