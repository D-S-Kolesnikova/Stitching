import torch

m = torch.Tensor(1, 256, 2048)
print(m.shape)

m = m.transpose(1, 2)
print(m.shape)