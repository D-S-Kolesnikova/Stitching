
import torch

def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps)
    x.sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)

tensor = torch.Tensor([1,2,3])
res = sift_to_rootsift(tensor)
print(res)