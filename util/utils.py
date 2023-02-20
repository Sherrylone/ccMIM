import torch

def idx_select(x, idx):
    D = x.size(2)
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, D))
