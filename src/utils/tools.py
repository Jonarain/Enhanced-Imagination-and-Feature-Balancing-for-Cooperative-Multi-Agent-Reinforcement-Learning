# ref: https://github.com/NM512/dreamerv3-torch/blob/2c7a81a0e2f5f0c7659ba73b0ddbedf2a7e2ecf4/tools.py
import torch

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
