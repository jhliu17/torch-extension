import torch

from torch_extension.ext_lib import torch_extension_ops


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """add two tensors with custom op

    Args:
        a (torch.Tensor): tensor a
        b (torch.Tensor): tensor b

    Returns:
        torch.Tensor: result
    """
    out = torch.empty_like(a)
    torch_extension_ops.add(out, a, b)
    return out


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """sub two tensors with custom op

    Args:
        a (torch.Tensor): tensor a
        b (torch.Tensor): tensor b

    Returns:
        torch.Tensor: result
    """
    out = torch.empty_like(a)
    torch_extension_ops.sub(out, a, b)
    return out
