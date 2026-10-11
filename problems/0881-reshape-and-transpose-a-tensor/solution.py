import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    # TODO: flatten x to 1-D, then rearrange into new_shape
    x = torch.flatten(x)
    x = torch.reshape(x, new_shape)
    return x

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    # TODO: swap the last two dimensions of x
    x = torch.transpose(x, -1, -2)
    return x
