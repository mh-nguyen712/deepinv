import torch
import numpy as np


def bump_function(x, a=1.0, b=1.0):
    r"""
    Defines a function which is 1 on the interval [-a,a]
    and goes to 0 smoothly on [-a-b,-a]U[a,a+b] using a bump function
    For the discretization of indicator functions, we advise b=1, so that
    a=0, b=1 yields a bump.

    :param torch.Tensor x: tensor of arbitrary size
        input.
    :param Float a: radius (default is 1)
    :param Float b: interval on which the function goes to 0. (default is 1)

    :return: the bump function sampled at points x
    :rtype: torch.Tensor

    :Examples:

    >>> import deepinv as dinv
    >>> x = torch.linspace(-15, 15, 31)
    >>> X, Y = torch.meshgrid(x, x, indexing = 'ij')
    >>> R = torch.sqrt(X**2 + Y**2)
    >>> Z = bump_function(R, 3, 1)
    >>> Z = Z / torch.sum(Z)
    """
    v = torch.zeros_like(x)
    v[torch.abs(x) <= a] = 1
    I = (torch.abs(x) > a) * (torch.abs(x) < a + b)
    v[I] = torch.exp(-1.0 / (1.0 - ((torch.abs(x[I]) - a) / b) ** 2)) / np.exp(-1.0)
    return v
