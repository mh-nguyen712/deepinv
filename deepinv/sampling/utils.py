import torch
from torch import Tensor
import numpy as np


class Welford:
    r"""
    Welford's algorithm for calculating mean and variance

    https://doi.org/10.2307/1266577
    """

    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self):
        return self.M

    def var(self):
        return self.S / (self.k - 1)


def refl_projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.clamp(x, min=lower, max=upper)


def exp(input):
    if isinstance(input, Tensor):
        return torch.exp(input)
    elif isinstance(input, (float, int, np.ndarray)):
        return np.exp(input)
    else:
        raise TypeError(f"Invalid type {type(input)} for computing exponential")


def get_edm_parameters(discretization: str = "edm"):
    r"""
    :param str discretization: discretization type for solving the reverse-time SDE, one of 've', 'vp', 'edm

    :return dict containing: solver (str)
                             timesteps_fn (Callable): function to compute time step for discretization
                             sigma_fn (Callable): function to compute sigma
                             sigma_inv_fn (Callable): function to compute the inverse function of sigma_fn
                             sigma_deriv_fn (Callable): function to compute the derivative of sigma_fn
    """

    # Helper functions for VP
    if discretization == "vp":
        vp_beta_d = 19.9
        vp_beta_min = 0.1

        vp_sigma = (
            lambda t: (np.e ** (0.5 * vp_beta_d * (t**2) + vp_beta_min * t) - 1) ** 0.5
        )
        vp_sigma_deriv = (
            lambda t: 0.5
            * (vp_beta_min + vp_beta_d * t)
            * (vp_sigma(t) + 1 / vp_sigma(t))
        )
        vp_sigma_inv = (
            lambda sigma: (
                (vp_beta_min**2 + 2 * vp_beta_d * np.log(sigma**2 + 1)) ** 0.5
                - vp_beta_min
            )
            / vp_beta_d
        )

        def vp_timesteps(num_steps):
            return 1 + np.arange(num_steps) * (1e-3 - 1) / (num_steps - 1)
            # return np.apply_along_axis(vp_sigma_inv, 0, arr=1 + np.arange(num_steps - 1) * (1e-3 - 1) / (num_steps - 1))

        solver = "euler"
        timesteps_fn = vp_timesteps
        sigma_fn = vp_sigma
        sigma_inv_fn = vp_sigma_inv
        sigma_deriv_fn = vp_sigma_deriv
        beta_fn = lambda t: vp_sigma_deriv(t) / vp_sigma(t)
        s_fn = lambda t: 1 / (
            np.e ** (0.5 * vp_beta_d * (t**2) + vp_beta_min * t) ** 0.5
        )
        s_deriv_fn = lambda t: -0.5 * (vp_beta_d * t + vp_beta_min) / s_fn(t)
        sigma_max = vp_sigma(1.0)

    elif discretization == "ve":
        ve_sigma = lambda t: t**0.5
        ve_sigma_deriv = lambda t: 0.5 / t**0.5
        ve_sigma_inv = lambda sigma: sigma**2
        ve_sigma_max = 100
        ve_sigma_min = 0.02

        def ve_timesteps(num_steps):
            return ve_sigma_max**2 * (ve_sigma_min**2 / ve_sigma_max**2) ** (
                np.arange(num_steps) / (num_steps - 1)
            )

        solver = "euler"
        timesteps_fn = ve_timesteps
        sigma_fn = ve_sigma
        sigma_inv_fn = ve_sigma_inv
        sigma_deriv_fn = ve_sigma_deriv
        beta_fn = lambda t: 0.5 / t
        s_fn = lambda t: 1.0
        s_deriv_fn = lambda t: 0.0
        sigma_max = ve_sigma_max

    elif discretization == "edm":
        edm_rho = 7.0
        edm_one_over_rho = 1.0 / edm_rho
        edm_sigma_min = 0.002
        edm_sigma_max = 80

        edm_sigma = lambda t: t
        edm_sigma_inv = lambda sigma: sigma
        edm_sigma_deriv = lambda t: 1.0

        def edm_timesteps(num_steps):
            return (
                edm_sigma_max**edm_one_over_rho
                + np.arange(num_steps)
                * (edm_sigma_min**edm_one_over_rho - edm_sigma_max**edm_one_over_rho)
                / (num_steps - 1)
            ) ** edm_rho

        solver = "edm_heun"
        timesteps_fn = edm_timesteps
        sigma_fn = edm_sigma
        sigma_inv_fn = edm_sigma_inv
        sigma_deriv_fn = edm_sigma_deriv
        beta_fn = lambda t: 0.0
        s_fn = lambda t: 1.0
        s_deriv_fn = lambda t: 0.0
        sigma_max = edm_sigma_max

    params = {
        "solver": solver,
        "timesteps_fn": timesteps_fn,
        "sigma_fn": sigma_fn,
        "sigma_inv_fn": sigma_inv_fn,
        "sigma_deriv_fn": sigma_deriv_fn,
        "beta_fn": beta_fn,
        "s_fn": s_fn,
        "s_deriv_fn": s_deriv_fn,
        "sigma_max": sigma_max,
    }
    return params


def stable_division(a: Tensor, b: Tensor, epsilon: float = 1e-7):
    b = torch.where(
        b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign()
    )
    return a / b


def to_zeros_and_ones(a: Tensor):
    if a.ndim == 3:
        a -= a.min()
        a /= a.max()
    elif a.ndim == 4:
        a -= a.min(dim=(1, 2, 3), keepdim=True)
        a /= a.max(dim=(1, 2, 3), keepdim=True)
    return a
