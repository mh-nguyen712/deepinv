# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Optional, Tuple
import numpy as np
from numpy import ndarray
import warnings
from utils import get_edm_parameters
from noisy_datafidelity import NoisyDataFidelity, DPSDataFidelity
from deepinv.optim.prior import ScorePrior
from deepinv.physics import Physics
from sde_solver import select_sde_solver
from scipy import integrate


class BaseSDE(nn.Module):
    r"""
    Base class for Stochastic Differential Equation (SDE):
    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion.

    It defines the common interface for drift and diffusion functions.

    :param callable drift: a time-dependent drift function f(x, t)
    :param callable diffusion: a time-dependent diffusion function g(t)
    """

    def __init__(
        self,
        drift: Callable,
        diffusion: Callable,
        rng: torch.Generator = None,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.dtype = dtype
        self.device = device
        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def sample(
        self,
        x_init: Optional[Tensor] = None,
        *args,
        timesteps: Union[Tensor, ndarray] = None,
        method: str = "euler",
        **kwargs,
    ):
        solver_fn = select_sde_solver(method)
        solver = solver_fn(sde=self, rng=self.rng, **kwargs)
        samples = solver.sample(x_init, timesteps=timesteps, *args, **kwargs)
        return samples

    def discretize(
        self, x: Tensor, t: Union[Tensor, float], *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return self.drift(x, t, *args, **kwargs), self.diffusion(t)

    def to(self, dtype=None, device=None):
        r"""
        Send the SDE to the desired device or dtype.
        This is useful when the drift of the diffusion term is parameterized (e.g., `deepinv.optim.ScorePrior`).
        """
        # Define the function to apply to each submodule
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device

        def apply_fn(module):
            module.to(device=device, dtype=dtype)

        # Use apply to run apply_fn on all submodules
        self.apply(apply_fn)

    def rng_manual_seed(self, seed: int = None):
        r"""
        Sets the seed for the random number generator.

        :param int seed: the seed to set for the random number generator. If not provided, the current state of the random number generator is used.
            Note: it will be ignored if the random number generator is not initialized.
        """
        if seed is not None:
            if self.rng is not None:
                self.rng = self.rng.manual_seed(seed)
            else:
                warnings.warn(
                    "Cannot set seed for random number generator because it is not initialized. The `seed` parameter is ignored."
                )

    def reset_rng(self):
        r"""
        Reset the random number generator to its initial state.
        """
        self.rng.set_state(self.initial_random_state)

    def randn_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.randn_like` but supports a pseudorandom number generator argument.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        drift: Callable = lambda x, t: -x,
        diffusion: Callable = lambda t: math.sqrt(2.0),
        prior: ScorePrior = None,
        rng: torch.Generator = None,
        use_backward_ode=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.prior = prior
        self.rng = rng
        self.use_backward_ode = use_backward_ode
        drift_forw = lambda x, t, *args, **kwargs: drift(x, t)
        diff_forw = lambda t: diffusion(t)
        self.forward_sde = BaseSDE(
            drift=drift_forw, diffusion=diff_forw, rng=rng, dtype=dtype, device=device
        )

        if self.use_backward_ode:
            drift_back = lambda x, t, *args, **kwargs: -drift(x, t) + 0.5 * (
                diffusion(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
        else:
            drift_back = lambda x, t, *args, **kwargs: -drift(x, t) + (
                diffusion(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
        diff_back = lambda t: diffusion(t)
        self.backward_sde = BaseSDE(
            drift=drift_back, diffusion=diff_back, rng=rng, dtype=dtype, device=device
        )

    def score(self, x: Tensor, sigma: Union[Tensor, float], rescale: bool = False):
        if rescale:
            x = (x + 1) * 0.5
            sigma_in = sigma * 0.5
        else:
            sigma_in = sigma
        score = -self.prior.grad(x, sigma_in)
        if rescale:
            score = score * 2 - 1
        return score

    @torch.no_grad()
    def forward(
        self, x_init: Optional[Tensor], timesteps: Tensor, method: str = "Euler"
    ):
        return self.backward_sde.sample(x_init, timesteps=timesteps, method=method)


class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        name: str = "ve",
        use_backward_ode=True,
        rng: torch.Generator = None,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_backward_ode = use_backward_ode
        params = get_edm_parameters(name)
        self.timesteps_fn = params["timesteps_fn"]
        self.sigma_max = params["sigma_max"]

        self.sde_T = params["sigma_inv_fn"](params["sigma_max"])

        sigma_fn = params["sigma_fn"]
        sigma_deriv_fn = params["sigma_deriv_fn"]
        beta_fn = params["beta_fn"]
        s_fn = params["s_fn"]
        s_deriv_fn = params["s_deriv_fn"]

        # Forward SDE
        drift_forw = lambda x, t, *args, **kwargs: (
            -sigma_deriv_fn(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
        ) * self.score(x, sigma_fn(t), *args, **kwargs)
        diff_forw = lambda t: sigma_fn(t) * (2 * beta_fn(t)) ** 0.5

        # Backward SDE
        if self.use_backward_ode:
            diff_back = lambda t: 0.0
            drift_back = lambda x, t, *args, **kwargs: -(
                (s_deriv_fn(t) / s_fn(t)) * x
                - (s_fn(t) ** 2)
                * sigma_deriv_fn(t)
                * sigma_fn(t)
                * self.score(x, sigma_fn(t), *args, **kwargs)
            )
        else:
            drift_back = lambda x, t, *args, **kwargs: (
                sigma_deriv_fn(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
            ) * self.score(x, sigma_fn(t), *args, **kwargs)
            diff_back = diff_forw

        self.forward_sde = BaseSDE(
            drift=drift_forw,
            diffusion=diff_forw,
            rng=rng,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        self.backward_sde = BaseSDE(
            drift=drift_back,
            diffusion=diff_back,
            rng=rng,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def forward(
        self,
        latents: Optional[Tensor] = None,
        shape: Tuple[int, ...] = None,
        method: str = "Euler",
        max_iter: int = 100,
    ):
        if latents is None:
            latents = (
                torch.randn(shape, device=device, generator=self.rng) * self.sigma_max
            )
        return self.backward_sde.sample(
            latents, timesteps=self.timesteps_fn(max_iter), method=method
        )

    @torch.no_grad()
    def likelihood(
        self,
        data: Tensor,
        hutchinson_type: str = "Rademacher",
        rtol=1e-5,
        atol=1e-5,
        method="RK45",
        eps=1e-5,
    ):
        def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
            return torch.from_numpy(x.reshape(shape))

        def prior_logp(z):
            shape = z.shape
            n_time_steps = np.prod(shape[1:])
            return -n_time_steps / 2.0 * np.log(
                2 * np.pi * self.sigma_max**2
            ) - torch.sum(z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

        assert self.use_backward_ode == True, "use_backward_ode should be True"
        shape = data.shape
        if hutchinson_type.lower() == "gaussian":
            epsilon = torch.randn_like(data)
        elif hutchinson_type.lower() == "rademacher":
            epsilon = torch.randint_like(data, low=0, high=2).type(data.dtype) * 2 - 1.0
        else:
            raise NotImplementedError(
                "Hutchinson type %s is not implemented." % hutchinson_type
            )

        def ode_func(t, x):
            sample = (
                from_flattened_numpy(x[: -shape[0]], shape)
                .to(data.device)
                .type(torch.float32)
            )
            vec_t = torch.ones(sample.shape[0], device=sample.device) * t
            drift = to_flattened_numpy(self.backward_sde.drift(sample, vec_t))
            logp_grad = to_flattened_numpy(self.div_fn(sample, vec_t, epsilon))
            return np.concatenate([drift, logp_grad], axis=0)

        init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
        solution = integrate.solve_ivp(
            ode_func, (eps, self.sde_T), init, rtol=rtol, atol=atol, method=method
        )
        zp = solution.y[:, -1]
        z = (
            from_flattened_numpy(zp[: -shape[0]], shape)
            .to(data.device)
            .type(torch.float32)
        )
        delta_logp = (
            from_flattened_numpy(zp[-shape[0] :], (shape[0],))
            .to(data.device)
            .type(torch.float32)
        )
        prior_logpx = prior_logp(z)
        print("Prior log p: ", prior_logpx)
        nll = -(prior_logpx + delta_logp)

        return nll

    def div_fn(self, x, t, noise):
        """
        Approximates the divergence of the drift function of the reverse SDE
        i.e., evaluate epsilon^T \nabla f(x, t) epsilon in eq (40)
        """
        return get_div_fn(lambda xx, tt: self.backward_sde.drift(xx, tt))(x, t, noise)


def get_div_fn(fn: Callable):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


class PosteriorEDMSDE(EDMSDE):
    def __init__(
        self, prior: ScorePrior, data_fidelity: NoisyDataFidelity, *args, **kwargs
    ):
        super().__init__(prior=prior, *args, **kwargs)
        self.data_fidelity = data_fidelity

    def score(
        self,
        x: Tensor,
        sigma: Union[Tensor, float],
        y: Tensor,
        physics: Physics,
        rescale: bool = False,
    ):
        if rescale:
            x = (x + 1) * 0.5
            sigma_in = sigma * 0.5
        else:
            sigma_in = sigma
        score_prior = -self.prior.grad(x, sigma_in)
        if rescale:
            score_prior = score_prior * 2 - 1
        return score_prior - self.data_fidelity.grad(x, y, physics, sigma)

    torch.no_grad()

    def forward(
        self,
        y: Tensor,
        physics: Physics,
        x_init: Optional[Tensor] = None,
        method: str = "Euler",
        max_iter: int = 100,
    ):
        if x_init is None:
            x_init = torch.randn_like(y) * self.sigma_max
        return self.backward_sde.sample(
            x_init, y, physics, timesteps=self.timesteps_fn(max_iter), method=method
        )


if __name__ == "__main__":
    from edm import load_model
    import numpy as np
    from deepinv.utils.demo import load_url_image, get_image_url
    import deepinv as dinv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device)
    x_noisy = x + torch.randn_like(x) * 0.3
    dinv.utils.plot(
        [x, x_noisy, denoiser(x_noisy, 0.3)], titles=["sample", "y", "denoised"]
    )

    # denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
    # denoiser = dinv.models.DRUNet(device=device)
    prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)

    # EDM generation
    sde = EDMSDE(name="ve", prior=prior, use_backward_ode=True)
    x = sde(shape=(1, 3, 64, 64), max_iter=100, method="heun")
    print("NLL: ", sde.likelihood(x))
    # Posterior EDM generation
    physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=0.5, device=device)
    noisy_data_fidelity = DPSDataFidelity(denoiser=denoiser)
    y = physics(x)
    posterior_sde = PosteriorEDMSDE(
        prior=prior,
        data_fidelity=noisy_data_fidelity,
        name="ve",
        use_backward_ode=True,
    )
    posterior_sample = posterior_sde(y, physics, max_iter=100, method="heun")

    # Plotting the samples
    # dinv.utils.plot([x], titles = ['sample'])
    dinv.utils.plot(
        [x, y, posterior_sample], titles=["sample", "y", "posterior_sample"]
    )
