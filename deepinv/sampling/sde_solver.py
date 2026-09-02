from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Any
from numpy import ndarray
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepinv.sampling.diffusion_sde import BaseSDE


class SDEOutput(dict):
    r"""
    A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.

    Attributes:
    :attr torch.Tensor sample: the final samples of the sampling process, of shape ``(B, C, H, W)``.
    :attr torch.Tensor trajectory: the trajectory of the sampling process, of shape ``(num_steps, B, C, H, W)`` if ``full_trajectory`` is ``True``, otherwise of shape ``(B, C, H, W)``.
    :attr torch.Tensor timesteps: the time steps at which the samples were taken, of shape ``(num_steps,)``.
    :attr int nfe: the number of function evaluations performed during the integration.
    """

    def __init__(self, sample: Tensor, trajectory: Tensor, timesteps: Tensor, nfe: int):
        sol = {
            "sample": sample,
            "trajectory": trajectory,
            "timesteps": timesteps,
            "nfe": nfe,
        }
        super().__init__(sol)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class BaseSDESolver(nn.Module):
    r"""
    Base class for solving Stochastic Differential Equations (SDEs) from :class:`deepinv.sampling.BaseSDE` of the form:

    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient, and :math:`w_t` is a standard Brownian process.

    Currently only supported for fixed time steps for numerical integration.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: a random number generator for reproducibility, optional.
    :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.


    .. note::

        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator | None = None,
    ):
        super().__init__()
        if timesteps is None:
            if t_start is None or t_end is None or num_steps is None:
                raise ValueError(
                    "If timesteps is not provided, t_start, t_end, and num_steps must be specified."
                )
            timesteps = torch.linspace(t_start, t_end, num_steps)
        if isinstance(timesteps, ndarray):
            self.timesteps = torch.from_numpy(timesteps.copy())
        elif isinstance(timesteps, Tensor):
            self.timesteps = timesteps
        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()
            self.timesteps = self.timesteps.to(rng.device)

    def step(
        self,
        sde: BaseSDE,
        t0: float,
        t1: float,
        x0: Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, int]:
        r"""
        Perform a single step with step size from time `t0` to time `t1`, with current state `x0`.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param float or torch.Tensor t0: Time at the start of the step, of size (,).
        :param float or torch.Tensor t1: Time at the end of the step, of size (,).
        :param torch.Tensor x0: Current state of the system, of size (batch_size, d).

        :return torch.Tensor, int: Updated state of the system after the step and number of function evaluations (NFE) performed during the step.
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        sde: BaseSDE,
        x_init: Tensor,
        seed: int = None,
        *args,
        timesteps: Tensor | ndarray = None,
        get_trajectory: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the Stochastic Differential Equation (SDE) with given time steps.

        This function iteratively applies the SDE solver step for each time interval
        defined by the provided timesteps.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param torch.Tensor x_init: The initial state of the system.
        :param int seed: The seed for the random number generator, if `rng` is provided.
        :param torch.Tensor, numpy.ndarray, list timesteps: A sequence of time points at which to solve the SDE. If None, default timesteps will be used.
        :param bool get_trajectory: whether to return the full trajectory of the SDE or only the last sample, optional. Default to False.
        :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.
        :param \*args: Variable length argument list to be passed to the step function.
        :param \*\*kwargs: Arbitrary keyword arguments to be passed to the step function.

        :return: SDEOutput
        """
        self.rng_manual_seed(seed)
        x = x_init
        nfe = 0
        trajectory = [x_init.clone()] if get_trajectory else []

        if timesteps is None:
            timesteps = self.timesteps.to(sde.device, sde.dtype)
        else:
            if isinstance(timesteps, ndarray):
                timesteps = torch.from_numpy(timesteps.copy())
            timesteps = timesteps.to(sde.device, sde.dtype)

        for t_cur, t_next in tqdm(
            zip(timesteps[:-1], timesteps[1:], strict=True),
            total=len(timesteps) - 1,
            disable=not verbose,
        ):
            x, cur_nfe = self.step(sde, t_cur, t_next, x, *args, **kwargs)
            nfe += cur_nfe
            if get_trajectory:
                trajectory.append(x.clone())
        if get_trajectory:
            trajectory = torch.stack(trajectory, dim=0)
        else:
            trajectory = x
        output = SDEOutput(
            sample=x, trajectory=trajectory, timesteps=timesteps, nfe=nfe
        )

        return output

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

    def randn_like(self, input: torch.Tensor, seed: int = None) -> torch.Tensor:
        r"""
        Equivalent to :func:`torch.randn_like` but supports a pseudorandom number generator argument.

        :param torch.Tensor input: The input tensor whose size will be used.
        :param int seed: The seed for the random number generator, if `rng` is provided.

        :return: A tensor of the same size as input filled with random numbers from a normal distribution.
        :rtype: torch.Tensor

        This method uses the `rng` attribute of the class, which is a pseudo-random number generator
        for reproducibility. If a seed is provided, it will be used to set the state of `rng` before
        generating the random numbers.

        .. note::
           The `rng` attribute must be initialized for this method to work properly.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class EulerSolver(BaseSDESolver):
    r"""
    Euler-Maruyama solver for SDEs.

    This solver uses the Euler-Maruyama method to numerically integrate SDEs. It is a first-order method that
    approximates the solution using the following update rule:

    .. math::

        x_{t+dt} = x_t + f(x_t,t)dt + g(t) W_{dt}

    where :math:`W_t` is a Gaussian random variable with mean 0 and variance dt.

    :param torch.Tensor timesteps: The time steps at which to evaluate the solution.
    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: A random number generator for reproducibility.

    .. note::

        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, t_start, t_end, num_steps, rng=rng)

    def step(
        self, sde: BaseSDE, t0: float, t1: float, x0: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, int]:
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift, diffusion = sde.discretize(x0, t0, *args, **kwargs)
        return x0 + drift * dt + diffusion * dW, 1


class HeunSolver(BaseSDESolver):
    r"""
    Heun solver for SDEs.

    This solver uses the second-order Heun method to numerically integrate SDEs, defined as:

    .. math::
        \tilde{x}_{t+dt} &= x_t + f(x_t,t)dt + g(t) W_{dt} \\
        x_{t+dt} &= x_t + \frac{1}{2}[f(x_t,t) + f(\tilde{x}_{t+dt},t+dt)]dt + \frac{1}{2}[g(t) + g(t+dt)] W_{dt}

    where :math:`W_t` is a Gaussian random variable with mean 0 and variance dt.

    :param torch.Tensor timesteps: The time steps at which to evaluate the solution.
    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: A random number generator for reproducibility.
    
    .. note::
    
        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, t_start, t_end, num_steps, rng=rng)

    def step(
        self,
        sde: BaseSDE,
        t0: float,
        t1: float,
        x0: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, int]:
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift_0, diffusion_0 = sde.discretize(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_0 * dt + diffusion_0 * dW
        drift_1, diffusion_1 = sde.discretize(x_euler, t1, *args, **kwargs)

        return (
            x0
            + 0.5 * (drift_0 + drift_1) * dt
            + 0.5 * (diffusion_0 + diffusion_1) * dW,
            2,
        )


class DDIMSolver(BaseSDESolver):
    r"""
    DDIM / ancestral solver for diffusion SDEs :footcite:t:`song2020denoising`.

    Unlike :class:`deepinv.sampling.EulerSolver` and :class:`deepinv.sampling.HeunSolver`, which integrate the
    drift and diffusion in time, this solver steps directly in the noise level :math:`\sigma(t)`, in the rescaled
    variable :math:`\bar{x}_t = x_t / s(t)`. In that variable every diffusion of
    :class:`deepinv.sampling.EDMDiffusionSDE` is a pure variance-exploding process
    :math:`\bar{x}_t = x_0 + \sigma(t) \varepsilon`, which makes the update below schedule-agnostic:

    .. math::
        x_{t_1} = s(t_1) \left( \hat{x}_0 + \sqrt{\sigma(t_1)^2 - w^2} \, \hat{\varepsilon} + w z \right),
        \quad w = \eta \, \sigma(t_1) \frac{\sqrt{\sigma(t_0)^2 - \sigma(t_1)^2}}{\sigma(t_0)}

    where :math:`\hat{x}_0 = \mathbb{E}\left[x_0 \vert x_{t_0}\right]` is given by
    :meth:`denoised <deepinv.sampling.DiffusionSDE.denoised>`,
    :math:`\hat{\varepsilon} = (x_{t_0} / s(t_0) - \hat{x}_0) / \sigma(t_0)` is the implied noise, and
    :math:`z \sim \mathcal{N}(0, \mathrm{Id})`.

    The coefficient :math:`w` is the exact standard deviation of the forward posterior
    :math:`q(\bar{x}_{t_1} \vert \bar{x}_{t_0}, x_0)`, so :math:`\eta` interpolates between the deterministic
    and the fully stochastic sampler:

        - :math:`\eta = 0` gives DDIM. On a variance-preserving diffusion this is the sampler of
          :footcite:t:`song2020denoising`; on a variance-exploding one it is the Euler sampler of
          :footcite:t:`karras2022elucidating`.
        - :math:`\eta = 1` gives DDPM ancestral sampling :footcite:t:`ho2020denoising`, see
          :class:`deepinv.sampling.DDPMSolver`.

    .. note::

        This solver requires an SDE exposing the noise schedule and a denoised estimate, i.e. a
        :class:`deepinv.sampling.EDMDiffusionSDE` (or a :class:`deepinv.sampling.PosteriorSDE` built from one),
        and not a bare :class:`deepinv.sampling.BaseSDE`.

    .. note::

        The stochasticity is controlled by `eta` and **not** by the `alpha` of the SDE, which this solver ignores
        (in the continuous-time limit the two are related by :math:`\alpha = \eta^2`). A warning is raised when
        the two disagree.

    .. note::

        The `timesteps` must be decreasing. Contrary to the drift/diffusion solvers, `t_end = 0` is allowed and
        makes the last step an exact denoising step, since :math:`\sigma(0) = 0`.

    :param float eta: the amount of noise injected at each step, between `0` (deterministic, DDIM) and `1`
        (fully stochastic, DDPM). Default to `0.0`.
    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: A random number generator for reproducibility.

    .. note::

        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        eta: float = 0.0,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, t_start, t_end, num_steps, rng=rng)
        self.eta = eta

    def _check_sde(self, sde: BaseSDE, timesteps: Tensor = None) -> None:
        r"""
        Check that the `sde` and the `timesteps` are compatible with an ancestral update, and warn otherwise.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param torch.Tensor timesteps: the time steps used for this call, if they override the ones of the solver.
        """
        if not hasattr(sde, "denoised") or not hasattr(sde, "sigma_t"):
            raise ValueError(
                f"{type(self).__name__} requires an SDE exposing `sigma_t`, `scale_t` and `denoised`, such as "
                f"`deepinv.sampling.EDMDiffusionSDE` or `deepinv.sampling.PosteriorSDE`, but got "
                f"`{type(sde).__name__}`. Use `deepinv.sampling.EulerSolver` for a generic `BaseSDE`."
            )
        ts = self.timesteps if timesteps is None else timesteps
        if ts is not None and len(ts) > 1 and ts[0] < ts[-1]:
            warnings.warn(
                f"{type(self).__name__} integrates the reverse-time SDE and expects decreasing `timesteps`, "
                f"but got timesteps increasing from {float(ts[0])} to {float(ts[-1])}."
            )
        try:
            alpha = getattr(sde, "alpha", None)
            if alpha is not None and ts is not None and len(ts) > 0:
                value = float(alpha(ts[0]) if callable(alpha) else alpha)
                if abs(value - self.eta**2) > 1e-8:
                    warnings.warn(
                        f"The `alpha={value}` of the SDE is ignored by {type(self).__name__}: the amount of "
                        f"stochasticity is set by `eta={self.eta}` instead, which corresponds to "
                        f"`alpha={self.eta ** 2}` in the continuous-time limit."
                    )
        except Exception:  # pragma: no cover
            pass

    def sample(self, sde: BaseSDE, x_init: Tensor, *args, **kwargs) -> SDEOutput:
        self._check_sde(sde, kwargs.get("timesteps", None))
        return super().sample(sde, x_init, *args, **kwargs)

    def step(
        self, sde: BaseSDE, t0: float, t1: float, x0: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, int]:
        scale_0, scale_1 = sde.scale_t(t0), sde.scale_t(t1)
        sigma_0, sigma_1 = sde.sigma_t(t0), sde.sigma_t(t1)

        denoised = sde.denoised(x0, t0, *args, **kwargs)
        noise = (x0 / scale_0 - denoised) / sigma_0

        w = (
            self.eta
            * sigma_1
            * torch.clamp(sigma_0**2 - sigma_1**2, min=0).sqrt()
            / sigma_0
        )
        x1 = scale_1 * (denoised + torch.clamp(sigma_1**2 - w**2, min=0).sqrt() * noise)
        if self.eta > 0:
            x1 = x1 + scale_1 * w * self.randn_like(x0)
        return x1, 1


class DDPMSolver(DDIMSolver):
    r"""
    DDPM ancestral sampling :footcite:t:`ho2020denoising`.

    This is :class:`deepinv.sampling.DDIMSolver` with :math:`\eta = 1`, for which the injected noise matches the
    standard deviation of the forward posterior :math:`q(x_{t_1} \vert x_{t_0}, x_0)`. On a
    :class:`deepinv.sampling.VariancePreservingDiffusion` the update is exactly the ancestral step of
    :footcite:t:`ho2020denoising` with the :math:`\tilde{\beta}` variance.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: A random number generator for reproducibility.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator = None,
    ):
        super().__init__(
            eta=1.0,
            timesteps=timesteps,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            rng=rng,
        )
