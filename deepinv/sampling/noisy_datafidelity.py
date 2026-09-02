from __future__ import annotations
import torch
from deepinv.optim import DataFidelity, Distance
import deepinv as dinv
from deepinv.physics import Physics, DecomposablePhysics
from deepinv.models import Denoiser


class NoisyDataFidelity(DataFidelity):
    r"""
    Preconditioned data fidelity term for noisy data :math:`- \log p(y|x + \sigma(t) \omega)`
    with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This is a base class for the conditional classes for approximating :math:`\log p_t(y|x_t)` used in diffusion
    algorithms for inverse problems, in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method computing the score :math:`\nabla_{x_t} \log p_t(y|x_t)`.

    By default we have

    .. math::

         \nabla_{x_t} \log p(y|x + \sigma(t) \omega) = P(\forw{x_t'}-y),


    where :math:`P` is a preconditioner and :math:`x_t'` is an estimation of the image :math:`x`.
    By default, :math:`P` is defined as :math:`A^\top`, :math:`x_t' = x_t` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.

    :param deepinv.optim.Distance d: Distance metric to use for the data fidelity term. Default to :class:`deepinv.optim.L2Distance`.
    :param float weight: Weighting factor for the data fidelity term. Default to 1.
    """

    def __init__(self, d: Distance = None, weight=1.0, *args, **kwargs):
        super().__init__()
        if d is not None:
            self.d = Distance(d)
        else:
            self.d = dinv.optim.L2Distance()
        self.weight = weight

    def precond(
        self, u: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        The preconditioner :math:`P` for the data fidelity term. Default to :math:`A^{\top}`.

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.

        :return: (torch.Tensor) preconditioned tensor :math:`P(u)`.
        """
        return (
            physics.A_adjoint(u)
            if isinstance(physics, dinv.physics.LinearPhysics)
            else physics.A_dagger(u)
        )

    def diff(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the difference :math:`A(x) - y` between the forward operator applied to the current iterate and the input data.


        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics), physics=physics)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y) * self.weight


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    Diffusion posterior sampling data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in `Diffusion Posterior Sampling for General Noisy Inverse Problems <https://arxiv.org/abs/2209.14687>`_.

    .. math::
            \nabla_x \log p_t(y|x) = \nabla_x \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|

    where :math:`\sigma = \sigma(t)` is the noise level, :math:`m` is the number of measurements (size of :math:`y`),
    and :math:`\lambda` controls the strength of the approximation.

    .. seealso::
        This class can be used for building custom DPS-based diffusion models.
        A self-contained implementation of the original DPS algorithm can be find in :class:`deepinv.sampling.DPS`.

    :param deepinv.models.Denoiser denoiser: Denoiser network
    :param float weight: Weighting factor for the data fidelity term. Default to 1.0 .
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight: float = 1.0,
        clip: tuple = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        if clip is not None:
            if len(clip) != 2:  # pragma: no cover
                raise ValueError(f"clip must be None or length 2, but got {clip}")
            clip = sorted(clip)
        self.clip = clip
        self.weight = weight

    def precond(
        self, x: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :param float sigma: Standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term (and denoised output if `get_model_outputs` is `True`).
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            out = self.forward(
                x,
                y,
                physics,
                sigma,
                *args,
                get_model_outputs=get_model_outputs,
                **kwargs,
            )
            # In case we also want the denoised output
            if get_model_outputs:
                l2_loss = out[0]
            else:
                l2_loss = out

            grad_outputs = torch.ones_like(l2_loss)
        norm_grad = torch.autograd.grad(
            outputs=l2_loss, inputs=x, grad_outputs=grad_outputs
        )[0]
        if get_model_outputs:
            return norm_grad, out[1].detach()
        else:
            return norm_grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the loss term :math:`\frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the loss. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) loss term (and denoised output if `get_model_outputs` is `True`).
        """

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)
        x0_t = self.denoiser(x.to(torch.float32), sigma, *args, **kwargs)

        if self.clip is not None:
            x0_t = torch.clip(x0_t, self.clip[0], self.clip[1])  # optional

        out = (self.d(physics.A(x0_t), y) * y.numel() / y.size(0)).sqrt() * self.weight

        if get_model_outputs:
            return out, x0_t
        else:
            return out


class DDRMDataFidelity(NoisyDataFidelity):
    r"""
    Denoising Diffusion Restoration Model (DDRM) data-fidelity term.

    This is the closed-form approximation of the measurement term for a decomposable linear
    operator :math:`A = U\Sigma V^{\top}`, see :footcite:t:`kawar2022denoising` and the survey
    :footcite:t:`daras2024survey`.

    Plugging the Gaussian approximation
    :math:`p(x_0 \vert x_t) \approx \mathcal{N}\left(\denoiser{x_t}{\sigma_t}, \sigma_t^2 \mathrm{Id}\right)`
    into :math:`p_t(y \vert x_t) = \int p(y \vert x_0) p(x_0 \vert x_t) \mathrm{d}x_0` gives

    .. math::
        p_t(y \vert x_t) \approx \mathcal{N}\left(y; A\denoiser{x_t}{\sigma_t},
        \sigma_y^2 \mathrm{Id} + \sigma_t^2 A A^{\top}\right),

    whose negative log-likelihood has the gradient

    .. math::
        -\nabla_{x_t} \log p_t(y \vert x_t) \approx
        V \Sigma^{\top}
        \left(\sigma_y^2 \mathrm{Id} + \sigma_t^2 \Sigma \Sigma^{\top}\right)^{-1}
        \left(\Sigma V^{\top} \denoiser{x_t}{\sigma_t} - U^{\top} y\right),

    where :math:`\sigma_t = \sigma(t)` is the diffusion model noise level and :math:`\sigma_y` is
    the standard deviation of the measurement noise.

    .. note::

        Because the covariance :math:`\sigma_y^2 \mathrm{Id} + \sigma_t^2 \Sigma \Sigma^{\top}` is
        positive definite, the spectral weights are bounded by
        :math:`\min(\sigma_y^{-2}, (\sigma_t s)^{-2})` and no pseudo-inverse is needed.

    .. note::

        Equation (3.33) of :footcite:t:`daras2024survey` writes this term with
        :math:`\left| \sigma_y^2 \mathrm{Id} - \sigma_t^2 \Sigma \Sigma^{\top} \right|^{\dagger}`,
        i.e. a *difference* and a pseudo-inverse. That form belongs to the DDRM sampler itself,
        where :math:`\bar{x}_t` is constructed from :math:`\bar{y}` and the two therefore share the
        measurement noise. Here :math:`x_t` comes from the reverse SDE, where
        :math:`x_t = x_0 + \sigma_t \omega` with :math:`\omega` independent of the measurement
        noise, so the two variances add. The subtraction does appear in
        :class:`deepinv.sampling.DDRM`, as the variance budget :math:`\sigma_{t+1}^2 - v` left to
        fill when re-noising -- see :meth:`fused_denoised`.

    .. seealso::

        :meth:`deepinv.sampling.DDRMDataFidelity.fused_denoised` returns the equivalent
        :math:`x_0` estimate, i.e. the Gaussian fusion of the denoiser output with the
        measurement. This is what :class:`deepinv.sampling.DDRM` iterates on.

    :param deepinv.models.Denoiser denoiser: Denoiser network.
    :param float weight: Weighting factor for the data fidelity term. Default to 1.0.
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    :param float eps: Numerical floor added to the spectral denominator. Default to 1e-8.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight: float = 1.0,
        clip: tuple = None,
        eps: float = 1e-8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        self.weight = weight
        self.eps = eps
        if clip is not None:
            if len(clip) != 2:  # pragma: no cover
                raise ValueError(f"clip must be None or length 2, but got {clip}")
            clip = sorted(clip)
        self.clip = clip

    @staticmethod
    def _sigma_y(physics: Physics) -> torch.Tensor | float:
        r"""Standard deviation of the measurement noise, defaulting to 0.01 if unknown."""
        return getattr(physics.noise_model, "sigma", 0.01)

    @staticmethod
    def _singular_values(physics: DecomposablePhysics) -> torch.Tensor:
        r"""Singular values :math:`s` of the physics, as a non-negative real tensor."""
        mask = physics.mask
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        return mask.abs()

    def _denoise(self, x: torch.Tensor, sigma, *args, **kwargs) -> torch.Tensor:
        r"""Evaluate the denoiser in float32 and cast the result back to the input dtype."""
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)
        x0 = self.denoiser(x.to(torch.float32), sigma, *args, **kwargs).to(x.dtype)
        if self.clip is not None:
            x0 = torch.clip(x0, self.clip[0], self.clip[1])
        return x0

    def fused_denoised(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: DecomposablePhysics,
        sigma,
        etab: float = 1.0,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Gaussian fusion of the denoiser output with the measurement.

        In the spectral domain this combines the prior estimate
        :math:`\bar{x}_0 = V^{\top}\denoiser{x_t}{\sigma_t}` (error variance :math:`\sigma_t^2`)
        with the measurement :math:`\bar{y} = U^{\top} y` (error variance
        :math:`\sigma_y^2 / s^2`):

        .. math::
            \tilde{x}_0 = V \frac{\sigma_y^2 \bar{x}_0 + \sigma_t^2 s \bar{y}}
            {\sigma_y^2 + \sigma_t^2 s^2}, \qquad
            v = \frac{\sigma_y^2 \sigma_t^2}{\sigma_y^2 + \sigma_t^2 s^2}.

        It is related to :meth:`grad` by
        :math:`\tilde{x}_0 = \denoiser{x_t}{\sigma_t} - \sigma_t^2 \nabla`, and it interpolates
        between hard data consistency (:math:`\sigma_t s \gg \sigma_y`) and the unconstrained
        denoiser output (:math:`\sigma_t s \ll \sigma_y`).

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input corrupted observation.
        :param deepinv.physics.DecomposablePhysics physics: decomposable physics model.
        :param float sigma: Standard deviation of the noise of the model.
        :param float etab: Strength of the measurement term, interpolating between the plain
            denoiser output (`etab=0`) and the full fusion above (`etab=1`). Default to 1.0.

        :return: (tuple of :class:`torch.Tensor`) the fused estimate :math:`\tilde{x}_0` and the
            per-singular-value residual variance :math:`v` (in the spectral domain).
        """
        sigma_y = self._sigma_y(physics)
        s = self._singular_values(physics)
        x0_t = self._denoise(x, sigma, *args, **kwargs)

        denom = sigma_y**2 + sigma**2 * s**2 + self.eps
        residual = physics.U_adjoint(physics.A(x0_t) - y)
        fused = x0_t - etab * sigma**2 * physics.A_adjoint(physics.U(residual / denom))
        var = etab**2 * sigma_y**2 * sigma**2 / denom
        return fused, var

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: DecomposablePhysics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input corrupted observation.
        :param deepinv.physics.DecomposablePhysics physics: decomposable physics model.
        :param float sigma: Standard deviation of the noise of the model.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term (and denoised output if `get_model_outputs` is `True`).
        """
        sigma_y = self._sigma_y(physics)
        s = self._singular_values(physics)

        # 1. get x_0
        x0_t = self._denoise(x, sigma, *args, **kwargs)

        # 2. residual in the SVD basis
        residual = physics.U_adjoint(physics.A(x0_t) - y)

        # 3. inverse of the (positive definite) spectral covariance
        inv_denom = 1.0 / (sigma_y**2 + sigma**2 * s**2 + self.eps)

        # 4. the weighted grad
        grad = self.weight * physics.A_adjoint(physics.U(inv_denom * residual))

        if get_model_outputs:
            return grad, x0_t.detach()

        return grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "DDRMDataFidelity is defined directly through its closed-form approximation."
        )
