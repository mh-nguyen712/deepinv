import gc
import pytest
import torch.nn
import numpy as np

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.sampling import (
    ULA,
    SKRock,
    DiffPIR,
    DPS,
    sampling_builder,
    DDRM,
    VarianceExplodingDiffusion,
    VariancePreservingDiffusion,
    EDMDiffusionSDE,
    FlowMatching,
    PosteriorDiffusion,
    DPSDataFidelity,
    EulerSolver,
    HeunSolver,
    DDIMSolver,
    DDPMSolver,
)
from deepinv.models import NCSNpp, ADMUNet, DRUNet

SAMPLING_ALGOS = ["DDRM", "ULA", "SKRock"]


def choose_algo(algo, likelihood, thresh_conv, sigma, sigma_prior):
    if algo == "ULA":
        out = ULA(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            thinning=1,
            step_size=0.01 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
            verbose=True,
        )
    elif algo == "SKRock":
        out = SKRock(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            inner_iter=5,
            step_size=1 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
            verbose=True,
        )
    elif algo == "DDRM":
        diff = dinv.sampling.DDRM(
            denoiser=GaussianDenoiser(sigma_prior),
            eta=1,
            sigmas=np.linspace(1, 0, 100),
        )
        out = dinv.sampling.DiffusionSampler(diff, clip=(-100, 100), max_iter=500)
    else:
        raise Exception("The sampling algorithm doesnt exist")

    return out


class GaussianScore(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def grad(self, x, sigma):
        return x / self.sigma_prior2


class GaussianDenoiser(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def forward(self, x, sigma):
        return x / (1 + sigma**2 / self.sigma_prior2)


@pytest.mark.parametrize("algo", SAMPLING_ALGOS)
def test_sampling_algo(algo, imsize, device):
    test_sample = torch.ones((1, *imsize))

    sigma = 1
    sigma_prior = 1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    convergence_crit = 0.1  # for fast tests
    likelihood = L2(sigma=sigma)
    f = choose_algo(
        algo,
        likelihood,
        thresh_conv=convergence_crit,
        sigma=sigma,
        sigma_prior=sigma_prior,
    )

    xmean, xvar = f(y, physics, seed=0)

    tol = 5  # can be lowered?
    sigma2 = sigma**2
    sigma_prior2 = sigma_prior**2

    # the posterior of a gaussian likelihood with a gaussian prior is gaussian
    post_var = (sigma2 * sigma_prior2) / (sigma2 + sigma_prior2)
    post_mean = y / (1 + sigma2 / sigma_prior2)

    mean_ok = (
        torch.sum((xmean - post_mean).abs() / post_mean < tol)
        > np.prod(xmean.shape) / 2
    )

    var_ok = (
        torch.sum((xvar - post_var).abs() / post_var < tol) > np.prod(xvar.shape) / 2
    )

    assert f.mean_has_converged and f.var_has_converged and mean_ok and var_ok


@pytest.mark.parametrize("name_algo", ["DiffPIR", "DPS"])
def test_algo(name_algo, device):
    test_sample = torch.ones((1, 3, 64, 64), device=device)

    sigma = 1
    # choose physics that changes the image size
    physics = dinv.physics.Blur(
        dinv.physics.functional.gaussian_blur(sigma=(3, 3)), device=device
    )
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    likelihood = L2(sigma=sigma)

    if name_algo == "DiffPIR":
        f = DiffPIR(
            dinv.models.DiffUNet().to(device),
            likelihood,
            max_iter=5,
            verbose=False,
            device=device,
        )
    elif name_algo == "DPS":
        f = DPS(
            dinv.models.DiffUNet().to(device),
            num_steps=5,
            verbose=False,
            device=device,
        )
    else:
        raise Exception("The sampling algorithm doesn't exist")

    x = f(y, physics)

    assert x.shape == test_sample.shape


@pytest.mark.parametrize("name_algo", ["DiffPIR", "DPS", "DDRM"])
def test_algo_inpaint(name_algo, device):
    x = torch.ones((1, 3, 32, 32)).to(device)
    x[:, 0, ...] = 0  # create a colored image

    torch.manual_seed(10)

    mask = torch.ones_like(x)
    mask[:, :, 10:20, 10:20] = 0

    physics = dinv.physics.Inpainting(mask=mask, img_size=x.shape[1:], device=device)

    y = physics(x)

    model = dinv.models.DRUNet(device=device)
    likelihood = L2()

    if name_algo == "DiffPIR":
        algorithm = DiffPIR(
            model, likelihood, max_iter=20, verbose=False, device=device, sigma=0.01
        )
    elif name_algo == "DPS":
        algorithm = DPS(
            model, num_steps=50, weight=2.0, alpha=0.5, verbose=False, device=device
        )
    elif name_algo == "DDRM":
        algorithm = DDRM(model)

    with torch.no_grad():
        out = algorithm(y, physics)

    assert out.shape == x.shape

    mean_crop = out[:, :, 10:20, 10:20].flatten().mean()

    mask = mask.bool()
    masked_out = out[mask]
    mean_outside_crop = masked_out.mean()

    masked_target = x[mask]
    mean_target_masked = masked_target.mean()
    mean_target_inmask = 2 / 3.0

    assert (mean_target_inmask - mean_crop).abs() < 0.2
    assert (mean_target_masked - mean_outside_crop).abs() < 0.02


# tests for sample_builder
BUILD_ALGOS = ["ULA", "SKRock"]


def choose_algo_build(algo, likelihood, thresh_conv, sigma, sigma_prior):
    prior = GaussianScore(sigma_prior)

    if algo == "ULA":
        params = {
            "step_size": 0.01 / (1 / sigma**2 + 1 / sigma_prior**2),
            "alpha": 1.0,
            "sigma": 1.0,
        }
    elif algo == "SKRock":
        params = {
            "step_size": 1 / (1 / sigma**2 + 1 / sigma_prior**2),
            "alpha": 1.0,
            "inner_iter": 5,
            "eta": 0.05,
            "sigma": 1.0,
        }
    else:
        raise Exception("The sampling algorithm doesn't exist")

    out = sampling_builder(
        iterator=algo,
        data_fidelity=likelihood,
        prior=prior,
        thresh_conv=thresh_conv,
        params_algo=params,
        max_iter=500,
        burnin_ratio=0.2,
        thinning=1,
        verbose=True,
        clip=(-100, 100),
    )

    return out


@pytest.mark.parametrize("algo", BUILD_ALGOS)
def test_build_algo(algo, imsize, device):
    # NOTE: redundancy here with the above test_sample_algo
    test_sample = torch.ones((1, *imsize))

    sigma = 1
    sigma_prior = 1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    convergence_crit = 0.1  # for fast tests
    likelihood = L2(sigma=sigma)
    f = choose_algo_build(
        algo,
        likelihood,
        thresh_conv=convergence_crit,
        sigma=sigma,
        sigma_prior=sigma_prior,
    )

    xmean, xvar = f.sample(y, physics, seed=0)

    tol = 5  # can be lowered?
    sigma2 = sigma**2
    sigma_prior2 = sigma_prior**2

    # the posterior of a gaussian likelihood with a gaussian prior is gaussian
    post_var = (sigma2 * sigma_prior2) / (sigma2 + sigma_prior2)
    post_mean = y / (1 + sigma2 / sigma_prior2)

    mean_ok = (
        torch.sum((xmean - post_mean).abs() / post_mean < tol)
        > np.prod(xmean.shape) / 2
    )

    var_ok = (
        torch.sum((xvar - post_var).abs() / post_var < tol) > np.prod(xvar.shape) / 2
    )

    assert f.mean_has_converged and f.var_has_converged and mean_ok and var_ok


@torch.no_grad()
@pytest.mark.parametrize(
    "sde_class",
    [
        FlowMatching,
        VarianceExplodingDiffusion,
        VariancePreservingDiffusion,
        EDMDiffusionSDE,
    ],
)
@pytest.mark.parametrize("solver_class", [EulerSolver, HeunSolver])
@pytest.mark.parametrize("denoiser_class", [NCSNpp, ADMUNet, DRUNet])
def test_sde(device, load_example_image, sde_class, solver_class, denoiser_class):
    try:
        if denoiser_class == ADMUNet:
            kwargs = dict(class_labels=torch.eye(1000, device=device)[0:1])
        else:
            kwargs = dict()
        denoiser = denoiser_class(pretrained="download").to(device)
        x = load_example_image(
            "celeba_example.jpg",
            img_size=64,
            resize_mode="resize",
        ).to(device)

        # Set up the SDEs
        num_steps = 2
        rng = torch.Generator(device)
        # Set up solvers
        timesteps = torch.linspace(0.99, 0.001, num_steps, device=device)
        solver = solver_class(
            timesteps=timesteps,
            rng=rng,
        )

        if sde_class == EDMDiffusionSDE:
            sigma_t = lambda t: 100 * t**2
            scale_t = lambda t: 1 / (1 + sigma_t(t) ** 2) ** 0.5
            sde = sde_class(
                sigma_t=sigma_t,
                scale_t=scale_t,
                denoiser=denoiser,
                solver=solver,
                device=device,
            )
        else:
            sde = sde_class(
                denoiser=denoiser,
                solver=solver,
                device=device,
            )
        # Test generation
        sample, _trajectory = sde.sample(
            (2, 3, 64, 64),
            seed=10,
            get_trajectory=True,
            **kwargs,
        )
        assert sample.shape == (2, 3, 64, 64)

        # Test posterior sampling
        posterior = PosteriorDiffusion(
            data_fidelity=DPSDataFidelity(denoiser=denoiser),
            sde=sde,
            denoiser=denoiser,
            solver=solver,
            dtype=torch.float64,
            device=device,
        )
        physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=0.5, device=device)
        y = physics(x)

        x_hat = posterior(
            y,
            physics,
            x_init=(2, 3, 64, 64),
            seed=111,
            **kwargs,
        )
        # Test output shape
        assert x_hat.shape == (2, 3, 64, 64)
    finally:
        # pytest seems to not clean objects properly, which can cause OOM errors.
        del denoiser, sde, posterior
        gc.collect()
        torch.cuda.empty_cache()


VE_VP_SDE_CLASSES = [VarianceExplodingDiffusion, VariancePreservingDiffusion]


@pytest.mark.parametrize("sde_class", VE_VP_SDE_CLASSES)
@pytest.mark.parametrize("t", [0.2, 0.5, 0.8, 0.95])
def test_sigma_scale_prime_matches_finite_difference(sde_class, t):
    """`sigma_prime_t`, `scale_prime_t` must match finite difference"""
    sde = sde_class(dtype=torch.float64)
    h = 1e-6
    # For sigma
    fd = (float(sde.sigma_t(t + h)) - float(sde.sigma_t(t - h))) / (2 * h)
    assert float(sde.sigma_prime_t(t)) == pytest.approx(fd, rel=1e-4)
    # For scale
    fd = (float(sde.scale_t(t + h)) - float(sde.scale_t(t - h))) / (2 * h)
    assert float(sde.scale_prime_t(t)) == pytest.approx(fd, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("sde_class", VE_VP_SDE_CLASSES)
def test_T_is_settable(sde_class):
    """The documented end time `T` must be reachable from the constructor."""
    assert sde_class(T=0.9).T == 0.9


@pytest.mark.parametrize("sde_class", VE_VP_SDE_CLASSES)
def test_sample_init_uses_given_time_step(sde_class, rng, device):
    """`sample_init` draws at the requested time, defaulting to `T`."""
    sde = sde_class(device=device)
    shape = (1, 3, 64, 64)
    for t in (sde.T, 0.5):
        init = sde.sample_init(shape, rng=rng, t=t)
        expected = float(sde.sigma_t(t) * sde.scale_t(t))
        assert float(init.std()) == pytest.approx(expected, rel=5e-2)

    # Default sample must be at time T
    rng.manual_seed(0)
    default = sde.sample_init(shape, rng=rng)
    rng.manual_seed(0)
    assert torch.allclose(default, sde.sample_init(shape, rng=rng, t=sde.T))


@torch.no_grad()
@pytest.mark.parametrize(
    "sde_class",
    [
        FlowMatching,
        VarianceExplodingDiffusion,
        VariancePreservingDiffusion,
        EDMDiffusionSDE,
    ],
)
def test_diffusion_reproducibility(load_example_image, device, rng, sde_class):
    timesteps = torch.linspace(0.99, 0.001, 2, device=device)
    denoiser = NCSNpp(pretrained="download").to(device)
    solver = EulerSolver(timesteps=timesteps, rng=rng)

    sigma_t = lambda t: 100 * t**2
    scale_t = lambda t: 1 / (1 + sigma_t(t) ** 2) ** 0.5
    kwargs = (
        {"sigma_t": sigma_t, "scale_t": scale_t} if sde_class == EDMDiffusionSDE else {}
    )
    sde = sde_class(
        denoiser=denoiser,
        solver=solver,
        device=device,
        **kwargs,
    )
    x = load_example_image(
        "celeba_example.jpg",
        img_size=64,
        resize_mode="resize",
    ).to(device)
    physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=0.5, device=device)
    y = physics(x)

    # Test posterior sampling
    posterior = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=denoiser),
        sde=sde,
        denoiser=denoiser,
        solver=solver,
        dtype=torch.float64,
        device=device,
    )

    x_hat_1 = posterior(
        y,
        physics,
        x_init=(2, 3, 64, 64),
        seed=111,
    )
    # Test output shape
    assert x_hat_1.shape == (2, 3, 64, 64)
    # Test reproducibility
    x_hat_2 = posterior(
        y,
        physics,
        x_init=(2, 3, 64, 64),
        seed=111,
    )
    torch.testing.assert_close(x_hat_1, x_hat_2, rtol=1e-2, atol=1e-2)


@torch.no_grad()
def test_noisy_data_fidelity(device):
    from deepinv.sampling import DPSDataFidelity, NoisyDataFidelity
    import itertools

    all_data_fid_classes = [NoisyDataFidelity, DPSDataFidelity]
    all_clip = [None, (-100, 100)]
    denoiser = dinv.models.DRUNet(pretrained="download").to(device)
    x = torch.rand(2, 3, 64, 64, device=device)
    physics = dinv.physics.Blur(
        filter=dinv.physics.functional.gaussian_blur(sigma=(3, 3)), device=device
    )
    y = physics(x)
    sigma = 0.1
    for data_fid_class, clip in itertools.product(all_data_fid_classes, all_clip):
        data_fid = data_fid_class(
            denoiser=denoiser,
            clip=clip,
        )
        # Test forward pass
        assert data_fid(x, y, physics, sigma).shape == torch.Size([x.size(0)])
        # Test grad pass
        assert data_fid.grad(x, y, physics, sigma).shape == x.shape
        # Test preconditioning
        try:
            output = data_fid.precond(y, physics, sigma)
            assert output.shape == x.shape
        except NotImplementedError:
            pass


# --------------------------------------------------------------------------------------------
# Ancestral (DDIM / DDPM) solvers
# --------------------------------------------------------------------------------------------

ANCESTRAL_SDE_CLASSES = [
    VarianceExplodingDiffusion,
    VariancePreservingDiffusion,
    FlowMatching,
]


class _ToyDenoiser(dinv.models.Denoiser):
    """A cheap nonlinear denoiser: the identities below must hold for any denoiser."""

    def forward(self, x, sigma, **kwargs):
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        return torch.tanh(3 * x) / (1 + sigma**2) + 0.2 * torch.sin(x)


def _toy_sde(sde_class, device, **kwargs):
    timesteps = torch.linspace(0.9, 1e-3, 3, dtype=torch.float64, device=device)
    return sde_class(
        denoiser=_ToyDenoiser(),
        solver=EulerSolver(timesteps=timesteps),
        minus_one_one=False,
        dtype=torch.float64,
        device=device,
        **kwargs,
    )


def _vp_alpha_bar(t, beta_min=0.1, beta_max=20.0):
    """alpha_bar(t) = exp(-B(t)) of `VariancePreservingDiffusion`."""
    return torch.exp(-(beta_min * t + 0.5 * t**2 * (beta_max - beta_min)))


@torch.no_grad()
@pytest.mark.parametrize("sde_class", ANCESTRAL_SDE_CLASSES)
def test_denoised_matches_tweedie(sde_class, device):
    """`denoised` and `score` must be two views of the same quantity (Tweedie's formula)."""
    sde = _toy_sde(sde_class, device)
    x = torch.randn(4, 1, 8, 8, dtype=torch.float64, device=device) * 0.7
    for t in (0.9, 0.5, 0.05):
        scale, sigma = sde.scale_t(t), sde.sigma_t(t)
        from_score = (x + (scale * sigma) ** 2 * sde.score(x, t)) / scale
        assert torch.allclose(sde.denoised(x, t), from_score, atol=1e-9)


@torch.no_grad()
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_ddim_solver_matches_ddim_reference(eta, device, rng):
    """A `DDIMSolver` step must reproduce the DDIM update of Song et al. exactly, for any eta."""
    sde = _toy_sde(VariancePreservingDiffusion, device)
    solver = DDIMSolver(eta=eta, timesteps=torch.linspace(0.9, 1e-3, 3), rng=rng)
    x = torch.randn(4, 1, 8, 8, dtype=torch.float64, device=device) * 0.7

    for t0, t1 in [(0.9, 0.85), (0.5, 0.4), (0.2, 0.05), (0.05, 1e-3)]:
        t0 = torch.tensor(t0, dtype=torch.float64, device=device)
        t1 = torch.tensor(t1, dtype=torch.float64, device=device)
        a_0, a_1 = _vp_alpha_bar(t0), _vp_alpha_bar(t1)
        x0_hat = sde.denoised(x, t0)
        eps = (x / sde.scale_t(t0) - x0_hat) / sde.sigma_t(t0)
        sigma_ddim = eta * (((1 - a_1) / (1 - a_0)) * (1 - a_0 / a_1)).sqrt()
        expected = (
            a_1.sqrt() * x0_hat
            + torch.clamp(1 - a_1 - sigma_ddim**2, min=0).sqrt() * eps
        )

        if eta > 0:
            rng.manual_seed(0)
            expected = expected + sigma_ddim * solver.randn_like(x, seed=0)
            rng.manual_seed(0)
        out, nfe = solver.step(sde, t0, t1, x)
        assert nfe == 1
        assert torch.allclose(out, expected, atol=1e-10)


@torch.no_grad()
def test_ddpm_solver_matches_ancestral_reference(device, rng):
    """A `DDPMSolver` step must reproduce Ho et al.'s ancestral update with the beta-tilde variance."""
    sde = _toy_sde(VariancePreservingDiffusion, device)
    solver = DDPMSolver(timesteps=torch.linspace(0.9, 1e-3, 3), rng=rng)
    assert solver.eta == 1.0
    x = torch.randn(4, 1, 8, 8, dtype=torch.float64, device=device) * 0.7

    for t0, t1 in [(0.9, 0.85), (0.5, 0.4), (0.2, 0.05)]:
        t0 = torch.tensor(t0, dtype=torch.float64, device=device)
        t1 = torch.tensor(t1, dtype=torch.float64, device=device)
        a_0, a_1 = _vp_alpha_bar(t0), _vp_alpha_bar(t1)
        alpha, beta = a_0 / a_1, 1 - a_0 / a_1  # 1 - beta_i and beta_i
        beta_tilde = beta * (1 - a_1) / (1 - a_0)
        expected = (x + beta * sde.score(x, t0)) / alpha.sqrt()

        rng.manual_seed(0)
        expected = expected + beta_tilde.sqrt() * solver.randn_like(x, seed=0)
        rng.manual_seed(0)
        out, _ = solver.step(sde, t0, t1, x)
        assert torch.allclose(out, expected, atol=1e-10)


@torch.no_grad()
def test_ddim_solver_matches_edm_euler_on_ve(device, rng):
    """On a variance-exploding SDE, `DDIMSolver(eta=0)` is the Euler sampler of Karras et al."""
    sde = _toy_sde(VarianceExplodingDiffusion, device)
    solver = DDIMSolver(eta=0.0, timesteps=torch.linspace(0.9, 1e-3, 3), rng=rng)
    x = torch.randn(4, 1, 8, 8, dtype=torch.float64, device=device) * 0.7

    for t0, t1 in [(0.9, 0.85), (0.5, 0.4), (0.05, 1e-3)]:
        t0 = torch.tensor(t0, dtype=torch.float64, device=device)
        t1 = torch.tensor(t1, dtype=torch.float64, device=device)
        sigma_0, sigma_1 = sde.sigma_t(t0), sde.sigma_t(t1)
        expected = x + (sigma_1 - sigma_0) * (x - sde.denoised(x, t0)) / sigma_0
        out, _ = solver.step(sde, t0, t1, x)
        assert torch.allclose(out, expected, atol=1e-10)


@torch.no_grad()
@pytest.mark.parametrize("sde_class", ANCESTRAL_SDE_CLASSES)
@pytest.mark.parametrize("solver_class", [DDIMSolver, DDPMSolver])
def test_ancestral_solver_runs_on_every_sde(sde_class, solver_class, device, rng):
    """The same solver must run unchanged on every schedule, including down to `t_end = 0`."""
    timesteps = torch.linspace(0.9, 0.0, 6, dtype=torch.float64, device=device)
    sde = _toy_sde(sde_class, device)
    sde.solver = solver_class(timesteps=timesteps, rng=rng)
    sample = sde.sample((2, 1, 8, 8), seed=0)
    assert sample.shape == (2, 1, 8, 8)
    assert torch.isfinite(sample).all()


@torch.no_grad()
def test_ancestral_solver_rejects_bare_sde(device, rng):
    """A bare `BaseSDE` carries no noise schedule, so the solver must say so instead of failing obscurely."""
    sde = dinv.sampling.BaseSDE(drift=lambda x, t: x, diffusion=lambda t: t)
    solver = DDIMSolver(timesteps=torch.linspace(0.9, 1e-3, 3), rng=rng)
    with pytest.raises(ValueError, match="requires an SDE exposing"):
        solver.sample(sde, torch.zeros(1, 1, 4, 4))


@torch.no_grad()
def test_ancestral_solver_warns_when_alpha_is_ignored(device, rng):
    """`alpha` is ignored by the ancestral solvers; the user must be told rather than silently surprised."""
    timesteps = torch.linspace(0.9, 1e-3, 3, dtype=torch.float64, device=device)
    sde = _toy_sde(VariancePreservingDiffusion, device, alpha=0.5)
    solver = DDIMSolver(eta=0.0, timesteps=timesteps, rng=rng)
    with pytest.warns(UserWarning, match="is ignored by DDIMSolver"):
        solver.sample(sde, torch.zeros(1, 1, 4, 4, dtype=torch.float64, device=device))

    # eta**2 == alpha: consistent, so no warning
    import warnings as _warnings

    consistent = DDIMSolver(eta=0.5**0.5, timesteps=timesteps, rng=rng)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        consistent.sample(
            sde, torch.zeros(1, 1, 4, 4, dtype=torch.float64, device=device)
        )


@torch.no_grad()
@pytest.mark.parametrize("solver_class", [EulerSolver, DDIMSolver, DDPMSolver])
def test_posterior_sde_exposes_schedule(solver_class, device, rng):
    """`PosteriorDiffusion` must work with every solver, including the ones needing the noise schedule."""
    timesteps = torch.linspace(0.9, 1e-3, 4, dtype=torch.float64, device=device)
    denoiser = _ToyDenoiser()
    sde = _toy_sde(VariancePreservingDiffusion, device)
    solver = solver_class(timesteps=timesteps, rng=rng)
    posterior = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=denoiser),
        sde=sde,
        denoiser=denoiser,
        solver=solver,
        minus_one_one=False,
        dtype=torch.float64,
        device=device,
    )
    # the posterior SDE mirrors the unconditional interface
    assert isinstance(posterior.posterior, dinv.sampling.PosteriorSDE)
    for name in ("scale_t", "sigma_t", "score", "denoised"):
        assert hasattr(posterior.posterior, name)

    physics = dinv.physics.Inpainting(img_size=(1, 8, 8), mask=0.5, device=device)
    x = torch.rand(2, 1, 8, 8, dtype=torch.float64, device=device)
    y = physics(x)
    x_hat = posterior(y, physics, x_init=(2, 1, 8, 8), seed=0)
    assert x_hat.shape == (2, 1, 8, 8)
    assert torch.isfinite(x_hat).all()


@torch.no_grad()
def test_posterior_denoised_matches_conditional_tweedie(device, rng):
    """The posterior `denoised` must be Tweedie applied to the *conditional* score."""
    timesteps = torch.linspace(0.9, 1e-3, 4, dtype=torch.float64, device=device)
    denoiser = _ToyDenoiser()
    sde = _toy_sde(VariancePreservingDiffusion, device)
    posterior = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=denoiser),
        sde=sde,
        denoiser=denoiser,
        solver=DDIMSolver(timesteps=timesteps, rng=rng),
        minus_one_one=False,
        dtype=torch.float64,
        device=device,
    )
    physics = dinv.physics.Inpainting(img_size=(1, 8, 8), mask=0.5, device=device)
    x = torch.rand(2, 1, 8, 8, dtype=torch.float64, device=device)
    y = physics(x)
    state = torch.randn(2, 1, 8, 8, dtype=torch.float64, device=device) * 0.5
    t = 0.5

    cond = posterior.posterior
    scale, sigma = cond.scale_t(t), cond.sigma_t(t)
    expected = (
        state + (scale * sigma) ** 2 * cond.score(state, t, y=y, physics=physics)
    ) / scale
    assert torch.allclose(
        cond.denoised(state, t, y=y, physics=physics), expected, atol=1e-9
    )
    # and it must differ from the unconditional one: the measurement has to enter
    assert not torch.allclose(
        cond.denoised(state, t, y=y, physics=physics), sde.denoised(state, t), atol=1e-6
    )
