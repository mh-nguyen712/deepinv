# %%
import torch
import torch.nn as nn
from sde import EDMSDE
from sde_solver import EulerSolver, HeunSolver
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
from edm import load_model
import numpy as np
from utils import get_edm_parameters

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ScorePrior(dinv.optim.prior.Prior):
    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False

    def grad(self, x: torch.Tensor, sigma_denoiser, *args, **kwargs):
        return (1 / sigma_denoiser**2) * (
            x - self.denoiser(x, sigma_denoiser, *args, **kwargs)
        )


# model = load_model("edm-imagenet-64x64-cond-adm.pkl").to(device)
model = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)

def denoiser(x, t, *args, **kwargs):
    return model(x.to(torch.float32), t, *args, **kwargs).to(torch.float64)


prior = ScorePrior(denoiser=denoiser)
url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)

# %%
x_noisy = x + torch.randn_like(x) * 40.0

dinv.utils.plot([x, x_noisy, denoiser(x_noisy, 40.0)])


# %%
def edm_sampler(
    model: nn.Module,
    latents: torch.Tensor,
    class_labels=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    # Time step discretization.
    step_indices = np.arange(num_steps)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = np.concatenate([t_steps, np.zeros_like(t_steps[:1])])  # t_N = 0

    print(t_steps)
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    # 0, ..., N-1
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily.

        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + np.sqrt(t_hat**2 - t_cur**2) * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = model(x_hat.to(torch.float32), t_hat, class_labels=class_labels).to(
            torch.float64
        )
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(
                x_next.to(torch.float32), t_next, class_labels=class_labels
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# %%
class_labels = torch.eye(1000, device=device)[
    torch.randint(1000, size=[2], device=device)
]
# class_labels = None
with torch.no_grad():
    latents = torch.randn(2, 3, 64, 64, device=device)
    samples = edm_sampler(
        model, latents=latents, class_labels=class_labels, num_steps=20
    )
    dinv.utils.plot([latents, samples])

# %%

sde = EDMSDE(
    prior=prior, name="ve", use_backward_ode=True, device=device, dtype=torch.float64
)

# %%
num_steps = 30
with torch.no_grad():
    # endpoint = sde.forward_sde.sample(x, ve_timesteps[::-1])
    # print(f"End point std: {endpoint.std()}")
    # dinv.utils.plot(endpoint)
    shape = (2, 3, 64, 64)
    samples = sde(shape=shape, method="heun", max_iter=num_steps)
dinv.utils.plot(samples)

# %% ODE Flow
params = get_edm_parameters('edm')
timesteps_fn = params["timesteps_fn"]
sigma_max = params["sigma_max"]
sde_T = params["sigma_inv_fn"](params["sigma_max"])

sigma_fn = params["sigma_fn"]
sigma_deriv_fn = params["sigma_deriv_fn"]
beta_fn = params["beta_fn"]
s_fn = params["s_fn"]
s_deriv_fn = params["s_deriv_fn"]

def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))
from scipy import integrate


# Black-box ODE solver for the probability flow ODE
def ode_flow(x, t):
    return (
        (s_deriv_fn(t) / s_fn(t)) * x
        - (s_fn(t) ** 2)
        * sigma_deriv_fn(t)
        * sigma_fn(t)
        * prior.grad(x / s_fn(t), sigma_fn(t))
    )
def ode_func(t, x):
    x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
    drift = ode_flow(x, t)
    return to_flattened_numpy(drift)

noise = torch.randn(2,3,64,64, device=device, dtype=torch.float32) * sigma_max
solution = integrate.solve_ivp(ode_func, (1., 1e-3), to_flattened_numpy(noise),
                                rtol=1e-5, atol=1e-5, method='RK45')
nfe = solution.nfev
ode_sample = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
dinv.utils.plot([noise, ode_sample])
# %%
