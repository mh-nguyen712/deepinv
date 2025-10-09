r"""
Tour of blur operators
===================================================

This example provides a tour of 2D blur operators in DeepInverse.
In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.

"""

# %%
# %%
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_example


# %% Load test images
# ----------------
#
# First, let's load some test images.

dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (173, 125)

x_rgb = load_example(
    "CBSD_0010.png", grayscale=False, device=device, dtype=dtype, img_size=img_size
)

x_gray = load_example(
    "barbara.jpeg", grayscale=True, device=device, dtype=dtype, img_size=img_size
)

# Next, set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# %%
# We are now ready to explore the different blur operators.
#
# Convolution Basics
# ------------------
#
# The class :class:`deepinv.physics.Blur` implements convolution operations with kernels.
#
# For instance, here is the convolution of a grayscale image with a grayscale filter:
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device)
y = physics(x_gray)
plot(
    [x_gray, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Grayscale convolution",
)

# %%
# When a single channel filter is used, all channels are convolved with the same filter:
#

physics = dinv.physics.Blur(filter_0, device=device)
y = physics(x_rgb)
plot(
    [x_rgb, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="RGB image + grayscale filter convolution",
)

# %%
# By default, the boundary conditions are ``'valid'``, but other options among (``'circular'``, ``'reflect'``, ``'replicate'``) are possible:
#

physics = dinv.physics.Blur(filter_0, padding="reflect", device=device)
y = physics(x_rgb)
plot(
    [x_rgb, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Reflection boundary conditions",
)

# %%
# For circular boundary conditions, an FFT implementation is also available. It is slower that :class:`deepinv.physics.Blur`,
# but inherits from :class:`deepinv.physics.DecomposablePhysics`, so that the pseudo-inverse and regularized inverse are computed faster and more accurately.
#
physics = dinv.physics.BlurFFT(img_size=x_rgb[0].shape, filter=filter_0, device=device)
y = physics(x_rgb)
plot(
    [x_rgb, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="FFT convolution with circular boundary conditions",
)

# %%
# One can also change the blur filter in the forward pass as follows:
filter_90 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=90.0).to(
    device=device, dtype=dtype
)
y = physics(x_rgb, filter=filter_90)
plot(
    [x_rgb, filter_90, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Changing the filter on the fly",
)

# %%
# When applied to a new image, the last filter is used:
y = physics(x_gray, filter=filter_90)
plot(
    [x_gray, filter_90, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Effect of on the fly change is persistent",
)

# %%
# We can also define color filters. In that situation, each channel is convolved with the corresponding channel of the filter:
psf_size = 9
filter_rgb = torch.zeros((1, 3, psf_size, psf_size), device=device, dtype=dtype)
filter_rgb[:, 0, :, psf_size // 2 : psf_size // 2 + 1] = 1.0 / psf_size
filter_rgb[:, 1, psf_size // 2 : psf_size // 2 + 1, :] = 1.0 / psf_size
filter_rgb[:, 2, ...] = (
    torch.diag(torch.ones(psf_size, device=device, dtype=dtype)) / psf_size
)
y = physics(x_rgb, filter=filter_rgb)
plot(
    [x_rgb, filter_rgb, y],
    titles=["signal", "Colour filter", "measurement"],
    suptitle="Color image + color filter convolution",
)

# %%
# Blur generators
# ----------------------
# More advanced kernel generation methods are provided with the toolbox thanks to
# the  :class:`deepinv.physics.generator.PSFGenerator`. In particular, motion blurs generators are implemented.

# %%
# Motion blur generators
# ~~~~~~~~~~~~~~~~~~~~~~
from deepinv.physics.generator import MotionBlurGenerator

# %%
# In order to generate motion blur kernels, we just need to instantiate a generator with specific the psf size.
# In turn, motion blurs can be generated on the fly by calling the ``step()`` method. Let's illustrate this now and
# generate 3 motion blurs. First, we instantiate the generator:
#
psf_size = 31
motion_generator = MotionBlurGenerator((psf_size, psf_size), device=device, dtype=dtype)
# %%
# To generate new filters, we call the step() function:
filters = motion_generator.step(batch_size=3)
# the `step()` function returns a dictionary:
print(filters.keys())
plot(
    [f for f in filters["filter"]],
    suptitle="Examples of randomly generated motion blurs",
)

# %%
# Other options, such as the regularity and length of the blur trajectory can also be specified:
motion_generator = MotionBlurGenerator(
    (psf_size, psf_size), l=0.6, sigma=1, device=device, dtype=dtype
)
filters = motion_generator.step(batch_size=3)
plot([f for f in filters["filter"]], suptitle="Different length and regularity")

# %%
# Diffraction blur generators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We also implemented diffraction blurs obtained through Fresnel theory and definition of the psf through the pupil
# plane expanded in Zernike polynomials

from deepinv.physics.generator import DiffractionBlurGenerator

diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), device=device, dtype=dtype
)

# %%
# Then, to generate new filters, it suffices to call the step() function as follows:

filters = diffraction_generator.step(batch_size=3)

# %%
# In this case, the `step()` function returns a dictionary containing the filters,
# their pupil function and Zernike coefficients:
print(filters.keys())

# Note that we use **0.2 to increase the image dynamics
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="Examples of randomly generated diffraction blurs",
)
plot(
    [
        f
        for f in torch.angle(filters["pupil"][:, None])
        * torch.abs(filters["pupil"][:, None])
    ],
    suptitle="Corresponding pupil phases",
)
print("Coefficients of the decomposition on Zernike polynomials")
print(filters["coeff"])

# %%
# We can change the cutoff frequency (below 1/4 to respect Shannon's sampling theorem)
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=1 / 8, device=device, dtype=dtype
)
filters = diffraction_generator.step(batch_size=3)
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="A different cutoff frequency",
)

# %%
# It is also possible to directly specify the Zernike decomposition.
# For instance, if the pupil is null, the PSF is the Airy pattern
n_zernike = diffraction_generator.n_zernike
filters = diffraction_generator.step(coeff=torch.zeros(3, n_zernike, device=device))
plot(
    [f for f in filters["filter"][:, None] ** 0.3],
    suptitle="Airy pattern",
)

# %%
# Finally, notice that you can activate the aberrations you want in the ANSI (or Noll)
# nomenclature https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size),
    fc=1 / 8,
    zernike_index=(5, 6),
    index_convention="ansi",
    device=device,
    dtype=dtype,
)
filters = diffraction_generator.step(batch_size=3)
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="PSF obtained with astigmatism only",
)

# %%
# Generator Mixture
# ~~~~~~~~~~~~~~~~~
#
# During training, it's more robust to train on multiple family of operators. This can be done
# seamlessly with the :class:`deepinv.physics.generator.GeneratorMixture`.

from deepinv.physics.generator import GeneratorMixture

torch.cuda.manual_seed(4)
torch.manual_seed(6)

generator = GeneratorMixture(
    ([motion_generator, diffraction_generator]), probs=[0.5, 0.5]
)
for i in range(4):
    filters = generator.step(batch_size=3)
    plot(
        [f for f in filters["filter"]],
        suptitle=f"Random PSF generated at step {i + 1}",
    )

# %%
# Space varying blurs
# --------------------
#
# Space varying blurs are also available using :class:`deepinv.physics.SpaceVaryingBlur`
#
# Two methods are implemented: the first one is based on a low-rank approximation of the operator
# using a few eigen-psf, while the second one is based on a tiling of the image in patches
# with locally invariant blur.
#
# We plot the impulse responses at different spatial locations by convolving a Dirac comb with the operator.

from deepinv.physics.generator import (
    ProductConvolutionBlurGenerator,
)
from deepinv.physics.blur import SpaceVaryingBlur

img_size = (256, 256)
n_eigenpsf = 16
spacing = (64, 64)
padding = "valid"
batch_size = 1

# Creating a Dirac comb to visualize the impulse responses
delta = 32
dirac_comb = torch.zeros((1, 1, *img_size), device=device)
dirac_comb[0, 0, ::delta, ::delta] = 1

# Now, scattered random psfs are synthesized and interpolated spatially
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size),
    fc=0.15,
    list_param=["Z4", "Z5", "Z6", "Z7", "Z8"],
    max_zernike_amplitude=0.2,
    device=device,
    dtype=dtype,
)
import numpy as np


class RotationGenerator(dinv.physics.generator.PSFGenerator):
    def __init__(self, psf_size, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.psf_size = psf_size
        self.dtype = dtype
        self.device = device

    def step(self, batch_size=1, seed=None, **kwargs):
        angle = np.linspace(0, 360, batch_size)
        psf = []
        for a in angle:
            f = dinv.physics.blur.gaussian_blur(sigma=(3, 0.1), angle=a)
            psf.append(f)
        psf = torch.cat(psf, dim=0)
        return {"filter": psf}


rotation_generator = RotationGenerator((psf_size, psf_size), device=device, dtype=dtype)
# First method: eigen-psfs
pc_generator = ProductConvolutionBlurGenerator(
    # psf_generator=diffraction_generator,
    psf_generator=rotation_generator,
    img_size=img_size,
    method="eigen_psf",
    n_eigen_psf=n_eigenpsf,
    spacing=spacing,
    padding=padding,
    device=device,
)
params_pc = pc_generator.step(batch_size, seed=0)

physics = SpaceVaryingBlur(**params_pc, device=device)

psf_grid_eigen = physics(dirac_comb)

plot(
    psf_grid_eigen.abs() ** 0.5,
    titles="Space varying impulse responses -- Eigen",
    rescale_mode="min_max",
    cbar=True,
    figsize=(5, 5),
)

# grid_psf = rotation_generator.step(batch_size=16, seed=0)["filter"]
# plot([_ for _ in grid_psf[5:8] ** 0.5])
# plot([_ for _ in grid_psf[9:12] ** 0.5])
# plot([_ for _ in grid_psf[13:16] ** 0.5])

# %%
pc_generator = ProductConvolutionBlurGenerator(
    psf_generator=rotation_generator,
    img_size=img_size,
    method="tiled_psf",
    patch_size=(64, 64),
    spacing=spacing,
    padding=padding,
    overlap=32,
    device=device,
)
params_pc = pc_generator.step(batch_size, seed=0)
physics = SpaceVaryingBlur(**params_pc, device=device)

psf_grid_tiled = physics(dirac_comb)
plot(
    psf_grid_tiled.abs() ** 0.5,
    titles="Space varying impulse responses -- Tiled",
    rescale_mode="min_max",
    cbar=True,
    figsize=(5, 5),
)

# %%
img_size = 32
patch_size = 16
overlap = 8
centers = ProductConvolutionBlurGenerator.get_tile_centers(
    img_size=img_size, patch_size=patch_size, overlap=overlap
)
# %%
# rotation_generator = RotationGenerator((1, 1), device=device, dtype=dtype)
# psf_list = rotation_generator.step(batch_size=centers.size(0))["filter"].transpose(0, 1)[None]

psf_list = torch.zeros(1, 1, 9, 5, 5, device=device)
psf_list[0, 0, :, 2, 2] = 1

generator = ProductConvolutionBlurGenerator(
    img_size=img_size,
    patch_size=patch_size,
    overlap=overlap,
    method="tiled_psf",
    device=device,
)
params = generator.step_from_psfs(psfs=psf_list)

physics = SpaceVaryingBlur(**params, device=device)
import torch.nn.functional as F

for i in range(16):
    dirac_comb = torch.zeros((1, 1, img_size, img_size), device=device)
    dirac_comb[0, 0, i::8, ::8] = 1
    psf_grid_eigen = physics(dirac_comb)
    psf_grid_eigen = F.pad(psf_grid_eigen, (2, 2, 2, 2))

    plot(
        [dirac_comb, psf_grid_eigen.abs() ** 0.5],
        titles=["Dirac comb", "Impulse responses"],
        suptitle="Space varying impulse responses -- Tile",
        rescale_mode="min_max",
        cbar=True,
        figsize=(5, 5),
    )
# %%
generator = ProductConvolutionBlurGenerator(
    img_size=img_size,
    patch_size=patch_size,
    method="eigen_psf",
    device=device,
)
params = generator.step_from_psfs(psfs=psf_list, psf_centers=centers / img_size )

physics = SpaceVaryingBlur(**params, device=device)

for i in range(16):
    dirac_comb = torch.zeros((1, 1, img_size, img_size), device=device)
    dirac_comb[0, 0, i::8, ::8] = 1
    psf_grid_eigen = physics(dirac_comb)
    psf_grid_eigen = F.pad(psf_grid_eigen, (2, 2, 2, 2))

    plot(
        [dirac_comb, psf_grid_eigen.abs() ** 0.5],
        titles=["Dirac comb", "Impulse responses"],
        suptitle="Space varying impulse responses -- Eigen",
        rescale_mode="min_max",
        cbar=True,
        figsize=(5, 5),
    )