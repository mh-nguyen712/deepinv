import torch
import numpy as np
from math import ceil, floor
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional import histogramdd, conv2d
from deepinv.physics.functional.interp import ThinPlateSpline


class PSFGenerator(PhysicsGenerator):
    r"""
    Base class for generating Point Spread Functions (PSFs).


    :param tuple psf_size: the shape of the generated PSF in 2D
        ``(kernel_size, kernel_size)``. If an `int` is given, it will be used for both dimensions.
    :param int num_channels: number of images channels. Defaults to 1.
    """

    def __init__(
        self,
        psf_size: tuple[int] = (31, 31),
        num_channels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(psf_size, int):
            psf_size = (psf_size, psf_size)

        self.shape = (num_channels,) + psf_size
        self.psf_size = psf_size
        self.num_channels = num_channels


class MotionBlurGenerator(PSFGenerator):
    r"""
    Random motion blur generator.

    See :footcite:t:`schuler2015learning` for more details.

    A blur trajectory is generated by sampling both its x- and y-coordinates independently
    from a Gaussian Process with a Matérn 3/2 covariance function.

    .. math::

        f_x(t), f_y(t) \sim \mathcal{GP}(0, k(t, t'))

    where :math:`k` is defined as

    .. math::

        k(t, s) = \sigma^2 \left( 1 + \frac{\sqrt{5} |t -s|}{l} + \frac{5 (t-s)^2}{3 l^2} \right) \exp \left(-\frac{\sqrt{5} |t-s|}{l}\right)

    :param tuple psf_size: the shape of the generated PSF in 2D, should be `(kernel_size, kernel_size)`
    :param int num_channels: number of images channels. Defaults to 1.
    :param float l: the length scale of the trajectory, defaults to 0.3
    :param float sigma: the standard deviation of the Gaussian Process, defaults to 0.25
    :param int n_steps: the number of points in the trajectory, defaults to 1000

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import MotionBlurGenerator
    >>> generator = MotionBlurGenerator((5, 5), num_channels=1)
    >>> blur = generator.step()  # dict_keys(['filter'])
    >>> print(blur['filter'].shape)
    torch.Size([1, 1, 5, 5])
    """

    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: type = torch.float32,
        l: float = 0.3,
        sigma: float = 0.25,
        n_steps: int = 1000,
    ) -> None:
        kwargs = {"l": l, "sigma": sigma, "n_steps": n_steps}
        if len(psf_size) != 2:
            raise ValueError(
                "psf_size must 2D. Add channels via num_channels parameter"
            )
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            rng=rng,
            **kwargs,
        )

    def matern_kernel(self, diff, sigma: float = None, l: float = None):
        r"""
        Compute the Matérn 3/2 covariance.

        :param torch.Tensor diff: the difference `t - s`
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory
        """
        if sigma is None:
            sigma = self.sigma
        if l is None:
            l = self.l
        fraction = 5**0.5 * diff.abs() / l
        return sigma**2 * (1 + fraction + fraction**2 / 3) * torch.exp(-fraction)

    def f_matern(self, batch_size: int = 1, sigma: float = None, l: float = None):
        r"""
        Generates the trajectory.

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process.
        :param float l: the length scale of the trajectory.
        :return: the trajectory of shape `(batch_size, n_steps)`
        """
        vec = torch.randn(
            batch_size, self.n_steps, generator=self.rng, **self.factory_kwargs
        )
        time = torch.linspace(-torch.pi, torch.pi, self.n_steps, **self.factory_kwargs)[
            None
        ]

        kernel = self.matern_kernel(time, sigma, l)
        kernel_fft = torch.fft.rfft(kernel)
        vec_fft = torch.fft.rfft(vec)
        return torch.fft.irfft(vec_fft * torch.sqrt(kernel_fft)).real[
            :,
            torch.arange(self.n_steps // (2 * torch.pi), **self.factory_kwargs).type(
                torch.int
            ),
        ]

    def step(
        self,
        batch_size: int = 1,
        sigma: float = None,
        l: float = None,
        seed: int = None,
        **kwargs,
    ):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory
        :param int seed: the seed for the random number generator.

        :return: dictionary with key **'filter'**: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        """
        self.rng_manual_seed(seed)
        f_x = self.f_matern(batch_size, sigma, l)[..., None]
        f_y = self.f_matern(batch_size, sigma, l)[..., None]

        trajectories = torch.cat(
            (
                f_x - torch.mean(f_x, dim=1, keepdim=True),
                f_y - torch.mean(f_y, dim=1, keepdim=True),
            ),
            dim=-1,
        )
        kernels = [
            histogramdd(trajectory, bins=list(self.psf_size), low=[-1, -1], upp=[1, 1])[
                None, None
            ]
            for trajectory in trajectories
        ]
        kernel = torch.cat(kernels, dim=0).to(**self.factory_kwargs)
        kernel = kernel / torch.sum(kernel, dim=(-2, -1), keepdim=True)
        return {
            "filter": kernel.expand(
                -1,
                self.num_channels,
                -1,
                -1,
            )
        }


class DiffractionBlurGenerator(PSFGenerator):
    r"""
    Diffraction limited blur generator.

    Generates 2D diffraction kernels in optics using Zernike decomposition of the phase mask
    (Fresnel/Fraunhoffer diffraction theory).

    Zernike polynomials are ordered following the
    Noll's sequential indices convention (https://en.wikipedia.org/wiki/Zernike_polynomials for more details).

    :param tuple psf_size: the shape ``H x W`` of the generated PSF in 2D
    :param int num_channels: number of images channels. Defaults to 1.
    :param list[str] list_param: list of activated Zernike coefficients in Noll's convention,
        defaults to ``["Z4", "Z5", "Z6","Z7", "Z8", "Z9", "Z10", "Z11"]``.
    :param float fc: cutoff frequency (NA/emission_wavelength) * pixel_size. Should be in ``[0, 0.25]``
        to respect the Shannon-Nyquist sampling theorem, defaults to ``0.2``.
    :param tuple[int] pupil_size: this is used to synthesize the super-resolved pupil.
        The higher the more precise, defaults to ``(256, 256)``.
        If a single ``int`` is given, a square pupil is considered.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import DiffractionBlurGenerator
    >>> generator = DiffractionBlurGenerator((5, 5), num_channels=3)
    >>> blur = generator.step()  # dict_keys(['filter', 'coeff', 'pupil'])
    >>> print(blur['filter'].shape)
    torch.Size([1, 3, 5, 5])


    """

    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        device: str = "cpu",
        dtype: type = torch.float32,
        rng: torch.Generator = None,
        list_param: list[str] = [
            "Z4",
            "Z5",
            "Z6",
            "Z7",
            "Z8",
            "Z9",
            "Z10",
            "Z11",
        ],
        fc: float = 0.2,
        max_zernike_amplitude: float = 0.15,
        pupil_size: tuple[int] = (256, 256),
    ):
        kwargs = {
            "list_param": list_param,
            "fc": fc,
            "pupil_size": pupil_size,
            "max_zernike_amplitude": max_zernike_amplitude,
        }
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            rng=rng,
            **kwargs,
        )

        self.list_param = list_param  # list of parameters to provide

        pupil_size = (
            max(self.pupil_size[0], self.psf_size[0]),
            max(self.pupil_size[1], self.psf_size[1]),
        )
        self.pupil_size = pupil_size

        lin_x = torch.linspace(-0.5, 0.5, self.pupil_size[0], **self.factory_kwargs)
        lin_y = torch.linspace(-0.5, 0.5, self.pupil_size[1], **self.factory_kwargs)
        self.step_rho = lin_x[1] - lin_x[0]

        # Fourier plane is discretized on [-0.5,0.5]x[-0.5,0.5]
        XX, YY = torch.meshgrid(lin_x / self.fc, lin_y / self.fc, indexing="ij")
        self.rho = cart2pol(XX, YY)  # Cartesian coordinates

        # The list of Zernike polynomial functions
        list_zernike_polynomial = define_zernike()

        # In order to avoid layover in Fourier convolution we need to zero pad and then extract a part of image
        # computed from pupil_size and psf_size

        self.pad_pre = (
            ceil((self.pupil_size[0] - self.psf_size[0]) / 2),
            ceil((self.pupil_size[1] - self.psf_size[1]) / 2),
        )
        self.pad_post = (
            floor((self.pupil_size[0] - self.psf_size[0]) / 2),
            floor((self.pupil_size[1] - self.psf_size[1]) / 2),
        )

        # a list of indices of the parameters
        self.index_params = np.sort([int(param[1:]) for param in list_param])
        assert (
            np.max(self.index_params) <= 38
        ), "The Zernike polynomial index can not be exceed 38"

        # the number of Zernike coefficients
        self.n_zernike = len(self.index_params)

        # the tensor of Zernike polynomials in the pupil plane
        self.Z = torch.zeros(
            (self.pupil_size[0], self.pupil_size[1], self.n_zernike),
            **self.factory_kwargs,
        )
        for k in range(len(self.index_params)):
            self.Z[:, :, k] = list_zernike_polynomial[self.index_params[k]](
                XX, YY
            )  # defining the k-th Zernike polynomial

    def __update__(self):
        r"""
        Update the device and dtype of Zernike polynomials and the coordinates
        """
        self.rho = self.rho.to(**self.factory_kwargs)
        self.Z = self.Z.to(**self.factory_kwargs)

    def step(
        self,
        batch_size: int = 1,
        coeff: torch.Tensor = None,
        seed: int = None,
        **kwargs,
    ):
        r"""
        Generate a batch of PFS with a batch of Zernike coefficients

        :param int batch_size: batch_size.
        :param torch.Tensor coeff: batch_size x len(list_param) coefficients of the Zernike decomposition (defaults is None)
        :param int seed: the seed for the random number generator.

        :return: dictionary with keys **'filter'**: tensor of size (batch_size x num_channels x psf_size[0] x psf_size[1]) batch of psfs,
            **'coeff'**: list of sampled Zernike coefficients in this realization, **'pupil'**: the pupil function
        :rtype: dict
        """
        self.__update__()
        self.rng_manual_seed(seed)
        if coeff is None:
            coeff = self.generate_coeff(batch_size)

        pupil1 = (self.Z @ coeff[:, : self.n_zernike].T).transpose(2, 0)
        pupil2 = torch.exp(-2.0j * torch.pi * pupil1)
        indicator_circ = bump_function(
            self.rho, 1 - self.step_rho / 2, b=self.step_rho / 2
        )
        pupil3 = pupil2 * indicator_circ
        psf1 = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(pupil3)))
        psf2 = torch.real(psf1 * torch.conj(psf1))

        psf3 = psf2[
            :,
            self.pad_pre[0] : self.pupil_size[0] - self.pad_post[0],
            self.pad_pre[1] : self.pupil_size[1] - self.pad_post[1],
        ].unsqueeze(1)

        psf = psf3 / torch.sum(psf3, dim=(-1, -2), keepdim=True)
        return {
            "filter": psf.expand(-1, self.shape[0], -1, -1),
            "coeff": coeff,
            "pupil": pupil3,
        }

    def generate_coeff(self, batch_size):
        r"""Generates random coefficients of the decomposition in the Zernike polynomials.

        :param int batch_size: batch_size.

        :return: a tensor of shape `(batch_size, len(list_param))` coefficients in the Zernike decomposition.

        """
        coeff = torch.rand(
            (batch_size, len(self.list_param)),
            generator=self.rng,
            **self.factory_kwargs,
        )
        coeff = (coeff - 0.5) * self.max_zernike_amplitude
        return coeff


def define_zernike():
    r"""
    Returns a list of Zernike polynomials lambda functions in Cartesian coordinates.

    :param list[func]: list of 37 lambda functions with the Zernike Polynomials. They are ordered as follows:

        Z1: Z0,0 Piston or Bias
        Z2: Z1,1 x Tilt
        Z3: Z1,-1 y Tilt
        Z4: Z2,0 Defocus
        Z5: Z2,-2 Primary Astigmatism at 45
        Z6: Z2,2 Primary Astigmatism at 0
        Z7: Z3,-1 Primary y Coma
        Z8: Z3,1 Primary x Coma
        Z9: Z3,-3 y Trefoil
        Z10: Z3,3 x Trefoil
        Z11: Z4,0 Primary Spherical
        Z12: Z4,2 Secondary Astigmatism at 0
        Z13: Z4,-2 Secondary Astigmatism at 45
        Z14: Z4,4 x Tetrafoil
        Z15: Z4,-4 y Tetrafoil
        Z16: Z5,1 Secondary x Coma
        Z17: Z5,-1 Secondary y Coma
        Z18: Z5,3 Secondary x Trefoil
        Z19: Z5,-3 Secondary y Trefoil
        Z20: Z5,5 x Pentafoil
        Z21: Z5,-5 y Pentafoil
        Z22: Z6,0 Secondary Spherical
        Z23: Z6,-2 Tertiary Astigmatism at 45
        Z24: Z6,2 Tertiary Astigmatism at 0
        Z25: Z6,-4 Secondary x Trefoil
        Z26: Z6,4 Secondary y Trefoil
        Z27: Z6-,6 Hexafoil Y
        Z28: Z6,6 Hexafoil X
        Z29: Z7,-1 Tertiary y Coma
        Z30: Z7,1 Tertiary x Coma
        Z31: Z7,-3 Tertiary y Trefoil
        Z32: Z7,3 Tertiary x Trefoil
        Z33: Z7,-5 Secondary Pentafoil Y
        Z34: Z7,5 Secondary Pentafoil X
        Z35: Z7,-7 Heptafoil Y
        Z36: Z7,7 Heptafoil X
        Z37: Z8,0 Tertiary Spherical
    """
    Z = [None for k in range(38)]

    def r2(x, y):
        return x**2 + y**2

    sq3 = 3**0.5
    sq5 = 5**0.5
    sq6 = 6**0.5
    sq7 = 7**0.5
    sq8 = 8**0.5
    sq10 = 10**0.5
    sq12 = 12**0.5
    sq14 = 14**0.5

    Z[0] = lambda x, y: torch.ones_like(x)  # piston
    Z[1] = lambda x, y: torch.ones_like(x)  # piston
    Z[2] = lambda x, y: 2 * x  # tilt x
    Z[3] = lambda x, y: 2 * y  # tilt y
    Z[4] = lambda x, y: sq3 * (2 * r2(x, y) - 1)  # defocus
    Z[5] = lambda x, y: 2 * sq6 * x * y
    Z[6] = lambda x, y: sq6 * (x**2 - y**2)
    Z[7] = lambda x, y: sq8 * y * (3 * r2(x, y) - 2)
    Z[8] = lambda x, y: sq8 * x * (3 * r2(x, y) - 2)
    Z[9] = lambda x, y: sq8 * y * (3 * x**2 - y**2)
    Z[10] = lambda x, y: sq8 * x * (x**2 - 3 * y**2)
    Z[11] = lambda x, y: sq5 * (6 * r2(x, y) ** 2 - 6 * r2(x, y) + 1)
    Z[12] = lambda x, y: sq10 * (x**2 - y**2) * (4 * r2(x, y) - 3)
    Z[13] = lambda x, y: 2 * sq10 * x * y * (4 * r2(x, y) - 3)
    Z[14] = lambda x, y: sq10 * (r2(x, y) ** 2 - 8 * x**2 * y**2)
    Z[15] = lambda x, y: 4 * sq10 * x * y * (x**2 - y**2)
    Z[16] = lambda x, y: sq12 * x * (10 * r2(x, y) ** 2 - 12 * r2(x, y) + 3)
    Z[17] = lambda x, y: sq12 * y * (10 * r2(x, y) ** 2 - 12 * r2(x, y) + 3)
    Z[18] = lambda x, y: sq12 * x * (x**2 - 3 * y**2) * (5 * r2(x, y) - 4)
    Z[19] = lambda x, y: sq12 * y * (3 * x**2 - y**2) * (5 * r2(x, y) - 4)
    Z[20] = (
        lambda x, y: sq12 * x * (16 * x**4 - 20 * x**2 * r2(x, y) + 5 * r2(x, y) ** 2)
    )
    Z[21] = (
        lambda x, y: sq12 * y * (16 * y**4 - 20 * y**2 * r2(x, y) + 5 * r2(x, y) ** 2)
    )
    Z[22] = lambda x, y: sq7 * (
        20 * r2(x, y) ** 3 - 30 * r2(x, y) ** 2 + 12 * r2(x, y) - 1
    )
    Z[23] = lambda x, y: 2 * sq14 * x * y * (15 * r2(x, y) ** 2 - 20 * r2(x, y) + 6)
    Z[24] = lambda x, y: sq14 * (x**2 - y**2) * (15 * r2(x, y) ** 2 - 20 * r2(x, y) + 6)
    Z[25] = lambda x, y: 4 * sq14 * x * y * (x**2 - y**2) * (6 * r2(x, y) - 5)
    Z[26] = (
        lambda x, y: sq14
        * (8 * x**4 - 8 * x**2 * r2(x, y) + r2(x, y) ** 2)
        * (6 * r2(x, y) - 5)
    )
    Z[27] = (
        lambda x, y: sq14
        * x
        * y
        * (32 * x**4 - 32 * x**2 * r2(x, y) + 6 * r2(x, y) ** 2)
    )
    Z[28] = lambda x, y: sq14 * (
        32 * x**6 - 48 * x**4 * r2(x, y) + 18 * x**2 * r2(x, y) ** 2 - r2(x, y) ** 3
    )
    Z[29] = (
        lambda x, y: 4
        * y
        * (35 * r2(x, y) ** 3 - 60 * r2(x, y) ** 2 + 30 * r2(x, y) + 10)
    )
    Z[30] = (
        lambda x, y: 4
        * x
        * (35 * r2(x, y) ** 3 - 60 * r2(x, y) ** 2 + 30 * r2(x, y) + 10)
    )
    Z[31] = (
        lambda x, y: 4
        * y
        * (3 * x**2 - y**2)
        * (21 * r2(x, y) ** 2 - 30 * r2(x, y) + 10)
    )
    Z[32] = (
        lambda x, y: 4
        * x
        * (x**2 - 3 * y**2)
        * (21 * r2(x, y) ** 2 - 30 * r2(x, y) + 10)
    )
    Z[33] = (
        lambda x, y: 4
        * (7 * r2(x, y) - 6)
        * (4 * x**2 * y * (x**2 - y**2) + y * (r2(x, y) ** 2 - 8 * x**2 * y**2))
    )
    Z[34] = lambda x, y: (
        4
        * (7 * r2(x, y) - 6)
        * (x * (r2(x, y) ** 2 - 8 * x**2 * y**2) - 4 * x * y**2 * (x**2 - y**2))
    )
    Z[35] = lambda x, y: (
        8 * x**2 * y * (3 * r2(x, y) ** 2 - 16 * x**2 * y**2)
        + 4 * y * (x**2 - y**2) * (r2(x, y) ** 2 - 16 * x**2 * y**2)
    )
    Z[36] = lambda x, y: (
        4 * x * (x**2 - y**2) * (r2(x, y) ** 2 - 16 * x**2 * y**2)
        - 8 * x * y**2 * (3 * r2(x, y) ** 2 - 16 * x**2 * y**2)
    )
    Z[37] = lambda x, y: 3 * (
        70 * r2(x, y) ** 4
        - 140 * r2(x, y) ** 3
        + 90 * r2(x, y) ** 2
        - 20 * r2(x, y)
        + 1
    )
    return Z


def cart2pol(x, y):
    r"""
    Cartesian to polar coordinates

    :param torch.Tensor x: x coordinates
    :param torch.Tensor y: y coordinates

    :return: rho of torch.Tensor of radius
    :rtype: tuple
    """

    rho = torch.sqrt(x**2 + y**2)
    return rho


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


class ProductConvolutionBlurGenerator(PhysicsGenerator):
    r"""
    Generates parameters of space-varying blurs.

    The parameters generated are  ``{'filters' : torch.tensor(...), 'multipliers': torch.tensor(...), 'padding': str}``
    see :class:`deepinv.physics.SpaceVaryingBlur` for more details.

    :param deepinv.physics.generator.PSFGenerator psf_generator: A psf generator
        (e.g. ``generator = DiffractionBlurGenerator((1, psf_size, psf_size), fc=0.25)``)
    :param tuple img_size: image size ``H x W``.
    :param int n_eigen_psf: each psf in the field of view will be a linear combination of ``n_eigen_psf`` eigen psf grids.
        Defaults to 10.
    :param tuple spacing: steps between the psf grids used for interpolation (defaults ``(H//8, W//8)``).
    :param str padding: boundary conditions in (options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``).
        Defaults to ``'valid'``.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import DiffractionBlurGenerator
    >>> from deepinv.physics.generator import ProductConvolutionBlurGenerator
    >>> psf_size = 7
    >>> psf_generator = DiffractionBlurGenerator((psf_size, psf_size), fc=0.25)
    >>> pc_generator = ProductConvolutionBlurGenerator(psf_generator, img_size=(64, 64), n_eigen_psf=8)
    >>> params = pc_generator.step(1)
    >>> print(params.keys())
    dict_keys(['filters', 'multipliers', 'padding'])

    """

    def __init__(
        self,
        psf_generator: PSFGenerator,
        img_size: tuple[int],
        n_eigen_psf: int = 10,
        spacing: tuple[int] = None,
        padding: str = "valid",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(spacing, int):
            spacing = (spacing, spacing)

        self.psf_generator = psf_generator
        self.img_size = img_size
        self.n_eigen_psf = n_eigen_psf
        self.spacing = (
            spacing
            if spacing is not None
            else (self.img_size[0] // 8, self.img_size[1] // 8)
        )
        self.padding = padding

        self.n_psf_prid = (self.img_size[0] // self.spacing[0]) * (
            self.img_size[1] // self.spacing[1]
        )

        # Interpolating the psf_grid coefficients with Thinplate splines
        T0 = torch.linspace(
            0, 1, self.img_size[0] // self.spacing[0], **self.factory_kwargs
        )
        T1 = torch.linspace(
            0, 1, self.img_size[1] // self.spacing[1], **self.factory_kwargs
        )
        yy, xx = torch.meshgrid(T0, T1, indexing="ij")
        self.X = torch.stack((yy.flatten(), xx.flatten()), dim=1)

        T0 = torch.linspace(0, 1, self.img_size[0], **self.factory_kwargs)
        T1 = torch.linspace(0, 1, self.img_size[1], **self.factory_kwargs)
        yy, xx = torch.meshgrid(T0, T1, indexing="ij")
        self.XX = torch.stack((yy.flatten(), xx.flatten()), dim=1)

        self.tps = ThinPlateSpline(0.0, **self.factory_kwargs)

    def step(self, batch_size: int = 1, seed: int = None, **kwargs):
        r"""
        Generates a random set of filters and multipliers for space-varying blurs.

        :param int batch_size: number of space-varying blur parameters to generate.
        :param int seed: the seed for the random number generator.

        :returns: a dictionary containing filters, multipliers and paddings.
            filters: a tensor of shape (B, C, n_eigen_psf, psf_size, psf_size).
            multipliers: a tensor of shape (B, C, n_eigen_psf, H, W).
        """
        self.rng_manual_seed(seed)
        self.psf_generator.rng_manual_seed(seed)

        # Generating psf_grid on a grid
        psf_grid = self.psf_generator.step(self.n_psf_prid * batch_size)["filter"]
        psf_size = psf_grid.shape[-2:]
        psf_grid = psf_grid.view(
            batch_size, self.n_psf_prid, psf_grid.size(1), *psf_size
        )

        # Computing the eigen-psf
        psf_grid = psf_grid.flatten(-2, -1).transpose(1, 2)
        _, _, V = torch.linalg.svd(psf_grid, full_matrices=False)
        V = V[..., : self.n_eigen_psf, :].transpose(-1, -2)
        eigen_psf = V.reshape(V.size(0), V.size(1), self.n_eigen_psf, *psf_size)

        coeffs = torch.matmul(psf_grid, V)

        self.tps.fit(self.X, coeffs)
        w = self.tps.transform(self.XX).transpose(-1, -2)
        w = w.reshape(w.size(0), w.size(1), self.n_eigen_psf, *self.img_size)

        # Ending
        params_blur = {"filters": eigen_psf, "multipliers": w, "padding": self.padding}
        return params_blur


class DiffractionBlurGenerator3D(PSFGenerator):
    r"""
    3D diffraction limited kernels using Zernike decomposition of the phase mask.

    Fresnel/Fraunhoffer diffraction theory, see :class:`deepinv.physics.generator.DiffractionBlurGenerator` for more details.

    :param tuple psf_size: give in the order (depth, height, width)
    :param int num_channels: number of channels. Default to 1.
    :param list[str] list_param: list of activated Zernike coefficients, defaults to ``["Z4", "Z5", "Z6","Z7", "Z8", "Z9", "Z10", "Z11"]``
    :param float fc: cutoff frequency (NA/emission_wavelength) * pixel_size. Should be in `[0, 1/4]` to respect Shannon, defaults to `0.2`
    :param float kb: wave number (NI/emission_wavelength) * pixel_size or (NA/NI) * fc. `Must be greater than fc`. Defaults to `0.3`.
    :param float max_zernike_amplitude: maximum amplitude of Zernike coefficients. Defaults to 0.15.
    :param tuple[int] pupil_size: this is used to synthesize the super-resolved pupil. The higher the more precise, defaults to (512, 512).
        If an int is given, a square pupil is considered.
    :param float stepz_pixel: Ratio between the physical size of the z direction to that in the x/y direction of the voxels in the 3D image.

    :return: a DiffractionBlurGenerator object

    .. note::

        NA: numerical aperture, NI: refraction index of the immersion medium,
        emission_wavelength: wavelength of the light,
        pixel_size: physical size of the pixels in the xy plane
        in the same unit as emission_wavelength

    |sep|

    :Examples:

    >>> import torch
    >>> from deepinv.physics.generator import DiffractionBlurGenerator3D
    >>> generator = DiffractionBlurGenerator3D((21, 51, 51), stepz_pixel = 2, list_param=['Z0'])
    >>> dict = generator.step()
    >>> filter = dict['filter']
    >>> print(filter.shape)
    torch.Size([1, 1, 21, 51, 51])
    >>> batch_size = 2
    >>> n_zernike = len(generator.generator2d.list_param)
    >>> dict = generator.step(batch_size=batch_size, coeff=0.1 * torch.rand(batch_size, n_zernike, **generator.factory_kwargs))
    >>> dict.keys()
    dict_keys(['filter', 'pupil', 'coeff'])


    """

    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        device: str = "cpu",
        dtype: type = torch.float32,
        list_param: list[str] = ["Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11"],
        fc: float = 0.2,
        kb: float = 0.25,
        max_zernike_amplitude: float = 0.15,
        pupil_size: tuple[int] = (512, 512),
        stepz_pixel: float = 1.0,
    ):
        if len(psf_size) != 3:
            raise ValueError(
                "You should provide a tuple of len == 3 to generate 3D PSFs."
            )

        kwargs = {
            "list_param": list_param,
            "fc": fc,
            "pupil_size": pupil_size,
            "max_zernike_amplitude": max_zernike_amplitude,
        }
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.generator2d = DiffractionBlurGenerator(
            psf_size=psf_size[1:],
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.stepz_pixel = stepz_pixel
        self.kb = kb
        self.nzs = self.psf_size[0]

    def __update__(self):
        # self.factory_kwargs = {"device": self.params.device, "dtype": self.params.dtype}
        self.generator2d.rho = self.generator2d.rho.to(**self.factory_kwargs)
        self.generator2d.Z = self.generator2d.Z.to(**self.factory_kwargs)

    def step(self, batch_size: int = 1, coeff: torch.Tensor = None, **kwargs):
        r"""
        Generate a batch of PSF with a batch of Zernike coefficients

        :return: dictionary with keys **'filter'**: tensor of size (batch_size x num_channels x psf_size[0] x psf_size[1]) batch of psfs,
             **'pupil'**: the pupil function, **'coeff'**: list of sampled Zernike coefficients in this realization,
        :rtype: dict
        """
        self.__update__()

        gen_dict = self.generator2d.step(batch_size=batch_size, coeff=coeff)

        pupil3 = gen_dict["pupil"]

        defocus = (
            torch.linspace(-self.nzs / 2, self.nzs / 2, self.nzs, **self.factory_kwargs)
            * self.stepz_pixel
        )

        d = ((self.kb) ** 2 - (self.generator2d.rho * self.fc) ** 2 + 0j) ** 0.5

        propKer = torch.exp(-1j * 2 * torch.pi * d * defocus[:, None, None]) + 0j
        p = pupil3[:, None, ...] * propKer[None, ...]
        p[torch.isnan(p)] = 0
        pshift = torch.fft.fftshift(p, dim=(-2, -1))
        pfft = torch.fft.fft2(pshift, dim=(-2, -1))
        psf1 = torch.fft.ifftshift(pfft, dim=(-2, -1))
        psf2 = torch.real(psf1 * torch.conj(psf1))

        psf3 = psf2[
            :,
            :,
            self.generator2d.pad_pre[0] : self.generator2d.pupil_size[0]
            - self.generator2d.pad_post[0],
            self.generator2d.pad_pre[1] : self.generator2d.pupil_size[1]
            - self.generator2d.pad_post[1],
        ].unsqueeze(1)

        psf = psf3 / torch.sum(psf3, dim=(-3, -2, -1), keepdim=True)

        return {
            "filter": psf.expand(-1, self.shape[0], -1, -1, -1),
            "pupil": pupil3,
            "coeff": gen_dict["coeff"],
        }


class ConfocalBlurGenerator3D(PSFGenerator):
    r"""
    Generates the 3D point spread function of a confocal laser scanning microsope.

    :param tuple psf_size: give in the order (depth, height, width)
    :param int num_channels: number of channels. Default to 1.
    :param list[str] list_param: list of activated Zernike coefficients, defaults to ``["Z4", "Z5", "Z6","Z7", "Z8", "Z9", "Z10", "Z11"]``
    :param float NI: Refractive index of  the immersion medium. Defaults to 1.51 (oil),
    :param float NA: Numerical aperture. Should be less than NI. Defaults to 1.37.
    :param float lambda_ill: Wavelength of the illumination light (fluorescence excitation). Defaults to 489e-9.
    :param float lambda_coll: Wavelength of the collection light (fluorescence emission). Defaults to 395e-9.
    :param float pixelsize_XY: Physical pixel size in the lateral direction (height, width). Defaults to 50e-9.
    :param float pixelsize_Z:  Physical pixel size in the axial direction (depth). Defaults to 100e-9.
    :param float pinhole_radius: Radius of pinhole in Airy units. Defaults to 1.
    :param float max_zernike_amplitude: maximum amplitude of Zernike coefficients. Defaults to 0.1.
    :param tuple[int] pupil_size: this is used to synthesize the super-resolved pupil. The higher the more precise, defaults to (512, 512).
            If an int is given, a square pupil is considered.
    :return: a DiffractionBlurGenerator object

    |sep|

    :Examples:

    >>> import torch
    >>> from deepinv.physics.generator import ConfocalBlurGenerator3D
    >>> generator = ConfocalBlurGenerator3D((21, 51, 51), list_param=['Z0'])
    >>> dict = generator.step()
    >>> filter = dict['filter']
    >>> print(filter.shape)
    torch.Size([1, 1, 21, 51, 51])
    >>> batch_size = 2
    >>> n_zernike = len(generator.generator_ill.generator2d.list_param)
    >>> dict = generator.step(batch_size=batch_size,
    ...                       coeff_ill = 0.1 * torch.rand(batch_size, n_zernike, **generator.factory_kwargs),
    ...                       coeff_coll = 0.1 * torch.rand(batch_size, n_zernike, **generator.factory_kwargs))
    >>> dict.keys()
    dict_keys(['filter', 'coeff_ill', 'coeff_coll'])

    """

    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        device: str = "cpu",
        dtype: type = torch.float32,
        list_param: list[str] = ["Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11"],
        NI: float = 1.51,
        NA: float = 1.37,
        lambda_ill: float = 489e-9,
        lambda_coll: float = 395e-9,
        pixelsize_XY: float = 50e-9,
        pixelsize_Z: float = 100e-9,
        pinhole_radius: float = 1,
        max_zernike_amplitude: float = 0.1,
        pupil_size: tuple[int] = (512, 512),
    ):
        if len(psf_size) != 3:
            raise ValueError(
                "You should provide a tuple of len == 3 to generate 3D PSFs."
            )

        kwargs = {
            "list_param": list_param,
            "pupil_size": pupil_size,
            "max_zernike_amplitude": max_zernike_amplitude,
        }
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.fc_ill = (
            NA / lambda_ill
        ) * pixelsize_XY  # cutoff frequency for illumination
        self.kb_ill = (NI / lambda_ill) * pixelsize_XY  # wavenumber for illumination

        self.fc_coll = (
            NA / lambda_coll
        ) * pixelsize_XY  # cutoff freauency for collection
        # wavenumber for collection
        self.kb_coll = (NI / lambda_coll) * pixelsize_XY  # wavenumber for collection
        self.pinhole_radius = pinhole_radius
        self.pixelsize_XY = pixelsize_XY
        self.pixel_size_Z = pixelsize_Z

        self.lambda_ill = lambda_ill
        self.lambda_coll = lambda_coll
        self.NI = NI
        self.NA = NA

        # Initialize generator for the Illumniation PSF
        self.generator_ill = DiffractionBlurGenerator3D(
            psf_size=psf_size,
            fc=self.fc_ill,
            kb=self.kb_ill,
            stepz_pixel=int(pixelsize_Z / pixelsize_XY),
            **self.factory_kwargs,
        )

        # Initialize generator for the Collection PSF
        self.generator_coll = DiffractionBlurGenerator3D(
            psf_size=psf_size,
            fc=self.fc_coll,
            kb=self.kb_coll,
            stepz_pixel=int(pixelsize_Z / pixelsize_XY),
            **self.factory_kwargs,
        )

    def __update__(self):
        self.generator_ill.generator2d.rho = self.generator_ill.generator2d.rho.to(
            **self.factory_kwargs
        )
        self.generator_ill.generator2d.Z = self.generator_ill.generator2d.Z.to(
            **self.factory_kwargs
        )

        self.generator_coll.generator2d.rho = self.generator_coll.generator2d.rho.to(
            **self.factory_kwargs
        )
        self.generator_coll.generator2d.Z = self.generator_coll.generator2d.Z.to(
            **self.factory_kwargs
        )

    def step(
        self,
        batch_size: int = 1,
        coeff_ill: torch.Tensor = None,
        coeff_coll: torch.Tensor = None,
        **kwargs,
    ):
        r"""
        Generate a batch of 3D confocal PSF with a batch of Zernike coefficients
        for illumination and collection

        :return: dictionary with keys **'filter'**: tensor of size (batch_size x num_channels x psf_size[0] x psf_size[1]) batch of psfs,
              **'coeff_ill'**: list of sampled Zernike coefficients in this realization of illumination,
              **'coeff_coll'**: list of sampled Zernike coefficients in this realization of collection,

        :rtype: dict
        """
        self.__update__()

        dict_ill = self.generator_ill.step(
            batch_size=batch_size, coeff=coeff_ill
        )  # generate illumuinition PSF
        psf_ill = dict_ill["filter"]
        coeff_ill = dict_ill["coeff"]
        dict_coll = self.generator_coll.step(
            batch_size=batch_size, coeff=coeff_coll
        )  # generate collection PSF
        psf_coll = dict_coll["filter"]
        coeff_coll = dict_coll["coeff"]

        # convolution of the collection PSF by pinhole
        # 1. Define the pinhole D
        airy_unit = 0.61 * self.lambda_coll / self.NA
        PH_radius = self.pinhole_radius * airy_unit
        lin_x = torch.linspace(
            -1.5 * PH_radius,
            1.5 * PH_radius,
            int(3 * PH_radius / self.pixelsize_XY),
            **self.factory_kwargs,
        )
        lin_y = torch.linspace(
            -1.5 * PH_radius,
            1.5 * PH_radius,
            int(3 * PH_radius / self.pixelsize_XY),
            **self.factory_kwargs,
        )
        PH_step_rho = lin_x[1] - lin_x[0]
        # The plane is discretized on [-1.5 * r_pinhole, 1.5 * r_pinhole] x  [-1.5 * r_pinhole, 1.5 * r_pinhole]
        XX, YY = torch.meshgrid(lin_x, lin_y, indexing="ij")
        PH_rho = torch.sqrt(XX**2 + YY**2)  # Cartesian coordinates
        D = bump_function(
            PH_rho, PH_radius - PH_step_rho / 2, b=PH_step_rho / 2
        )  # D(r) in equation

        # 2. Apply 2D convolution in all z planes
        psf_coll_convolved = torch.zeros(psf_coll.shape, **self.factory_kwargs)
        for i in range(psf_coll.shape[-3]):
            psf_coll_convolved[:, :, i] = conv2d(
                psf_coll[:, :, i], filter=D[None, None], padding="constant"
            )

        psf_confocal = psf_ill * psf_coll_convolved  # final PSF of confocal microscope

        psf = psf_confocal / torch.sum(psf_confocal, dim=(-3, -2, -1), keepdim=True)

        return {
            "filter": psf.expand(-1, self.shape[0], -1, -1, -1),
            "coeff_ill": coeff_ill,
            "coeff_coll": coeff_coll,
        }
