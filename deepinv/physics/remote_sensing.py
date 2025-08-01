from deepinv.physics.noise import GaussianNoise
from deepinv.physics.forward import StackedLinearPhysics
from deepinv.physics.blur import Downsampling
from deepinv.physics.range import Decolorize
from deepinv.utils.tensorlist import TensorList


class Pansharpen(StackedLinearPhysics):
    r"""
    Pansharpening forward operator.

    The measurements consist of a high resolution grayscale image and a low resolution RGB image, and
    are represented using :class:`deepinv.utils.TensorList`, where the first element is the RGB image and the second
    element is the grayscale image.

    By default, the downsampling is done with a gaussian filter with standard deviation equal to the downsampling,
    however, the user can provide a custom downsampling filter.

    It is possible to assign a different noise model to the RGB and grayscale images.

    :param tuple[int] img_size: size of the high-resolution multispectral input image, must be of shape (C, H, W).
    :param torch.Tensor, str, None filter: Downsampling filter. It can be 'gaussian', 'bilinear' or 'bicubic' or a
        custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param int factor: downsampling factor/ratio.
    :param str, tuple, list srf: spectral response function of the decolorize operator to produce grayscale from multispectral.
        See :class:`deepinv.physics.Decolorize` for parameter options. Defaults to ``flat`` i.e. simply average the bands.
    :param bool use_brovey: if ``True``, use the `Brovey method :footcite:t:`vivone2014critical`.
        to compute the pansharpening, otherwise use the conjugate gradient method.
    :param torch.nn.Module noise_color: noise model for the RGB image.
    :param torch.nn.Module noise_gray: noise model for the grayscale image.
    :param torch.device, str device: torch device.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.
    :param bool normalize: if ``True``, normalize the downsampling operator to have unit norm.
    :param float eps: small value to avoid division by zero in the Brovey method.

    |sep|

    :Examples:

        Pansharpen operator applied to a random 32x32 image:

        >>> from deepinv.physics import Pansharpen
        >>> import torch
        >>> x = torch.randn(1, 3, 32, 32) # Define random 32x32 color image
        >>> physics = Pansharpen(img_size=x.shape[1:], device=x.device)
        >>> x.shape
        torch.Size([1, 3, 32, 32])
        >>> y = physics(x)
        >>> y[0].shape
        torch.Size([1, 3, 8, 8])
        >>> y[1].shape
        torch.Size([1, 1, 32, 32])

    """

    def __init__(
        self,
        img_size,
        filter="bilinear",
        factor=4,
        srf="flat",
        noise_color=GaussianNoise(sigma=0.0),
        noise_gray=GaussianNoise(sigma=0.05),
        use_brovey=True,
        device="cpu",
        padding="circular",
        normalize=False,
        eps=1e-6,
        **kwargs,
    ):
        assert len(img_size) == 3, "img_size must be of shape (C,H,W)"

        noise_color = noise_color if noise_color is not None else lambda x: x
        noise_gray = noise_gray if noise_gray is not None else lambda x: x
        self.use_brovey = use_brovey
        self.normalize = normalize
        self.eps = eps

        downsampling = Downsampling(
            img_size=img_size,
            factor=factor,
            filter=filter,
            noise_model=noise_color,
            device=device,
            padding=padding,
        )
        decolorize = Decolorize(
            srf=srf, noise_model=noise_gray, channels=img_size[0], device=device
        )

        super().__init__(physics_list=[downsampling, decolorize], **kwargs)

        # Set convenience attributes
        self.downsampling = downsampling
        self.decolorize = decolorize
        self.solver = "lsqr"  # more stable than CG

    def A_dagger(self, y: TensorList, **kwargs):
        """
        If the Brovey method is used, compute the classical Brovey solution, otherwise compute the conjugate gradient solution.

        See the review paper :footcite:t:`vivone2014critical` for more details.

        :param deepinv.utils.TensorList y: input tensorlist of (MS, PAN)
        :return: Tensor of image pan-sharpening using the Brovey method.
        """

        if self.use_brovey:
            if self.downsampling.filter is not None and not self.normalize:
                factor = self.downsampling.factor**2
            else:
                factor = 1

            x = self.downsampling.A_adjoint(y[0], **kwargs) * factor
            x = x * y[1] / (x.mean(1, keepdim=True) + self.eps)
            return x
        else:
            return super().A_dagger(y, **kwargs)
