import torch
import torch.nn.functional as F

from .base import Denoiser

class NonLocalMeans(Denoiser):
    r"""
    Non-local means denoiser.

    This implements the IPOL non-local means :footcite:t:`ipol.2011.bcm_nlm` variant with patch-based distances
    and an exponential weight lookup table. Parameter defaults follow the
    reference IPOL implementation for sigma in [0, 100] on the 0-255 scale.
    
    This implementation is a Pytorch re-implementation of the original C++ code available in the IPOL article :footcite:t:`ipol.2011.bcm_nlm`, and is optimized for GPU execution. It supports batch processing and can handle both grayscale and color images.
    
    See `this article <https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf>`_ for details.


    :param int patch_size: Half-size of the patch used to compute distances.
        If `None`, it is selected from the IPOL table based on sigma and channels.
    :param int window_size: Half-size of the search window around each pixel.
        If `None`, it is selected from the IPOL table based on sigma and channels.
    :param float filtering: Filtering parameter controlling decay of weights. 
        The higher the value, the more permissive one is in accepting patches. A higher `filtering` results in a smoother image, at the expense of blurring features. For a white Gaussian noise, a rule of thumb is to choose the value of `filtering` to be `sigma` of slightly less.
        
        If `None`, it is selected from the IPOL table based on sigma and channels.
        The weight scale uses `(filtering * sigma)^2`.
    """
    def __init__(
        self,
        patch_size: int | None = None,
        window_size: int | None = None,
        filtering: float | None = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.window_size = window_size
        self.filtering = filtering

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        sigma = self._handle_sigma(
            sigma, batch_size=x.shape[0], ndim=1, device=x.device
        )
        return self._nlmeans_ipol(
            x,
            sigma,
            patch_size=self.patch_size,
            search_window=self.window_size,
            filtering=self.filtering,
        )

    @staticmethod
    def _params_from_sigma_tensor(
        s_255: torch.Tensor, channels: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = s_255.device
        dtype = s_255.dtype
        B = s_255.size(0)

        patch_size = torch.zeros((B,), device=device, dtype=torch.int64)
        search_window = torch.zeros((B,), device=device, dtype=torch.int64)
        filtering = torch.zeros((B,), device=device, dtype=dtype)

        if channels == 1:
            m1 = (s_255 > 0.0) & (s_255 <= 15.0)
            m2 = (s_255 > 15.0) & (s_255 <= 30.0)
            m3 = (s_255 > 30.0) & (s_255 <= 45.0)
            m4 = (s_255 > 45.0) & (s_255 <= 75.0)
            m5 = (s_255 > 75.0) & (s_255 <= 100.0)

            patch_size[m1] = 1
            patch_size[m2] = 2
            patch_size[m3] = 3
            patch_size[m4] = 4
            patch_size[m5] = 5

            search_window[m1 | m2] = 10
            search_window[m3 | m4 | m5] = 17

            filtering[m1 | m2] = 0.4
            filtering[m3 | m4] = 0.35
            filtering[m5] = 0.30
        else:
            m1 = (s_255 > 0.0) & (s_255 <= 25.0)
            m2 = (s_255 > 25.0) & (s_255 <= 55.0)
            m3 = (s_255 > 55.0) & (s_255 <= 100.0)

            patch_size[m1] = 1
            patch_size[m2] = 2
            patch_size[m3] = 3

            search_window[m1] = 10
            search_window[m2 | m3] = 17

            filtering[m1] = 0.55
            filtering[m2] = 0.4
            filtering[m3] = 0.35

        return patch_size, search_window, filtering

    @staticmethod
    def _lut_exp(dif: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the exponential weight lookup table for the given distance values.
        The constant 1000 is used to scale the input to the exponential function, following the original IPOL implementation. 
        """
        
        dif = torch.clamp(dif, min=0.0)
        mask = dif < 29.0
        x_idx_lut = torch.floor(dif * 1000.0)
        frac = dif * 1000.0 - x_idx_lut
        y1 = torch.exp(-x_idx_lut / 1000.0)
        y2 = torch.exp(-(x_idx_lut + 1.0) / 1000.0)
        return torch.where(mask, y1 + (y2 - y1) * frac, 0.0)

    def _nlmeans_ipol(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        patch_size: int | None = None,
        search_window: int | None = None,
        filtering: float | None = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        s_255 = sigma * 255.0
        invalid = (s_255 <= 0.0) | (s_255 > 100.0)
        if torch.any(invalid) and (patch_size is None or search_window is None or filtering is None):
            raise ValueError("Sigma values must be in the range (0, 100] when using automatic parameter selection.")

        _patch_size, _search_window, _filtering = self._params_from_sigma_tensor(
            s_255, C
        )

        if patch_size is not None:
            patch_size = torch.full(
                (B,), int(patch_size), device=device, dtype=torch.int64
            )
        else:
            patch_size = _patch_size
        if search_window is not None:
            search_window = torch.full(
                (B,), int(search_window), device=device, dtype=torch.int64
            )
        else:
            search_window = _search_window
        if filtering is not None:
            filtering = torch.full((B,), float(filtering), device=device, dtype=dtype)
        else:
            filtering = _filtering

        max_patch_size = int(patch_size.max().item())
        max_window = int(search_window.max().item())

        iwl = (2 * patch_size + 1) ** 2
        icwl = C * iwl
        fH2_b = ((filtering * sigma) ** 2 * icwl).view(B, 1, 1, 1)
        bias_b = (2.0 * icwl * (sigma**2)).view(B, 1, 1, 1)

        y_idx = torch.arange(H, device=device).view(1, 1, H, 1)
        x_idx = torch.arange(W, device=device).view(1, 1, 1, W)
        r_xy = torch.min(
            torch.min(y_idx, H - 1 - y_idx), torch.min(x_idx, W - 1 - x_idx)
        )

        kernel_size = 2 * max_patch_size + 1
        kernel_bank = torch.zeros(
            (max_patch_size + 1, 1, kernel_size, kernel_size),
            dtype=dtype,
            device=device,
        )
        for r in range(max_patch_size + 1):
            start = max_patch_size - r
            end = max_patch_size + r + 1
            kernel_bank[r, 0, start:end, start:end] = 1.0

        r_mask_b = torch.minimum(r_xy, patch_size.view(B, 1, 1, 1)).to(
            dtype=torch.int64
        )
        r_onehot = F.one_hot(r_mask_b.squeeze(1), num_classes=max_patch_size + 1).to(
            dtype=dtype
        )
        r_onehot = r_onehot.permute(0, 3, 1, 2)

        x_padded = F.pad(
            x,
            (max_window, max_window, max_window, max_window),
            mode="constant",
            value=0,
        )

        total_weight = torch.zeros((B, 1, H, W), dtype=dtype, device=device)
        max_weight = torch.zeros((B, 1, H, W), dtype=dtype, device=device)

        if max_window > 0:
            offs = torch.arange(-max_window, max_window + 1, device=device)
            dy_grid, dx_grid = torch.meshgrid(offs, offs, indexing="ij")
            dy_list = dy_grid.flatten()
            dx_list = dx_grid.flatten()
            center_mask = (dy_list == 0) & (dx_list == 0)
            dy_list = dy_list[~center_mask]
            dx_list = dx_list[~center_mask]
            S = dy_list.numel()

            x_idx_full = x_idx.expand(1, 1, H, W)
            y_idx_full = y_idx.expand(1, 1, H, W)
            x_idx_shifted = x_idx_full + dx_list.view(1, S, 1, 1)
            y_idx_shifted = y_idx_full + dy_list.view(1, S, 1, 1)

            r_mask_b_hw = r_mask_b[:, 0].unsqueeze(1)
            valid_x = (x_idx_shifted >= r_mask_b_hw) & (
                x_idx_shifted <= W - 1 - r_mask_b_hw
            )
            valid_y = (y_idx_shifted >= r_mask_b_hw) & (
                y_idx_shifted <= H - 1 - r_mask_b_hw
            )

            dx_abs = dx_list.abs().view(1, S, 1, 1)
            dy_abs = dy_list.abs().view(1, S, 1, 1)
            bloc_valid = (dx_abs <= search_window.view(B, 1, 1, 1)) & (
                dy_abs <= search_window.view(B, 1, 1, 1)
            )
            valid_mask = valid_x & valid_y & bloc_valid

            Hp = H + 2 * max_window
            Wp = W + 2 * max_window
            y_base = torch.arange(H, device=device).view(1, 1, 1, H, 1) + max_window
            x_base = torch.arange(W, device=device).view(1, 1, 1, 1, W) + max_window
            y_idx_all = y_base + dy_list.view(1, S, 1, 1, 1)
            x_idx_all = x_base + dx_list.view(1, S, 1, 1, 1)
            lin_idx = (y_idx_all * Wp + x_idx_all).view(1, 1, -1)

            x_padded_flat = x_padded.view(B, C, Hp * Wp)
            idx_expanded = lin_idx.expand(B, C, -1)
            x_shifted_all = torch.gather(x_padded_flat, 2, idx_expanded)
            x_shifted_all = x_shifted_all.view(B, C, S, H, W).permute(0, 2, 1, 3, 4)
            diff_sq_sum_c = ((x.unsqueeze(1) - x_shifted_all) ** 2).sum(dim=2)

            diff_reshaped = diff_sq_sum_c.reshape(B * S, 1, H, W)
            conv_all = F.conv2d(diff_reshaped, kernel_bank, padding=max_patch_size)
            conv_all = conv_all.view(B, S, max_patch_size + 1, H, W)
            r_mask_exp = r_mask_b_hw.expand(B, S, H, W)
            dist = torch.gather(conv_all, 2, r_mask_exp.unsqueeze(2)).squeeze(2)

            fDif = torch.clamp(dist - bias_b, min=0.0) / fH2_b
            weight_all = self._lut_exp(fDif)
            weight_all = torch.where(valid_mask, weight_all, 0.0)

            max_weight = weight_all.max(dim=1, keepdim=True).values
            total_weight = weight_all.sum(dim=1, keepdim=True)

        total_weight += max_weight

        out_num = torch.zeros_like(x)
        valid_norm_mask = total_weight > torch.finfo(dtype).eps

        if max_window > 0:
            weight_norm = torch.where(valid_norm_mask, weight_all / total_weight, 0)

            w_masked = weight_norm.unsqueeze(2) * r_onehot.unsqueeze(1)
            w_masked = w_masked.reshape(B * S, max_patch_size + 1, H, W)
            conv_out = F.conv2d(
                w_masked, kernel_bank, padding=max_patch_size, groups=max_patch_size + 1
            )
            conv_out = conv_out.view(B, S, max_patch_size + 1, H, W)
            weight_summed = conv_out.sum(dim=2)

            out_num += (x_shifted_all * weight_summed.unsqueeze(2)).sum(dim=1)

        weight_norm_max = torch.where(valid_norm_mask, max_weight / total_weight, 0)
        w_masked = weight_norm_max * r_onehot
        conv_out = F.conv2d(
            w_masked, kernel_bank, padding=max_patch_size, groups=max_patch_size + 1
        )
        max_weight_summed = conv_out.sum(dim=1, keepdim=True)

        out_num += x * max_weight_summed

        count_mask = valid_norm_mask.to(dtype)
        c_masked = count_mask * r_onehot
        conv_out = F.conv2d(
            c_masked, kernel_bank, padding=max_patch_size, groups=max_patch_size + 1
        )
        out_count = conv_out.sum(dim=1, keepdim=True)

        out = torch.where(out_count > 0.0, out_num / out_count, x)
        return out
