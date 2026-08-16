"""
Metric-matched composite loss for SpectraRestore.

L = 1.00 · Charbonnier + 0.20 · (1−SSIM) + 0.05 · FFT-L1 + 0.10 · LPIPS (after warmup)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class SSIMLoss(nn.Module):
    """Differentiable SSIM → returns (1 − SSIM) for minimisation."""

    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", self._create_window(window_size, channel))

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            window = self.window.to(device=pred.device, dtype=pred.dtype)
        else:
            window = self.window
        c = pred.shape[1]
        if c != self.channel:
            window = self._create_window(self.window_size, c).to(device=pred.device, dtype=pred.dtype)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=c)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=c) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return 1.0 - ssim_map.mean()


class FFTLoss(nn.Module):
    """L1 on 2D FFT magnitudes — restores periodic semiconductor structure."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred, norm="ortho")
        target_f = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))


class LPIPSLoss(nn.Module):
    """Optional LPIPS wrapper. Degrades gracefully if the package is missing."""

    def __init__(self, net: str = "alex"):
        super().__init__()
        self.available = False
        self.lpips = None
        try:
            import lpips  # type: ignore

            self.lpips = lpips.LPIPS(net=net)
            for p in self.lpips.parameters():
                p.requires_grad_(False)
            self.available = True
        except Exception as e:
            print(f"[losses] LPIPS unavailable ({e}); perceptual term disabled.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return pred.new_zeros(())
        # LPIPS expects 3-channel in [-1, 1]
        def to_lpips(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return x * 2.0 - 1.0

        # clamp for LPIPS stability (GT is [0,1]; pred may briefly overshoot)
        pred_c = pred.clamp(0, 1)
        target_c = target.clamp(0, 1)
        return self.lpips(to_lpips(pred_c), to_lpips(target_c)).mean()


class CompositeRestoreLoss(nn.Module):
    def __init__(
        self,
        w_char: float = 1.0,
        w_ssim: float = 0.2,
        w_fft: float = 0.05,
        w_lpips: float = 0.1,
        lpips_warmup_frac: float = 0.2,
        total_iters: int = 200_000,
        use_lpips: bool = True,
    ):
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        self.w_lpips = w_lpips
        self.lpips_start = int(total_iters * lpips_warmup_frac)

        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.fft = FFTLoss()
        self.lpips = LPIPSLoss() if use_lpips else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, step: int = 0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_char = self.char(pred, target)
        l_ssim = self.ssim(pred, target)
        l_fft = self.fft(pred, target)

        loss = self.w_char * l_char + self.w_ssim * l_ssim + self.w_fft * l_fft
        logs = {
            "loss/char": float(l_char.detach()),
            "loss/ssim": float(l_ssim.detach()),
            "loss/fft": float(l_fft.detach()),
        }

        if self.lpips is not None and self.lpips.available and step >= self.lpips_start:
            l_lpips = self.lpips(pred, target)
            loss = loss + self.w_lpips * l_lpips
            logs["loss/lpips"] = float(l_lpips.detach())
        else:
            logs["loss/lpips"] = 0.0

        logs["loss/total"] = float(loss.detach())
        return loss, logs
