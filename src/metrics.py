"""Evaluation metrics: SSIM, pSNR, LPIPS."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred = pred.clamp(0, data_range)
    target = target.clamp(0, data_range)
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> float:
    pred = pred.clamp(0, data_range)
    target = target.clamp(0, data_range)
    c = pred.shape[1]
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).expand(c, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean().item())


class MetricMeter:
    def __init__(self, use_lpips: bool = True):
        self.use_lpips = use_lpips
        self._lpips = None
        self.reset()
        if use_lpips:
            try:
                import lpips  # type: ignore

                self._lpips = lpips.LPIPS(net="alex")
                self._lpips.eval()
                for p in self._lpips.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                print(f"[metrics] LPIPS unavailable ({e})")
                self._lpips = None

    def reset(self) -> None:
        self.n = 0
        self.sum_psnr = 0.0
        self.sum_ssim = 0.0
        self.sum_lpips = 0.0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        b = pred.shape[0]
        for i in range(b):
            p = pred[i : i + 1]
            t = target[i : i + 1]
            self.sum_psnr += psnr(p, t)
            self.sum_ssim += ssim(p, t)
            if self._lpips is not None:
                device = next(self._lpips.parameters()).device
                pc = p.clamp(0, 1).to(device)
                tc = t.clamp(0, 1).to(device)
                if pc.shape[1] == 1:
                    pc = pc.repeat(1, 3, 1, 1)
                    tc = tc.repeat(1, 3, 1, 1)
                self.sum_lpips += float(self._lpips(pc * 2 - 1, tc * 2 - 1).mean().item())
            self.n += 1

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        return {
            "psnr": self.sum_psnr / self.n,
            "ssim": self.sum_ssim / self.n,
            "lpips": self.sum_lpips / self.n if self._lpips is not None else float("nan"),
        }

    def combined_score(self) -> float:
        """Higher is better. Used for checkpoint selection."""
        m = self.compute()
        lp = m["lpips"] if m["lpips"] == m["lpips"] else 0.0  # NaN → 0
        return m["ssim"] + m["psnr"] / 50.0 + (1.0 - lp)
