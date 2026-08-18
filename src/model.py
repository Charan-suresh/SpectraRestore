"""
SpectraRestore — NAFNet backbone + 2× PixelShuffle super-resolution tail.

Architecture (see SOLUTION.md §3.2):
  degraded (1×H×W) → input-only standardize → NAFNet U-Net @ input res
  → PixelShuffle(2) residual + bilinear-upsample(raw input) → absolute-intensity output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (NAFNet / Chen et al., ECCV 2022)
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (from NAFNet)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Nonlinear Activation Free block with SimpleGate + SCA."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_ch = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        self.norm2 = LayerNorm2d(c)
        ffn_ch = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return x + y * self.gamma


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Conv2d(n_feat, n_feat * 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SpectraRestore(nn.Module):
    """
    Joint denoise + 2× SR network.

    Args:
        width: channel width at the first stage (32 default ≈29M; 48 for large).
        enc_blk_nums: NAFBlocks per encoder stage.
        middle_blk_num: NAFBlocks in the bottleneck.
        dec_blk_nums: NAFBlocks per decoder stage.
        in_ch / out_ch: grayscale = 1.
        scale: upsampling factor (2 for this challenge).
        standardize: apply per-image (x-mean)/std to the *input features only*.
            Output stays in absolute intensity space (GT is [0,1]); we do NOT
            re-apply the degraded mean/std to the prediction.
    """

    def __init__(
        self,
        width: int = 32,
        enc_blk_nums: list[int] | None = None,
        middle_blk_num: int = 12,
        dec_blk_nums: list[int] | None = None,
        in_ch: int = 1,
        out_ch: int = 1,
        scale: int = 2,
        standardize: bool = True,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.scale = scale
        self.standardize = standardize
        self.width = width

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for n in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(n)])
            )
            self.downs.append(Downsample(chan))
            chan *= 2

        self.middle = nn.Sequential(
            *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blk_nums:
            self.ups.append(Upsample(chan))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(n)])
            )

        # 2× SR tail via sub-pixel convolution
        self.up_tail = nn.Sequential(
            nn.Conv2d(width, width * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, out_ch, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-initialize the final projection in the residual tail so that
        # the network initially outputs the global residual skip (bilinear baseline)
        if hasattr(self, "up_tail") and len(self.up_tail) > 0:
            last_conv = self.up_tail[-1]
            if isinstance(last_conv, nn.Conv2d):
                nn.init.zeros_(last_conv.weight)
                if last_conv.bias is not None:
                    nn.init.zeros_(last_conv.bias)

    @staticmethod
    def _per_image_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # mean/std over H,W per sample (and channel)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) degraded input — may exceed [0, 1].
        Returns:
            (B, 1, scale·H, scale·W) restored image in absolute intensity space
            (clamp to [0, 1] only when saving).
        """
        # Global residual stays in absolute intensity space (same space as GT).
        # The skip is a noisy bilinear upsample; the network learns the correction.
        skip = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # Input-only standardization: absorbs out-of-range speckle / source shifts
        # without forcing the prediction to inherit degraded mean/std.
        if self.standardize:
            mean, std = self._per_image_stats(x)
            x_in = (x - mean) / std
        else:
            x_in = x

        feats = self.intro(x_in)
        enc_feats = []
        for encoder, down in zip(self.encoders, self.downs):
            feats = encoder(feats)
            enc_feats.append(feats)
            feats = down(feats)

        feats = self.middle(feats)

        for up, decoder, skip_feat in zip(self.ups, self.decoders, reversed(enc_feats)):
            feats = up(feats)
            # handle odd sizes from aggressive downsampling
            if feats.shape[-2:] != skip_feat.shape[-2:]:
                feats = F.interpolate(feats, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)
            feats = feats + skip_feat
            feats = decoder(feats)

        residual = self.up_tail(feats)
        return residual + skip

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: str | dict = "default") -> SpectraRestore:
    """Factory for named presets.

    Presets (approx params):
      default — NAFNet-32 full depth (~29M)  ← ship this
      large   — width 48 full depth (~65M)   quality push if GPU allows
      fast    — width 32 light (~15M)        timing-benchmark fallback
      tiny    — ~2.7M                        smoke tests only
    """
    if isinstance(config, str):
        presets = {
            "default": dict(width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
            "large": dict(width=48, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
            "fast": dict(width=32, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=6, dec_blk_nums=[1, 1, 1, 1]),
            "tiny": dict(width=16, enc_blk_nums=[1, 1, 2, 2], middle_blk_num=4, dec_blk_nums=[1, 1, 1, 1]),
        }
        if config not in presets:
            raise ValueError(f"Unknown preset '{config}'. Choose from {list(presets)}")
        kwargs = presets[config]
    else:
        kwargs = config
    return SpectraRestore(**kwargs)


if __name__ == "__main__":
    for name in ("tiny", "fast", "default"):
        m = build_model(name)
        x = torch.randn(1, 1, 128, 128)
        y = m(x)
        print(f"{name:8s}  params={m.num_params()/1e6:.2f}M  in={tuple(x.shape)} → out={tuple(y.shape)}")
