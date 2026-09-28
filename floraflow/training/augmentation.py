"""Visual data augmentation operations for robot imitation learning.

Provides GPU-accelerated spatial shift augmentation using affine grid sampling
to encourage translational invariance in convolutional backbones without external dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShifter(nn.Module):
    """Applies random translation shifts of up to max_shift pixels on (B, C, H, W) images."""

    def __init__(self, max_shift: int = 4) -> None:
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random spatial shift to image tensor.

        Args:
            x: Image tensor of shape (B, C, H, W).

        Returns:
            shifted: Translated image tensor of shape (B, C, H, W).
        """
        if self.max_shift <= 0:
            return x

        b, c, h, w = x.shape
        device = x.device

        # Sample pixel offsets dx, dy in [-max_shift, max_shift]
        delta_x = (torch.randint(-self.max_shift, self.max_shift + 1, (b, 1), device=device) * (2.0 / w)).float()
        delta_y = (torch.randint(-self.max_shift, self.max_shift + 1, (b, 1), device=device) * (2.0 / h)).float()

        # Build 2x3 affine matrix for translation
        theta = torch.zeros(b, 2, 3, device=device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = delta_x.squeeze(-1)
        theta[:, 1, 2] = delta_y.squeeze(-1)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
