"""Spatial Softmax keypoint extraction layer for vision-based robot manipulation.

Converts 2D convolutional feature maps into expected spatial coordinate pairs
(x, y) in [-1, 1], preserving geometric positions while discarding spatial dimensionality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Spatial Softmax pooling layer.

    Computes the expected 2D spatial coordinates (center of activation mass)
    for each channel of an input feature map of shape (B, C, H, W).
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int,
        temperature: float = 1.0,
    ) -> None:
        """Initialize Spatial Softmax.

        Args:
            height: Height of the input feature map.
            width: Width of the input feature map.
            num_channels: Number of feature channels C.
            temperature: Softmax temperature parameter.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.temperature = nn.Parameter(torch.tensor(float(temperature), dtype=torch.float32))

        # Create coordinate grids normalized to [-1.0, 1.0]
        pos_x = torch.linspace(-1.0, 1.0, steps=width, dtype=torch.float32)
        pos_y = torch.linspace(-1.0, 1.0, steps=height, dtype=torch.float32)

        # Meshgrid: grid_x is (H, W), grid_y is (H, W)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        # Register coordinate buffers (B, C, H*W)
        self.register_buffer("grid_x", grid_x.reshape(1, 1, -1))
        self.register_buffer("grid_y", grid_y.reshape(1, 1, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass converting feature maps into 2D keypoint coordinates.

        Args:
            features: Tensor of shape (B, C, H, W).

        Returns:
            keypoints: Tensor of shape (B, 2 * C) containing [x_0, y_0, x_1, y_1, ...]
                       normalized spatial coordinates in [-1, 1].
        """
        b, c, h, w = features.shape
        assert h == self.height and w == self.width, (
            f"Expected feature map of size ({self.height}, {self.width}), got ({h}, {w})"
        )
        assert c == self.num_channels, (
            f"Expected {self.num_channels} channels, got {c}"
        )

        # Flatten spatial dimensions: (B, C, H * W)
        flat_features = features.reshape(b, c, -1)

        # Compute spatial softmax probabilities over H * W
        temp = self.temperature.abs() + 1e-4
        probs = F.softmax(flat_features / temp, dim=-1)

        # Expected coordinates: sum(probs * grid)
        expected_x = torch.sum(probs * self.grid_x, dim=-1)  # (B, C)
        expected_y = torch.sum(probs * self.grid_y, dim=-1)  # (B, C)

        # Interleave coordinates: (B, C, 2) -> (B, 2 * C)
        keypoints = torch.stack([expected_x, expected_y], dim=-1).reshape(b, 2 * c)
        return keypoints
