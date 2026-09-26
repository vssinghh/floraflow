"""Spatial Softmax keypoint extraction layer for vision-based robot manipulation.

Converts 2D convolutional feature maps into expected spatial coordinate pairs
(x, y) in [-1, 1], preserving geometric positions while discarding spatial dimensionality.
"""

from __future__ import annotations

from typing import Tuple

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

    def forward_with_confidence(
        self,
        features: torch.Tensor,
        viz_temperature: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both 2D keypoint coordinates and peak channel confidences.

        Args:
            features: Tensor of shape (B, C, H, W).
            viz_temperature: Optional sharp softmax temperature for diagnostic visualization
                to remove post-GroupNorm background dilution without altering model forward().

        Returns:
            keypoints: Tensor of shape (B, 2 * C) containing [x_0, y_0, x_1, y_1, ...]
                       normalized spatial coordinates in [-1, 1].
            confidence: Tensor of shape (B, C) containing maximum spatial probability
                        per channel in [1 / (H * W), 1.0].
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
        temp = viz_temperature if viz_temperature is not None else (self.temperature.abs() + 1e-4)
        probs = F.softmax(flat_features / temp, dim=-1)

        # Peak probability per channel measures spatial localization sharpness
        confidence = probs.max(dim=-1).values  # (B, C)

        # Expected coordinates: sum(probs * grid)
        expected_x = torch.sum(probs * self.grid_x, dim=-1)  # (B, C)
        expected_y = torch.sum(probs * self.grid_y, dim=-1)  # (B, C)

        # Interleave coordinates: (B, C, 2) -> (B, 2 * C)
        keypoints = torch.stack([expected_x, expected_y], dim=-1).reshape(b, 2 * c)
        return keypoints, confidence

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass converting feature maps into 2D keypoint coordinates.

        Args:
            features: Tensor of shape (B, C, H, W).

        Returns:
            keypoints: Tensor of shape (B, 2 * C) containing [x_0, y_0, x_1, y_1, ...]
                       normalized spatial coordinates in [-1, 1].
        """
        keypoints, _ = self.forward_with_confidence(features)
        return keypoints
