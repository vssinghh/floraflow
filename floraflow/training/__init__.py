"""Training Pipeline for FloraFlow."""

from floraflow.training.augmentation import RandomShifter
from floraflow.training.dataset import (
    OBS_BASE_KEYS,
    TOTAL_DEMO_HORIZON,
    VisionWateringDataset,
    WateringDemonstrationDataset,
    extract_observation_vector,
)
from floraflow.training.flow_matching import ConditionalFlowMatcher
from floraflow.training.model import FlowMatchingPolicy, ResMlpBlock, SinusoidalPosEmb
from floraflow.training.spatial_softmax import SpatialSoftmax
from floraflow.training.trainer import resolve_compute_device, train_policy, train_vision_policy
from floraflow.training.vision_model import (
    MultiCameraCrossAttention,
    SpatialSoftmaxConvNet,
    VisionFlowMatchingPolicy,
)

__all__ = [
    "ConditionalFlowMatcher",
    "FlowMatchingPolicy",
    "MultiCameraCrossAttention",
    "OBS_BASE_KEYS",
    "RandomShifter",
    "ResMlpBlock",
    "SinusoidalPosEmb",
    "SpatialSoftmax",
    "SpatialSoftmaxConvNet",
    "TOTAL_DEMO_HORIZON",
    "VisionFlowMatchingPolicy",
    "VisionWateringDataset",
    "WateringDemonstrationDataset",
    "extract_observation_vector",
    "resolve_compute_device",
    "train_policy",
    "train_vision_policy",
]
