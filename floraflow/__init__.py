"""FloraFlow: Modular Multi-Camera Vision Flow Matching Pipeline for Robot Manipulation.

Organized into four decoupled stages sharing a common simulation & kinematics core:
1. floraflow.common      : Shared MuJoCo DeskWateringEnv and SE(3) DLS Inverse Kinematics
2. floraflow.collection  : Scripted expert trajectory planner and HDF5 demonstration collector
3. floraflow.training    : Dataset loaders, GPU augmentations, Flow Matching models, and trainer
4. floraflow.evaluation  : Closed-loop MuJoCo benchmark evaluators and 4-layer HUD visualizer
"""

from floraflow import collection, common, evaluation, training

__all__ = [
    "collection",
    "common",
    "evaluation",
    "training",
]
